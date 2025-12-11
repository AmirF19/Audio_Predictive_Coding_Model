"""
GPU-Accelerated Predictive Coding Model for Auditory Word Recognition

Implementing hierarchical predictive coding for auditory N400 experiments.
Based on Samer's model but adapted for auditory input instead of orthographic.

BASIC IDEA:
    - 4 layers: Audio → Lexical → Semantic → Context
    - Each layer predicts the layer below it (top-down)
    - Prediction errors flow upward (bottom-up)
    - Model "settles" when predictions match input
    
LAYERS:
    Audio (bottom): 
        Our phoneme vectors (10 slots x 40 features = 400 dims, so 10-hot encoding)
        Each phoneme slot picks one feature out of 40 (1-hot per slot)
        Words <10 phonemes get padded with empty slots
        NOTE: This replaces Samer's orthographic layer (4 letters x 26 = 104 dims)
        
    Lexical: 
        One unit per word in our lexicon (~800 words)
        Gets bottom-up from audio, sends top-down predictions back
        
    Semantic: 
        Distributed semantic features (we have ~20k features)
        TODO: Check if this many features causes issues vs Samer's
        
    Context (top): 
        High-level concepts
        Can clamp this for priming experiments (optional)

HOW IT WORKS (simplified):
    Each iteration:
    1. Bottom-up pass: compute prediction errors (PE = state / reconstruction)
    2. Update states: blend current state with multiplicative update (adaptive momentum)
    3. Top-down pass: higher layers predict what lower layers should be
    4. Repeat until it settles (~20 iterations usually enough)
    
    Key features:
    - MULTIPLICATIVE updates (not additive) keep activations positive
    - ADAPTIVE MOMENTUM: 
      * During input (prime/target): momentum=0.7 (stable, prevents oscillations)
      * During blanks (ISI): momentum=0.3 (fast decay to low baseline)
      This solves high-baseline issue during ISI and oscillations during same-word repetition
    - Epsilon guards prevent numerical instability

REFERENCE:
    Samer Nour Eddine's predictive coding model (2024)
    His: 4-letter orthographic input (4-hot)
    Ours: 10-phoneme auditory input (10-hot)

AUTHORS:
    Muhammad Fusening, Alba Jorquera, William Zumchak
"""

import torch
import numpy as np



# DEVICE SELECTION

def get_device():
    """Check if we have a GPU available (way faster than CPU for this)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU (will be slow!)")
    return device



# MAIN MODEL CLASS


class AuditoryPCModelGPU:
    """
    Main predictive coding model class - handles the hierarchical processing.
    
    # Origin note:
    # This class mirrors Samer Nour Eddine's Simulation dynamics (PredictiveCoding_Model.py):
    # - Same 4-layer hierarchy and state/tdR/tdB/PE slots
    # - Same epsilon-guarded multiplicative updates
    # Adaptations (ours):
    # - Audio layer replaces orthographic layer (10-slot phoneme input)
    # - GPU tensors instead of numpy arrays
    # - Adaptive momentum to prevent oscillations / speed decay
    # - Optional precision scaling (clean vs noisy)
    # - Semantic PE normalization (feature-count scaling)
    # - Batched processing helper

    STRUCTURE:
        Each layer has 4 things stored in statespace[layer][index]:
            [0] = state (ST): what's currently active
            [1] = reconstruction (tdR): what the layer above predicts this should be
            [2] = top-down bias (tdB): reconstruction/state (used in top-down mode)
            [3] = prediction error (PE): state/reconstruction (this drives everything!)
    
    BASIC LOOP each iteration:
        1. Clamp audio input to sensory data
        2. Compute audio PE = input / top-down prediction
        3. Lexical layer updates based on audio PE (weighted by connections)
        4. Compute lexical PE
        5. Semantic layer updates based on lexical PE
        6. Compute semantic PE  
        7. Context layer accumulates semantic info
        8. Generate top-down predictions (reconstructions) for next iteration
    
    IMPORTANT BITS:
        - Uses multiplicative updates (state = state * update), not additive (state = state + update)
          This keeps everything positive and implements a kind of normalization
        - Epsilon guards prevent divide-by-zero and keep states from collapsing to 0
        - Weight matrices are normalized (following Samer's scheme exactly)
        - Frequency bias added to audio→lexical weights (high-frequency words easier to recognize)
    
    TODO: Double-check that epsilon values match Samer's exactly
    """
    
    def __init__(self, lexicon_words, audio_matrix, semantic_matrix, 
                 EPSILON1=0.005, EPSILON2=0.0001, frequency_bias=None,
                 device=None):
        """
        Initialize the predictive coding model.
        
        Args:
            lexicon_words: List of words in the lexicon (e.g., 800 words)
            audio_matrix: (audio_dim x n_words) array of phoneme vectors
                         Each column is a 5-hot vector (400 dims) for one word
            semantic_matrix: (n_features x n_words) binary feature matrix
                           Each column indicates which features a word has
            EPSILON1: Minimum denominator for division (prevents divide-by-zero)
            EPSILON2: Minimum value for multiplicative updates (prevents zero states)
            frequency_bias: Optional (n_words,) array of log-frequency priors
            device: torch.device (cuda or cpu); auto-detected if None
        """
        # Store device and hyperparameters
        self.device = device if device else get_device()
        self.EPSILON1 = EPSILON1  # Guards division operations (0.005 from Samer)
        self.EPSILON2 = EPSILON2  # Guards multiplicative updates (0.0001 from Samer)
        
        # Momentum parameter for state updates
        # ISSUE: Pure multiplicative updates can oscillate when state ≈ reconstruction
        #        (especially for same-word repetition where PE should be near zero)
        # FIX: Blend current state with proposed update: new = α*old + (1-α)*proposed
        # JUSTIFICATION: Neural adaptation/inertia - states don't change instantaneously
        # LITERATURE: Common in recurrent predictive coding models (Rao & Ballard 1999)
        self.MOMENTUM_INPUT = 0.7    # During input presentation (slow, stable updates)
        self.MOMENTUM_BLANK = 0.3    # During blank/ISI (fast decay to low baseline)
        
        # Store lexicon information
        self.words = np.array(lexicon_words)
        self.lexicon_size = len(lexicon_words)
        
        # Store dimensionality of each level
        self.audio_dim = audio_matrix.shape[0]     # e.g., 400 (phoneme features)
        self.sem_dim = semantic_matrix.shape[0]    # e.g., 20533 (semantic features)
        
        # Convert input matrices to GPU tensors
        self.audio_matrix = torch.tensor(audio_matrix, dtype=torch.float32, device=self.device)
        self.semantic_matrix = torch.tensor(semantic_matrix, dtype=torch.float32, device=self.device)
        
        # Store frequency bias if provided
        if frequency_bias is not None:
            self.frequency_bias = torch.tensor(frequency_bias, dtype=torch.float32, device=self.device)
        else:
            self.frequency_bias = None
        
        # Build connection weight matrices
        self._build_weights()
        
        # Statespace will be initialized when simulation starts
        self.statespace = None
        self.current_iteration = 0
        
    # ========================================================================
    # WEIGHT MATRIX CONSTRUCTION
    # ========================================================================
        
    def _build_weights(self):
        """
        Build all the connection weights between layers.
        
        This normalization scheme is a bit weird but following Samer's approach exactly:
        (Reference: Samer PredictiveCoding_Model.py define_weights, lines ~82-110)
        
        For each connection (e.g. Audio→Lexical):
        1. Stack the connection matrix with an identity matrix: [W, I]
        2. Sum across columns to get normalization factors
        3. Divide original W by these factors
        
        Why? This keeps the total input to each unit controlled - prevents
        activations from blowing up or collapsing. The identity part ensures
        even units with no bottom-up input get some normalization.
        
        FREQUENCY BIAS:
        After normalizing, we add frequency to the audio→lexical weights.
        This makes high-frequency words easier to activate (like in humans).
        Only add it to non-zero connections (masking).
        
        OUTPUTS:
        Creates bidirectional weights for each layer pair:
        - A_to_L & L_to_A: audio ↔ lexical
        - L_to_S & S_to_L: lexical ↔ semantic  
        - S_to_C & C_to_S: semantic ↔ context
        Plus identity matrices I1, I2 for top-down bias terms
        
        NOTE: Top-down weights are just transposes of bottom-up (symmetric)
        """
        self.weights = {}
        
        # ---- AUDIO → LEXICAL ----
        # Samer analog: O_to_L; here adapted to audio (phoneme slots)
        # Which phoneme features predict which words?
        # Transpose so rows=words, cols=audio_features
        A = self.audio_matrix.T  # (n_words, audio_dim)
        
        # Normalization trick from Samer: stack [A, I] then divide by column sums
        # Not totally sure why the identity is needed but keeping it for consistency
        block_A = torch.cat([A, torch.eye(self.lexicon_size, device=self.device)], dim=1)
        ones_A = torch.ones((self.lexicon_size + self.audio_dim, 1), device=self.device)
        self.weights['divide_wt_A_to_L'] = torch.mm(block_A, ones_A)
        
        # Actual normalization step
        self.weights['A_to_L'] = A / self.weights['divide_wt_A_to_L']
        
        # Add frequency bias: common words should be easier to activate
        # Only add to existing connections (use mask)
        if self.frequency_bias is not None:
            freq = self.frequency_bias.reshape(-1, 1)
            mask = (self.audio_matrix.T > 0).float()  # Binary: is there a connection?
            self.weights['A_to_L'] = (self.weights['A_to_L'] + freq) * mask
        
        # Top-down weights = just transpose (symmetric connections)
        self.weights['L_to_A'] = self.weights['A_to_L'].T
        
        # ---- LEXICAL → SEMANTIC MAPPING ----
        # Connection matrix: which semantic features belong to which words
        self.weights['L_to_S'] = self.semantic_matrix.clone()
        
        # Normalization
        block_LS = torch.cat([self.weights['L_to_S'], 
                              torch.eye(self.sem_dim, device=self.device)], dim=1)
        ones_LS = torch.ones((self.lexicon_size + self.sem_dim, 1), device=self.device)
        self.weights['divide_wt_L_to_S'] = torch.mm(block_LS, ones_LS)
        self.weights['L_to_S'] = self.weights['L_to_S'] / self.weights['divide_wt_L_to_S']
        
        # Top-down prediction
        self.weights['S_to_L'] = self.weights['L_to_S'].T
        
        # ---- SEMANTIC → CONTEXTUAL MAPPING ----
        # Connection matrix: transpose of semantic matrix (features→concepts)
        self.weights['S_to_C'] = self.semantic_matrix.T.clone()
        
        # Normalization
        ones_SC = torch.ones((self.sem_dim, 1), device=self.device)
        self.weights['divide_wt_S_to_C'] = torch.mm(self.weights['S_to_C'], ones_SC)
        self.weights['S_to_C'] = self.weights['S_to_C'] / self.weights['divide_wt_S_to_C']
        
        # Top-down prediction
        self.weights['C_to_S'] = self.weights['S_to_C'].T
        
        # ---- IDENTITY MATRICES FOR TOP-DOWN BIAS ----
        # These allow top-down signals to modulate states directly
        # (used in top-down mode experiments, not in our bottom-up paradigm)
        self.I1 = torch.eye(self.lexicon_size, device=self.device) / self.weights['divide_wt_A_to_L']
        self.I2 = torch.eye(self.sem_dim, device=self.device) / self.weights['divide_wt_L_to_S']
    
    # ========================================================================
    # STATE INITIALIZATION
    # ========================================================================
        
    def reset_state(self, n_trials=1):
        """
        Reset the model for a new simulation (clear all activations).
        
        Everything starts as uniform distributions: ones / n_units
        This means initially all words/features are equally likely.
        (Reference: Samer define_statespace, lines ~69-80)
        
        Biases (index [2]) start at zero - they'll build up from top-down signals.
        Context layer is special: only has state, no reconstruction/bias/PE.
        
        Args:
            n_trials: How many trials to process in parallel (for batching)
        """
        self.statespace = {
            # Audio level: uniform over phoneme features
            'audio': torch.ones((4, self.audio_dim, n_trials), 
                               device=self.device) / self.audio_dim,
            
            # Lexical level: uniform over words
            'lex': torch.ones((4, self.lexicon_size, n_trials), 
                             device=self.device) / self.lexicon_size,
            
            # Semantic level: uniform over features
            'sem': torch.ones((4, self.sem_dim, n_trials), 
                             device=self.device) / self.sem_dim,
            
            # Contextual level: uniform over concepts
            'ctx': torch.ones((4, self.lexicon_size, n_trials), 
                             device=self.device) / self.lexicon_size
        }
        
        # Context has no reconstruction, bias, or PE (top-level representation)
        self.statespace['ctx'][1:, :, :] = float('nan')
        
        # Initialize biases to zero (will build up from top-down predictions)
        self.statespace['lex'][2, :, :] = 0
        self.statespace['sem'][2, :, :] = 0
        
        self.current_iteration = 0
    
    # ========================================================================
    # EPSILON-GUARDED OPERATIONS
    # ========================================================================
    
    def _eps_div(self, x, y):
        """
        Safe division: x / max(eps, y)
        
        Prevents divide-by-zero when y gets too small.
        EPSILON1 = 0.005 (from Samer's model)
        (Reference: Samer eps_div, lines ~112-115)
        """
        return x / torch.clamp(y, min=self.EPSILON1)
    
    def _eps_mul(self, x, y):
        """
        Safe multiplication: max(eps, x) * y
        
        Prevents x from collapsing to exactly zero (would kill all signal).
        EPSILON2 = 0.0001 (from Samer's model)
        Basically adds a tiny floor to keep things alive.
        (Reference: Samer eps_mul, lines ~112-115)
        """
        return torch.clamp(x, min=self.EPSILON2) * y
    
    # ========================================================================
    # CORE DYNAMICS: ONE ITERATION
    # ========================================================================
        
    def run_one_iteration(self, audio_input, mode="bottom_up", ctx_clamp=None, 
                         precision_audio=None):
        """
        Run one iteration of the model (one time step).
        
        Structure follows Samer's run_one_iteration (PredictiveCoding_Model.py):
        - Same ordering: clamp input -> PE_audio -> lexical update -> PE_lex -> semantic update -> PE_sem -> context update -> reconstructions.
        Additions (ours):
        - Precision scaling on audio PE (clean vs noisy)
        - Adaptive momentum (stability, fast ISI decay)
        - Audio layer replaces orthographic layer
        (Reference: Samer run_one_iteration, lines ~111-160)
        BASIC FLOW:
            1. Clamp audio to input (bottom-up mode)
            2. Compute prediction error at audio layer  
            3. Update lexical states based on audio PE
            4. Compute lexical PE
            5. Update semantic states based on lexical PE
            6. Compute semantic PE
            7. Update context based on semantic PE
            8. Generate top-down predictions for next iteration
        
        EXPERIMENTAL STUFF:
            - precision_audio: weight the audio PE differently per trial
              (we use this for clean vs noisy - noisy gets higher PE weight)
            - ctx_clamp: force context to stay at prime word during target
              (optional priming mechanism, usually off)
        
        Args:
            audio_input: The sensory input (10-hot phoneme vector)
            mode: "bottom_up" is normal, "top_down" for control experiments  
            ctx_clamp: Force context activation (optional)
            precision_audio: Per-trial weights for audio PE (optional)
        """
        # ---- TYPE CONVERSION ----
        # Ensure all inputs are torch tensors on correct device
        if isinstance(audio_input, np.ndarray):
            audio_input = torch.tensor(audio_input, dtype=torch.float32, device=self.device)
        
        if ctx_clamp is not None and isinstance(ctx_clamp, np.ndarray):
            ctx_clamp = torch.tensor(ctx_clamp, dtype=torch.float32, device=self.device)

        # Default precision to ones if not provided
        if precision_audio is None:
            precision_audio_t = torch.ones((audio_input.shape[1],), device=self.device)
        else:
            if isinstance(precision_audio, np.ndarray):
                precision_audio_t = torch.tensor(precision_audio, dtype=torch.float32, device=self.device)
            else:
                precision_audio_t = precision_audio.to(self.device)
        
        # ---- ADAPTIVE MOMENTUM ----
        # Detect if input is blank (ISI period) vs actual input
        # Blank: use low momentum (fast decay to baseline)
        # Input: use high momentum (stable processing)
        is_blank = torch.sum(torch.abs(audio_input)) < 0.01  # Blank if all zeros
        momentum = self.MOMENTUM_BLANK if is_blank else self.MOMENTUM_INPUT
        
        # ======== AUDIO LAYER ========
        # In bottom-up mode, directly clamp state to sensory input
        # (matches reference model's orthographic clamping)
        if mode == "bottom_up":
            self.statespace['audio'][0] = audio_input
        else:
            # Top-down mode: state updates via bias (not used in our experiments)
            self.statespace['audio'][0] = self._eps_mul(
                self.statespace['audio'][0], 
                self.statespace['audio'][2]
            )
        
        # Compute audio prediction error: sensory input vs top-down prediction
        # PE = state / reconstruction (higher when mismatch is large)
        self.statespace['audio'][3] = self._eps_div(
            self.statespace['audio'][0],
            self.statespace['audio'][1]
        )
        
        # ** EXPERIMENTAL: Precision scaling for noisy manipulation **
        # Multiply audio PE by precision weight (1.0 for clean, 1.21 for noisy)
        # Idea: noisy input = harder to process = more PE
        # Based on measured clean-noisy similarity (0.575)
        self.statespace['audio'][3] = self.statespace['audio'][3] * precision_audio_t.unsqueeze(0)
        
        # Compute top-down bias (for top-down mode, not used here)
        self.statespace['audio'][2] = self._eps_div(
            self.statespace['audio'][1],
            self.statespace['audio'][0]
        )
        
        # ======== LEXICAL LAYER ========
        # State update: combine bottom-up PE and top-down bias
        # Bottom-up: A_to_L @ audio_PE propagates phoneme errors to words
        # Top-down: I1 @ lex_bias allows semantic predictions to constrain lexical states
        lex_update_term = (
            torch.mm(self.weights['A_to_L'], self.statespace['audio'][3]) +
            torch.mm(self.I1, self.statespace['lex'][2])
        )
        
        # Multiplicative update with momentum dampening
        # Adaptive momentum: fast decay during blanks, stable during input
        # Blank (ISI): momentum=0.3 → fast decay to low baseline
        # Input: momentum=0.7 → stable processing, prevents oscillations
        old_lex_state = self.statespace['lex'][0].clone()
        proposed_lex_state = self._eps_mul(old_lex_state, lex_update_term)
        self.statespace['lex'][0] = (
            momentum * old_lex_state + 
            (1 - momentum) * proposed_lex_state
        )
        
        # Compute lexical prediction error
        self.statespace['lex'][3] = self._eps_div(
            self.statespace['lex'][0],
            self.statespace['lex'][1]
        )
        
        # Compute lexical top-down bias
        self.statespace['lex'][2] = self._eps_div(
            self.statespace['lex'][1],
            self.statespace['lex'][0]
        )
        
        # ======== SEMANTIC LAYER ========
        # State update: combine bottom-up lexical PE and top-down context bias
        sem_update_term = (
            torch.mm(self.weights['L_to_S'], self.statespace['lex'][3]) +
            torch.mm(self.I2, self.statespace['sem'][2])
        )
        
        # Adaptive momentum (same as lexical layer)
        old_sem_state = self.statespace['sem'][0].clone()
        proposed_sem_state = self._eps_mul(old_sem_state, sem_update_term)
        self.statespace['sem'][0] = (
            momentum * old_sem_state +
            (1 - momentum) * proposed_sem_state
        )
        
        # Compute semantic prediction error
        self.statespace['sem'][3] = self._eps_div(
            self.statespace['sem'][0],
            self.statespace['sem'][1]
        )
        
        # Compute semantic top-down bias
        self.statespace['sem'][2] = self._eps_div(
            self.statespace['sem'][1],
            self.statespace['sem'][0]
        )
        
        # ======== CONTEXTUAL LAYER ========
        # Context accumulates semantic PE (top-level concept extraction)
        ctx_update_term = torch.mm(self.weights['S_to_C'], self.statespace['sem'][3])
        
        if ctx_clamp is not None:
            # Experimental: clamp context to prime word identity
            # (used in some priming paradigms, not in reference model)
            self.statespace['ctx'][0] = ctx_clamp
        else:
            # Standard: context evolves via semantic PE (with adaptive momentum)
            old_ctx_state = self.statespace['ctx'][0].clone()
            proposed_ctx_state = self._eps_mul(old_ctx_state, ctx_update_term)
            self.statespace['ctx'][0] = (
                momentum * old_ctx_state +
                (1 - momentum) * proposed_ctx_state
            )
        
        # ======== TOP-DOWN RECONSTRUCTIONS ========
        # Higher levels generate predictions (reconstructions) for lower levels
        # These will be used to compute PE in the next iteration
        self.statespace['audio'][1] = torch.mm(self.weights['L_to_A'], self.statespace['lex'][0])
        self.statespace['lex'][1] = torch.mm(self.weights['S_to_L'], self.statespace['sem'][0])
        self.statespace['sem'][1] = torch.mm(self.weights['C_to_S'], self.statespace['ctx'][0])
        
        self.current_iteration += 1
    
    # ========================================================================
    # SUMMARY METRICS
    # ========================================================================
    
    def get_total_lexsem_error(self):
        """
        Compute N400: sum of lexical + semantic prediction error.
        
        IDEA: N400 in EEG = semantic integration difficulty
        Here: operationalize as total PE (how much input mismatches predictions)
        
        NORMALIZATION ISSUE:
            We have 20,533 semantic features, Samer had 3,715.
            If we just sum all semantic PE, it'll be ~5.5x too big!
            Solution: scale semantic PE by (3715 / 20533) = 0.181
            This brings peaks to ~400 range (matching Samer's model)
            
        TODO: Double-check this doesn't mess up condition differences
        (it shouldn't - just scales everything equally)
        
        Returns:
            N400 value for each trial in the batch
        """
        # NOTE: Samer uses raw sums (no scaling). We scale semantic PE to account
        # for larger feature space so magnitudes are comparable.
        # Reference: Samer get_summary total_lexsem_err, lines ~43 in get_summary.py
        SAMER_N_SEMANTIC_FEATURES = 3715  # From his model
        
        # Sum all lexical PEs
        total_lex_err = torch.sum(self.statespace['lex'][3], dim=0)
        
        # Sum all semantic PEs  
        total_sem_err = torch.sum(self.statespace['sem'][3], dim=0)
        
        # Scale semantic to match Samer's feature space size
        semantic_scale = SAMER_N_SEMANTIC_FEATURES / self.sem_dim  # ~0.181
        
        return (total_lex_err + total_sem_err * semantic_scale).cpu().numpy()
    
    def get_total_lex_error(self):
        """Get total lexical prediction error (for decomposition analyses)."""
        return torch.sum(self.statespace['lex'][3], dim=0).cpu().numpy()
    
    def get_total_sem_error(self):
        """Get total semantic prediction error (for decomposition analyses)."""
        # Note: NOT normalized (for raw analysis)
        return torch.sum(self.statespace['sem'][3], dim=0).cpu().numpy()
    
    def get_max_lex_state_activation(self):
        """Get activation of most active lexical unit (winner activation)."""
        return torch.max(self.statespace['lex'][0], dim=0)[0].cpu().numpy()
    
    def get_max_lex_state_identity(self):
        """Get index of most active lexical unit (winner identity)."""
        return torch.argmax(self.statespace['lex'][0], dim=0).cpu().numpy()
    
    def get_total_lex_state(self):
        """Get total lexical activation (for normalization checks)."""
        return torch.sum(self.statespace['lex'][0], dim=0).cpu().numpy()
    
    def get_summary(self):
        """
        Extract all summary metrics at current iteration.
        
        Returns dictionary with N400, activations, and identities.
        """
        return {
            'max_lex_state_activation': self.get_max_lex_state_activation(),
            'total_lex_state': self.get_total_lex_state(),
            'total_lexsem_err': self.get_total_lexsem_error(),
            'total_lex_err': self.get_total_lex_error(),
            'total_sem_err': self.get_total_sem_error(),
            'max_lex_state_identity': self.get_max_lex_state_identity(),
        }


# BATCHED MODEL (FOR PARALLEL TRIAL PROCESSING)

class BatchedAuditoryPCModelGPU(AuditoryPCModelGPU):
    """
    Batched version - runs multiple trials in parallel on GPU (in like 10 seconds).
    
    WHY: We have ~800 prime-target pairs to test. Running them one-by-one
    would take forever. Instead, process 256 at a time in parallel on GPU.
    
    USAGE: Same init as base class, just call run_batch_trials() instead
    of manually looping.
    """
    
    def run_batch_trials(self, prime_vecs, target_vecs, prime_indices,
                         prime_iters=20, blank_iters=5, target_iters=20,
                         post_target_iters=0, use_ctx_clamp=True, precision_audio=None):
        """
        Run a bunch of prime-target trials in parallel.
        
        TIMELINE for each trial:
            1. Prime phase (~20 iterations): show prime word, let model settle
            2. Blank (ISI, ~5 iterations): nothing, just decay
            3. Target phase (~20 iterations): show target word
            4. Post-target (~5 iterations): more decay, optional
        
        We extract N400 from iterations 2-11 of target phase
        (corresponds roughly to 300-500ms in real EEG)
        
        Args:
            prime_vecs, target_vecs: Phoneme vectors for each trial
            prime_indices: Which word each prime is (for context clamp)
            prime_iters, blank_iters, target_iters, post_target_iters: Timeline
            use_ctx_clamp: Keep context fixed to prime during target? (usually False)
            precision_audio: Per-trial PE weights (for clean vs noisy)
        
        Returns:
            Dict with PE traces, activations, etc. for all trials
        """
        n_trials = prime_vecs.shape[1]
        
        # Convert inputs to GPU tensors
        prime_input = torch.tensor(prime_vecs, dtype=torch.float32, device=self.device)
        target_input = torch.tensor(target_vecs, dtype=torch.float32, device=self.device)
        blank_input = torch.zeros_like(prime_input)
        post_input = blank_input  # Same as blank (zero input)

        # Handle precision weights
        precision_audio_t = None
        if precision_audio is not None:
            if isinstance(precision_audio, np.ndarray):
                precision_audio_t = torch.tensor(precision_audio, dtype=torch.float32, device=self.device)
            else:
                precision_audio_t = precision_audio.to(self.device)
        
        # Reset model statespace for all trials at once
        self.reset_state(n_trials=n_trials)
        
        # Storage for timecourse data
        all_lexsem_err = []
        all_lex_err = []
        all_sem_err = []
        all_max_activation = []
        all_max_identity = []
        
        # ---- PHASE 1: PRIME ----
        for _ in range(prime_iters):
            self.run_one_iteration(prime_input, precision_audio=precision_audio_t)
            all_lexsem_err.append(self.get_total_lexsem_error())
            all_lex_err.append(self.get_total_lex_error())
            all_sem_err.append(self.get_total_sem_error())
            all_max_activation.append(self.get_max_lex_state_activation())
            all_max_identity.append(self.get_max_lex_state_identity())
        
        # ---- PHASE 2: BLANK (ISI) ----
        for _ in range(blank_iters):
            self.run_one_iteration(blank_input, precision_audio=precision_audio_t)
            all_lexsem_err.append(self.get_total_lexsem_error())
            all_lex_err.append(self.get_total_lex_error())
            all_sem_err.append(self.get_total_sem_error())
            all_max_activation.append(self.get_max_lex_state_activation())
            all_max_identity.append(self.get_max_lex_state_identity())
        
        # Prepare context clamp for target phase (if requested)
        # Clamps context layer to prime word identity (simulates sustained prime activation)
        ctx_clamp = None
        if use_ctx_clamp:
            ctx_clamp = torch.zeros((self.lexicon_size, n_trials), dtype=torch.float32, device=self.device)
            for trial_idx, prime_idx in enumerate(prime_indices):
                ctx_clamp[prime_idx, trial_idx] = 1.0
        
        # ---- PHASE 3: TARGET ----
        for _ in range(target_iters):
            self.run_one_iteration(target_input, ctx_clamp=ctx_clamp, precision_audio=precision_audio_t)
            all_lexsem_err.append(self.get_total_lexsem_error())
            all_lex_err.append(self.get_total_lex_error())
            all_sem_err.append(self.get_total_sem_error())
            all_max_activation.append(self.get_max_lex_state_activation())
            all_max_identity.append(self.get_max_lex_state_identity())

        # ---- PHASE 4: POST-TARGET SETTLING ----
        for _ in range(post_target_iters):
            self.run_one_iteration(post_input, precision_audio=precision_audio_t)
            all_lexsem_err.append(self.get_total_lexsem_error())
            all_lex_err.append(self.get_total_lex_error())
            all_sem_err.append(self.get_total_sem_error())
            all_max_activation.append(self.get_max_lex_state_activation())
            all_max_identity.append(self.get_max_lex_state_identity())
        
        # Stack results: (n_iters, n_trials) → transpose to (n_trials, n_iters)
        # This format allows easy per-trial indexing
        return {
            'total_lexsem_err': np.array(all_lexsem_err).T,
            'total_lex_err': np.array(all_lex_err).T,
            'total_sem_err': np.array(all_sem_err).T,
            'max_lex_state_activation': np.array(all_max_activation).T,
            'max_lex_state_identity': np.array(all_max_identity).T,
        }
