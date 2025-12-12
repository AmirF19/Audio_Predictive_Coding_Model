"""
BASIC IDEA:
    - 4 layers: Audio → Lexical → Semantic → Context
    - Each layer predicts the layer below it (top-down)
    - Prediction errors flow upward (bottom-up)
    - Model "settles" when predictions match input
    
LAYERS:
    Audio (bottom): 
        Our phoneme vectors (10 slots x 40 features = 400 dims, so 10-hot encoding)
        Words <10 phonemes get padded with empty slots
    Lexical: 
        One unit per word in our lexicon (~800 words)
    Semantic:
        Distributed semantic features (we have ~20k features)
    Conceptual (top): 
        Words

AUTHORS:
    Muhammad Fusening, Alba Jorquera, William Zumchak
"""

import torch
import numpy as np


def get_device():
    """Check if we have a GPU available (way faster than CPU for this)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU (will be slow!)")
    return device

class AuditoryPCModelGPU:
    """
    Based on Samer Nour Eddine's model (PredictiveCoding_Model.py).
    Same core structure: 4-layer hierarchy, etc.
    Adaptations:
    - Audio layer instead of orthographic (10-slot phonemes vs 4-letter slots)
    - GPU updates (runs in like 10 seconds)

    STRUCTURE:
    WHAT EACH LAYER TRACKS (statespace[layer][index]):
        [0] = state (ST): what's currently active in this layer
        [1] = reconstruction (tdR): what the layer above predicts this should be
        [2] = top-down bias (tdB): how to adjust based on predictions (mainly for experiments)
        [3] = prediction error (PE): how wrong we are - this becomes the N400 signal!

    LOOP for each iteration:
        1. Feed audio input directly to the bottom layer
        2. Calculate how wrong the audio prediction is (PE = input ÷ prediction)
        3. Let lexical layer adjust based on audio errors
        4. Check lexical prediction errors
        5. Update semantic layer based on lexical errors
        6. Calculate semantic prediction errors
        7. Context layer builds up from semantic patterns
        8. Generate predictions for what should happen next
    """
    
    def __init__(self, lexicon_words, audio_matrix, semantic_matrix, 
                 EPSILON1=0.005, EPSILON2=0.0001, frequency_bias=None,
                 device=None):
        """
        Initialize the predictive coding model.
        
        Args:
            lexicon_words: List of words in the lexicon (e.g., 800 words)
            audio_matrix: My phoneme features (400 x n_words for 10-slot encoding)
            semantic_matrix: Semantic features for each word (20k+ binary features per word)
            EPSILON1: Safety margin for divisions (0.005 prevents crashes)
            EPSILON2: Safety margin for updates (0.0001 keeps states alive)
            frequency_bias: Optional word frequency weights (helps common words)
        """
        # Store device and hyperparameters
        self.device = device if device else get_device()
        self.EPSILON1 = EPSILON1  # Guards division operations (0.005 from Samer)
        self.EPSILON2 = EPSILON2  # Guards multiplicative updates (0.0001 from Samer)
        
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
        Set up all the weight connections between layers.

        I follow Samer's exact normalization scheme (PredictiveCoding_Model.py lines ~82-110).
        For each connection (like Audio→Lexical):
        1. Combine the weight matrix with an identity matrix
        2. Calculate normalization factors from the column sums
        3. Scale the weights to keep total inputs reasonable

        This prevents any single unit from getting overwhelmed by inputs - keeps everything balanced and prevents
        activations from blowing up or collapsing. The identity part ensures
        even units with no bottom-up input get some normalization.
        
        FREQUENCY BIAS:
        After normalizing, we add frequency to the audio→lexical weights.
        This makes high-frequency words easier to activate (like in humans).
        Only add it to non-zero connections (masking).
        
        OUTPUTS:
        Creates bidirectional weights for each layer pair:
        - A_to_L & L_to_A: audio to lexical
        - L_to_S & S_to_L: lexical to semantic  
        - S_to_C & C_to_S: semantic to concept
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
        """
        self.statespace = {
            'audio': torch.ones((4, self.audio_dim, n_trials), 
                               device=self.device) / self.audio_dim,
            'lex': torch.ones((4, self.lexicon_size, n_trials), 
                             device=self.device) / self.lexicon_size,
            'sem': torch.ones((4, self.sem_dim, n_trials), 
                             device=self.device) / self.sem_dim,
            'cpt': torch.ones((4, self.lexicon_size, n_trials), 
                             device=self.device) / self.lexicon_size
        }
        
        self.statespace['cpt'][1:, :, :] = float('nan')
        
        self.statespace['lex'][2, :, :] = 0
        self.statespace['sem'][2, :, :] = 0
        
        self.current_iteration = 0
    
    def _eps_div(self, x, y):
        return x / torch.clamp(y, min=self.EPSILON1)
    
    def _eps_mul(self, x, y):
        return torch.clamp(x, min=self.EPSILON2) * y
    
    # RUNNING MODEL
        
    def run_one_iteration(self, audio_input, mode="bottom_up", cpt_clamp=None):
        """
        Run one iteration of the model (one time step).
        
        Structure follows Samer's run_one_iteration (PredictiveCoding_Model.py):
        - Same ordering: clamp input -> PE_audio -> lexical update -> PE_lex -> semantic update -> PE_sem -> concept update -> reconstructions.
        Additions (ours):
        - Audio layer replaces orthographic layer
        (Reference: Samer run_one_iteration, lines ~111-160)
        BASIC FLOW:
            1. Clamp audio to input (bottom-up mode)
            2. Compute prediction error at audio layer  
            3. Update lexical states based on audio PE
            4. Compute lexical PE
            5. Update semantic states based on lexical PE
            6. Compute semantic PE
            7. Update concept based on semantic PE
            8. Generate top-down predictions for next iteration
        
        Args:
            audio_input: Phoneme features for this time step (10-hot binaryvector)
            mode: Usually "bottom_up" (normal processing), "top_down" for special experiments
            cpt_clamp: Concept layer override (forces specific concept activation)
        """
        # ---- TYPE CONVERSION ----
        # Ensure all inputs are torch tensors on correct device
        if isinstance(audio_input, np.ndarray):
            audio_input = torch.tensor(audio_input, dtype=torch.float32, device=self.device)
        
        if cpt_clamp is not None and isinstance(cpt_clamp, np.ndarray):
            cpt_clamp = torch.tensor(cpt_clamp, dtype=torch.float32, device=self.device)

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
        
        # Multiplicative update (Samer style)
        self.statespace['lex'][0] = self._eps_mul(
            self.statespace['lex'][0],
            lex_update_term
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
            # State update: combine bottom-up lexical PE and top-down concept bias
        sem_update_term = (
            torch.mm(self.weights['L_to_S'], self.statespace['lex'][3]) +
            torch.mm(self.I2, self.statespace['sem'][2])
        )
        
        # Multiplicative update (Samer style)
        self.statespace['sem'][0] = self._eps_mul(
            self.statespace['sem'][0],
            sem_update_term
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
        cpt_update_term = torch.mm(self.weights['S_to_C'], self.statespace['sem'][3])
        
        if cpt_clamp is not None:
            # Experimental: clamp concept to prime word identity
            # (used in some priming paradigms, not in reference model)
            self.statespace['cpt'][0] = cpt_clamp
        else:
            # Standard: concept evolves via semantic PE
            self.statespace['cpt'][0] = self._eps_mul(
                self.statespace['cpt'][0],
                cpt_update_term
            )
        
        # ======== TOP-DOWN RECONSTRUCTIONS ========
        # Higher levels generate predictions (reconstructions) for lower levels
        # These will be used to compute PE in the next iteration
        self.statespace['audio'][1] = torch.mm(self.weights['L_to_A'], self.statespace['lex'][0])
        self.statespace['lex'][1] = torch.mm(self.weights['S_to_L'], self.statespace['sem'][0])
        self.statespace['sem'][1] = torch.mm(self.weights['C_to_S'], self.statespace['cpt'][0])
        
        self.current_iteration += 1
    
    # ========================================================================
    # SUMMARY METRICS
    # ========================================================================
    
    def get_total_lexsem_error(self):
        """
        Calculate the N400 signal - sum of lexical + semantic prediction errors.

        The idea: N400 reflects how hard it is to integrate new semantic information.
        Here I measure it as total prediction error across lex and sem layers.

        Returns:
            N400 value for each trial in batch
        """
        # Sum all lexical PEs
        total_lex_err = torch.sum(self.statespace['lex'][3], dim=0)
        # Sum all semantic PEs
        total_sem_err = torch.sum(self.statespace['sem'][3], dim=0)

        # N400 CALCULATION OPTIONS:
        # Comment/uncomment the blocks below to switch between scaling approaches

        # OPTION 1: Raw sum (no scaling) - current approach
        n400_total = total_lex_err + total_sem_err

        # OPTION 2: Scaled semantic PE to match Samer's feature space (3700 vs ~19450)
        # samer_semantic_features = 3700  # From Samer's model
        # semantic_scale = samer_semantic_features / self.sem_dim  # ≈0.190
        # scaled_sem_err = total_sem_err * semantic_scale
        # n400_total = total_lex_err + scaled_sem_err

        # Use OPTION 1 by default (uncomment OPTION 2 and comment OPTION 1 to switch)
        return n400_total.cpu().numpy()
    
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
    Set everything up for batch processing on GPU.
    I used Claude to help me with this, it all looks good to me.
    """
    
    def run_batch_trials(self, prime_vecs, target_vecs, prime_indices,
                         prime_iters=20, blank_iters=5, target_iters=20,
                         post_target_iters=0, use_cpt_clamp=True):

        n_trials = prime_vecs.shape[1]
        
        # Convert inputs to GPU tensors
        prime_input = torch.tensor(prime_vecs, dtype=torch.float32, device=self.device)
        target_input = torch.tensor(target_vecs, dtype=torch.float32, device=self.device)
        blank_input = torch.zeros_like(prime_input)
        post_input = blank_input  # Same as blank (zero input)

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
            self.run_one_iteration(prime_input)
            all_lexsem_err.append(self.get_total_lexsem_error())
            all_lex_err.append(self.get_total_lex_error())
            all_sem_err.append(self.get_total_sem_error())
            all_max_activation.append(self.get_max_lex_state_activation())
            all_max_identity.append(self.get_max_lex_state_identity())
        
        # ---- PHASE 2: BLANK (ISI) ----
        for _ in range(blank_iters):
            self.run_one_iteration(blank_input)
            all_lexsem_err.append(self.get_total_lexsem_error())
            all_lex_err.append(self.get_total_lex_error())
            all_sem_err.append(self.get_total_sem_error())
            all_max_activation.append(self.get_max_lex_state_activation())
            all_max_identity.append(self.get_max_lex_state_identity())
        
        # Prepare concept clamp for target phase (if requested)
        # Clamps concept layer to prime word identity (simulates sustained prime activation)
        cpt_clamp = None
        if use_cpt_clamp:
            cpt_clamp = torch.zeros((self.lexicon_size, n_trials), dtype=torch.float32, device=self.device)
            for trial_idx, prime_idx in enumerate(prime_indices):
                cpt_clamp[prime_idx, trial_idx] = 1.0
        
        # ---- PHASE 3: TARGET ----
        for _ in range(target_iters):
            self.run_one_iteration(target_input, cpt_clamp=cpt_clamp)
            all_lexsem_err.append(self.get_total_lexsem_error())
            all_lex_err.append(self.get_total_lex_error())
            all_sem_err.append(self.get_total_sem_error())
            all_max_activation.append(self.get_max_lex_state_activation())
            all_max_identity.append(self.get_max_lex_state_identity())

        # ---- PHASE 4: POST-TARGET SETTLING ----
        for _ in range(post_target_iters):
            self.run_one_iteration(post_input)
            all_lexsem_err.append(self.get_total_lexsem_error())
            all_lex_err.append(self.get_total_lex_error())
            all_sem_err.append(self.get_total_sem_error())
            all_max_activation.append(self.get_max_lex_state_activation())
            all_max_identity.append(self.get_max_lex_state_identity())
        
        # Stack results: (n_iters, n_trials) → transpose to (n_trials, n_iters)
        return {
            'total_lexsem_err': np.array(all_lexsem_err).T,
            'total_lex_err': np.array(all_lex_err).T,
            'total_sem_err': np.array(all_sem_err).T,
            'max_lex_state_activation': np.array(all_max_activation).T,
            'max_lex_state_identity': np.array(all_max_identity).T,
        }
