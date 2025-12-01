"""
Predictive Coding Model for N400 Simulation

Based on Nour Eddine et al. (2024) - adapted for auditory priming.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from config import (
    DT,
    PRECISION_LEXICAL,
    PRECISION_SEMANTIC,
    LEARNING_RATE_LEXICAL,
    LEARNING_RATE_SEMANTIC,
    SEMANTIC_DECAY_RATE,
    DEVICE,
    SEMANTIC_PERSISTENCE,
    RESET_LEXICAL_ON_TARGET,
)


class PCModel:
    """
    Three-layer predictive coding model: Phoneme -> Lexical -> Semantic
    
    N400 is operationalized as semantic prediction error - the mismatch
    between expected semantics (from prime) and actual semantics (from target).
    """

    def __init__(self, V_LP: torch.Tensor, V_SL: torch.Tensor):
        self.V_LP = V_LP.float().to(DEVICE)
        self.V_SL = V_SL.float().to(DEVICE)

        self.INPUT_DIM = self.V_LP.shape[0]
        self.LEXICAL_DIM = self.V_LP.shape[1]
        self.SEMANTIC_DIM = self.V_SL.shape[0]

        self.state_lexical = None
        self.state_semantic = None
        self.error_phoneme = None
        self.error_lexical = None
        self.error_semantic = None
        self.semantic_expectation = None
        
        self.reset_state()

    def reset_state(self):
        """Reset all states for a new trial."""
        self.state_lexical = torch.zeros(self.LEXICAL_DIM, device=DEVICE)
        self.state_semantic = torch.zeros(self.SEMANTIC_DIM, device=DEVICE)
        self.error_phoneme = torch.zeros(self.INPUT_DIM, device=DEVICE)
        self.error_lexical = torch.zeros(self.LEXICAL_DIM, device=DEVICE)
        self.error_semantic = torch.zeros(self.SEMANTIC_DIM, device=DEVICE)
        self.semantic_expectation = None

    def step(self, input_u: torch.Tensor, pi_input: float) -> Tuple[float, float]:
        """Single timestep of PC dynamics."""
        # Top-down predictions
        pred_lexical = self.V_SL.T @ self.state_semantic
        pred_phoneme = self.V_LP @ self.state_lexical

        # Prediction errors
        self.error_phoneme = (input_u - pred_phoneme) * pi_input
        self.error_lexical = (self.state_lexical - pred_lexical) * PRECISION_LEXICAL
        self.error_semantic = (self.state_semantic - SEMANTIC_DECAY_RATE) * PRECISION_SEMANTIC

        # State updates
        drive_lex = self.V_LP.T @ self.error_phoneme
        delta_lex = drive_lex - self.error_lexical
        self.state_lexical += DT * LEARNING_RATE_LEXICAL * delta_lex
        
        drive_sem = self.V_SL @ self.error_lexical
        delta_sem = drive_sem - self.error_semantic
        self.state_semantic += DT * LEARNING_RATE_SEMANTIC * delta_sem

        # Non-negative activations
        self.state_lexical = F.relu(self.state_lexical)
        self.state_semantic = F.relu(self.state_semantic)

        return self.error_lexical.abs().sum().item(), self.error_semantic.abs().sum().item()

    def store_semantic_expectation(self):
        """Store current semantic state as expectation (call after prime settles)."""
        self.semantic_expectation = self.state_semantic.clone()
    
    def prepare_for_target(self):
        """Reset for target presentation while preserving some semantic activation."""
        if RESET_LEXICAL_ON_TARGET:
            self.state_lexical = torch.zeros(self.LEXICAL_DIM, device=DEVICE)
        
        self.state_semantic = self.state_semantic * SEMANTIC_PERSISTENCE
        self.error_phoneme = torch.zeros(self.INPUT_DIM, device=DEVICE)
        self.error_lexical = torch.zeros(self.LEXICAL_DIM, device=DEVICE)
        self.error_semantic = torch.zeros(self.SEMANTIC_DIM, device=DEVICE)
    
    def get_semantic_prediction_error(self) -> float:
        """N400 metric: mismatch between expected and actual semantics."""
        if self.semantic_expectation is None:
            return self.state_semantic.abs().sum().item()
        
        mismatch = torch.abs(self.semantic_expectation - self.state_semantic)
        return mismatch.sum().item()
    
    def get_semantic_similarity(self) -> float:
        """Cosine similarity between expected and actual semantics."""
        if self.semantic_expectation is None:
            return 0.0
        
        dot = torch.dot(self.semantic_expectation, self.state_semantic)
        norm_exp = torch.norm(self.semantic_expectation)
        norm_act = torch.norm(self.state_semantic)
        
        if norm_exp < 1e-10 or norm_act < 1e-10:
            return 0.0
        
        return (dot / (norm_exp * norm_act)).item()
    
    def get_lexical_entropy(self) -> float:
        """Entropy of lexical activation distribution."""
        probs = F.softmax(self.state_lexical, dim=0)
        log_probs = torch.log(probs + 1e-10)
        return -torch.sum(probs * log_probs).item()
    
    def get_winner_info(self) -> Tuple[int, float]:
        """Get index and activation of most active lexical unit."""
        idx = self.state_lexical.argmax().item()
        return idx, self.state_lexical[idx].item()
    
    def get_recognition_confidence(self, target_idx: int) -> float:
        """Probability assigned to the target word."""
        probs = F.softmax(self.state_lexical, dim=0)
        return probs[target_idx].item()
    
    def get_metrics(self, target_idx: Optional[int] = None) -> Dict[str, float]:
        """Get all relevant metrics for analysis."""
        winner_idx, winner_act = self.get_winner_info()
        
        metrics = {
            'n400': self.get_semantic_prediction_error(),
            'semantic_similarity': self.get_semantic_similarity(),
            'lexical_entropy': self.get_lexical_entropy(),
            'winner_idx': winner_idx,
            'winner_activation': winner_act,
        }
        
        if target_idx is not None:
            metrics['correct'] = (winner_idx == target_idx)
            metrics['target_prob'] = self.get_recognition_confidence(target_idx)
        
        return metrics
