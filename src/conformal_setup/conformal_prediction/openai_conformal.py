"""OpenAI-specific conformal prediction implementation."""

import math
from typing import Any

from ..models.openai_model import OpenAIModel
from .base_conformal import BaseConformalPredictor


class OpenAIConformalPredictor(BaseConformalPredictor):
    """Conformal predictor specifically for OpenAI models using adaptive
    prediction sets."""

    def __init__(self, model: OpenAIModel, confidence_level: float = 0.9):
        """Initialize OpenAI conformal predictor.

        Args:
            model: OpenAI model instance
            confidence_level: Desired coverage probability
        """
        super().__init__(model, confidence_level)

    def _compute_score(self, logprobs: dict[Any, Any], true_label: str) -> float:
        """Compute adaptive prediction set score.

        Uses 1 - P(true_label) as the nonconformity score, which corresponds
        to the adaptive prediction set method for classification.

        Args:
            logprobs: Dictionary mapping tokens to log probabilities
            true_label: True label for this example

        Returns:
            Nonconformity score (1 - probability of true label)
        """
        if true_label not in logprobs:
            # If true label not in top predictions, assign maximum score
            return 1.0

        # Convert log probability to probability
        prob_true_label = math.exp(logprobs[true_label])

        # Score is 1 - probability (higher score = less conforming)
        return 1.0 - prob_true_label
