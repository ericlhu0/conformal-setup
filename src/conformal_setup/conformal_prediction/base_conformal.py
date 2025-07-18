"""Base conformal prediction class."""

from abc import ABC, abstractmethod
from typing import Any, List, Union

import numpy as np
from torch.utils.data import Dataset

from ..models.base_model import BaseModel


class ConformalDataset(Dataset):
    """PyTorch Dataset for conformal prediction calibration."""

    def __init__(self, inputs: List[str], labels: List[str]):
        """Initialize dataset.

        Args:
            inputs: List of input texts
            labels: List of corresponding labels
        """
        if len(inputs) != len(labels):
            raise ValueError("Inputs and labels must have same length")

        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.inputs[idx], self.labels[idx]


class BaseConformalPredictor(ABC):
    """Base class for conformal prediction methods."""

    def __init__(self, model: BaseModel, confidence_level: float = 0.9):
        """Initialize conformal predictor.

        Args:
            model: Model that returns probability distributions
            confidence_level: Desired coverage probability (e.g., 0.9 for 90% coverage)
        """
        self.model = model
        self.confidence_level = confidence_level
        self.calibration_scores = None
        self.is_calibrated = False

    def calibrate(self, calibration_dataset: Dataset):
        """Calibrate predictor using labeled calibration data.

        Args:
            calibration_dataset: PyTorch Dataset with (input, label) pairs
        """
        # Extract inputs and labels from dataset
        inputs = []
        labels = []
        for input_text, label in calibration_dataset:
            inputs.append(input_text)
            labels.append(label)

        # Get model predictions for calibration set
        model_outputs = self.model(inputs)

        # Compute nonconformity scores
        scores = []
        for logprobs, true_label in zip(model_outputs, labels):
            score = self._compute_score(logprobs, true_label)
            scores.append(score)

        self.calibration_scores = np.array(scores)
        self.is_calibrated = True

    @abstractmethod
    def _compute_score(self, logprobs: dict[Any, Any], true_label: str) -> float:
        """Compute nonconformity score for a single example.

        Args:
            logprobs: Dictionary mapping tokens to log probabilities
            true_label: True label for this example

        Returns:
            Nonconformity score (higher = more nonconforming)
        """
        pass

    def predict(self, inputs: Union[str, List[str]]) -> List[List[str]]:
        """Generate prediction sets with coverage guarantees.

        Args:
            inputs: Single input or list of inputs to predict on

        Returns:
            List of prediction sets (one per input)
        """
        if not self.is_calibrated:
            raise ValueError("Must calibrate predictor before making predictions")

        if isinstance(inputs, str):
            inputs = [inputs]

        # Get model predictions
        model_outputs = self.model(inputs)

        # Compute threshold from calibration scores
        n = len(self.calibration_scores)
        alpha = 1 - self.confidence_level
        quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
        threshold = np.quantile(self.calibration_scores, quantile_level)

        # Generate prediction sets
        prediction_sets = []
        for logprobs in model_outputs:
            pred_set = self._get_prediction_set(logprobs, threshold)
            prediction_sets.append(pred_set)

        return prediction_sets

    def _get_prediction_set(
        self, logprobs: dict[Any, Any], threshold: float
    ) -> List[str]:
        """Generate prediction set for single input.

        Args:
            logprobs: Dictionary mapping tokens to log probabilities
            threshold: Threshold for inclusion in prediction set

        Returns:
            List of tokens in prediction set
        """
        prediction_set = []
        for token, logprob in logprobs.items():
            score = self._compute_score(logprobs, token)
            if score <= threshold:
                prediction_set.append(token)

        return prediction_set
