"""Base model interface for conformal prediction."""

from abc import ABC, abstractmethod
from typing import Any, List, Union


class BaseModel(ABC):
    """Abstract base class for models used in conformal prediction.

    Models should return probability distributions that can be used for
    calibration in conformal prediction methods.
    """

    @abstractmethod
    def __call__(self, inputs: Union[str, List[str]]) -> List[dict[Any, Any]]:
        """Get probability predictions for inputs.

        Args:
            inputs: Single input or list of inputs to predict on

        Returns:
            List with $n_samples$ elements, each containing a dictionary
            with token and logits for the most likely outputs for each input
        """
