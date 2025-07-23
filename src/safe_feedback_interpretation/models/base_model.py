"""Base model interface for conformal prediction."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseModel(ABC):
    """Abstract base class for prediction models.

    Models should return a probability distribution over outputs.
    """

    @abstractmethod
    def get_single_token_logits(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]],
    ) -> Dict[Any, Any]:
        """Get probability predictions for inputs.

        Args:
            inputs: Single input or list of inputs to predict on

        Returns:
            List with $n_samples$ elements, each containing a dictionary
            with token and logits for the most likely outputs for each input
        """
