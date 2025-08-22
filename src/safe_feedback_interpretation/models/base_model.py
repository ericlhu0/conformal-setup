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
        """Get logits for single next predicted token.

        Args:
            text_input: input to classify
            image_input: image input(s) specified with file path string

        Returns:
            dict with token and logits for the 10 most likely outputs for next token
        """

    @abstractmethod
    def get_last_single_token_logits(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> Dict[Any, Any]:
        """Get logits for the last predicted token.

        Args:
            text_input: input to classify
            image_input: image input(s) specified with file path string

        Returns:
            dict with token and logits for the 10 most likely outputs for the last token
            of the output sequence.
        """

    @abstractmethod
    def get_full_output(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """Get sentence output from LLM."""
