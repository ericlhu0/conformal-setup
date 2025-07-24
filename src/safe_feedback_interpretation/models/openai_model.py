"""OpenAI model wrapper."""

import base64
import os
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
from openai import OpenAI

from .base_model import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI API-based model for classification tasks using logits."""

    def __init__(
        self,
        model: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ):
        """Initialize OpenAI model. Assumes OPENAI_API_KEY and optionally
        OPENAI_ORG_ID environment variables are set.

        Args:
            model: OpenAI model name (e.g., "gpt-4.1-nano")
            temperature: Sampling temperature for responses
            max_tokens: Maximum tokens in response
            system_prompt: System prompt for the model
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORG_ID"),
        )

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _create_prompt(
        self, text_input: str, image_input: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Create simple prompt for LLM.

        Args:
            text_input: Text input to classify
            image_input: Image path(s)
        """

        if isinstance(image_input, str):
            image_input = [image_input]

        if image_input is None:
            image_input = []

        return [{"type": "text", "text": text_input}] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
                },
            }
            for image in image_input
        ]

    def get_single_token_logits(
        self,
        text_input: str,
        image_input: Optional[Union[str, List[str]]] = None,
    ) -> Dict[Any, Any]:
        """Get logits for single next predicted token.

        Args:
            text_input: input to classify
            image_input: image input(s) specified with OpenAI File API string id

        Returns:
            List with $n_samples$ elements, each containing a dictionary
            with token and logits for the 20 most likely outputs for each input
        """
        # Process inputs
        # Create classification prompt
        user_prompt = self._create_prompt(text_input, image_input)

        # Call OpenAI API with logprobs
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": cast(Any, user_prompt)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=True,
            top_logprobs=10,
        )

        # https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs
        print(response)

        top_20_token_probs = {}

        # Appease mypy (output should be the format where you request logprobs)
        # choices[0] because choices is 1 long except requesting multiple answers
        # content[0] because we only check probability of first token
        assert response.choices[0].logprobs is not None
        assert response.choices[0].logprobs.content is not None
        assert response.choices[0].logprobs.content[0].top_logprobs is not None

        possible_outputs = response.choices[0].logprobs.content[0].top_logprobs
        for possible_output in possible_outputs:
            top_20_token_probs[possible_output.token] = np.exp(possible_output.logprob)

        return top_20_token_probs
