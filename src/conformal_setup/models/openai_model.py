"""OpenAI-based model implementation for conformal prediction."""

import os
from typing import Any, List, Optional, Union

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

    def _create_prompt(self, input_text: str) -> str:
        """Create simple prompt for LLM."""
        return input_text

    def __call__(
        self,
        inputs: Union[str, List[str]],
    ) -> List[dict[Any, Any]]:
        """Get probability predictions for inputs.

        Args:
            inputs: Single input or list of inputs to classify

        Returns:
            List with $n_samples$ elements, each containing a dictionary
            with token and logits for the 20 most likely outputs for each input
        """

        if isinstance(inputs, str):
            inputs = [inputs]

        # Process inputs
        output = []

        for input_text in inputs:
            try:
                # Create classification prompt
                user_prompt = self._create_prompt(input_text)

                # Call OpenAI API with logprobs
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    logprobs=True,
                    top_logprobs=20,
                )

                # https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs
                print(response)

                top_20_token_logprobs = {}

                # Appease mypy (output should be the format where you request logprobs)
                # choices[0] because choices is 1 long except requesting multiple answers
                # content[0] because we only check probability of first token
                assert response.choices[0].logprobs is not None
                assert response.choices[0].logprobs.content is not None
                assert response.choices[0].logprobs.content[0].top_logprobs is not None

                possible_outputs = response.choices[0].logprobs.content[0].top_logprobs
                for possible_output in possible_outputs:
                    top_20_token_logprobs[possible_output.token] = (
                        possible_output.logprob
                    )

                output.append(top_20_token_logprobs)

            except Exception as e:
                print(f"Error calling OpenAI API: {e}")

        return output
