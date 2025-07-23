"""Check disagreement when only varying modality."""

import json
from pathlib import Path
from typing import List

import hydra

from safe_feedback_interpretation.models.openai_model import OpenAIModel


@hydra.main(version_base=None, config_name="modality_disagreement", config_path="conf/")
def _main(cfg):
    """Run the experiment."""
    temperature: float = cfg.temperature
    sys_prompt: str = cfg.sys_prompt
    model_names: List[str] = cfg.models
    image_inputs: List[str] = cfg.expression_input.image_input
    expression_texts: List[str] = cfg.expression_input.expression_text
    expression_text_placeholder: str = cfg.prompt.expression_text_placeholder
    text_inputs: List[str] = [
        (cfg.prompt.text_input).replace(expression_text_placeholder, expression_text)
        for expression_text in expression_texts
    ]

    for model_name in model_names:
        model = OpenAIModel(
            model=model_name,
            system_prompt=sys_prompt,
            temperature=temperature,
        )

        for image_input, text_input in zip(image_inputs, text_inputs):
            response = model.get_single_token_logits(text_input, image_input)

            output_dir = Path(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            )

            results = response

            # Save to file additively
            output_file = output_dir.parent / "results.json"

            # Load existing results if file exists
            existing_results = []
            if output_file.exists():
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)

            # Append new result with metadata
            result_entry = {
                "config": {
                    "model": model_name,
                    "temperature": temperature,
                    "text_input": text_input,
                    "image_input": image_input,
                },
                "results": results,
            }
            existing_results.append(result_entry)

            # Save updated results
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, indent=2)

            print(f"Saved results to {output_file}")


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
