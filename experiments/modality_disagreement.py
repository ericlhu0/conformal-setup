"""Check disagreement when only varying modality."""

import json
from itertools import product
from pathlib import Path
from typing import List

import hydra

from safe_feedback_interpretation.models.openai_model import OpenAIModel
from safe_feedback_interpretation.viz_utils import visualize_results


@hydra.main(version_base=None, config_name="modality_disagreement", config_path="conf/")
def _main(cfg):
    """Run the experiment."""
    temperature: float = cfg.temperature
    base_sys_prompt: str = cfg.sys_prompt
    model_names: List[str] = cfg.models
    image_inputs: List[str] = cfg.expression_input.image_input
    expression_texts: List[str] = cfg.expression_input.expression_text
    expression_text_placeholder: str = cfg.prompt.expression_text_placeholder
    sensitivity_specs: List[str] = cfg.sensitivity_specs
    sensitivity_spec_placeholder: str = cfg.sensitivity_spec_placeholder
    base_text_input: str = cfg.prompt.text_input

    for model_name, sensitivity_spec in product(model_names, sensitivity_specs):
        sys_prompt = base_sys_prompt.replace(
            sensitivity_spec_placeholder, sensitivity_spec
        )

        model = OpenAIModel(
            model=model_name,
            system_prompt=sys_prompt,
            temperature=temperature,
        )

        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

        # Save to file additively
        output_file = output_dir.parent / "results.json"

        for image_input, expression_text in product(image_inputs, expression_texts):
            text_input = base_text_input.replace(
                expression_text_placeholder, expression_text
            )
            results = model.get_single_token_logits(text_input, image_input)

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
                    "sensitivity_spec": sensitivity_spec,
                    "text_input": base_text_input,
                    "expression_text": expression_text,
                    "expression_image": image_input,
                },
                "results": results,
            }
            existing_results.append(result_entry)

            # Save updated results
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, indent=2)

            print(f"Saved results to {output_file}")

    figures = visualize_results(output_file)

    # Save and show figures
    for i, (_, fig) in enumerate(figures):
        fig.savefig(output_dir.parent / f"results_fig_{i+1}.png")


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
