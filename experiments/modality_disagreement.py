"""Check disagreement when only varying modality."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra

from safe_feedback_interpretation.models.openai_model import OpenAIModel


@hydra.main(version_base=None, config_name="modality_disagreement", config_path="conf/")
def _main(cfg):
    """Run the experiment."""
    temperature = cfg.temperature
    sys_prompt = cfg.sys_prompt
    model_name = cfg.model.model_name
    image_input = cfg.expression_input.image_input
    expression_text = cfg.expression_input.expression_text
    expression_text_placeholder = cfg.prompt.expression_text_placeholder
    text_input = (cfg.prompt.text_input).replace(
        expression_text_placeholder, expression_text
    )

    model = OpenAIModel(
        model=model_name,
        system_prompt=sys_prompt,
        temperature=temperature,
    )

    print(temperature, sys_prompt, model_name, text_input, image_input)

    response = model.get_single_token_logits(text_input, image_input)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    results = response

    # Save to file additively
    output_file = output_dir.parent / "results.json"
    
    # Load existing results if file exists
    existing_results = []
    if output_file.exists():
        with open(output_file, "r") as f:
            existing_results = json.load(f)
    
    # Append new result with metadata
    result_entry = {
        "timestamp": hydra.core.hydra_config.HydraConfig.get().runtime.output_dir.split('/')[-1],
        "config": {
            "model": model_name,
            "temperature": temperature,
            "text_input": text_input,
            "image_input": image_input
        },
        "results": results
    }
    existing_results.append(result_entry)
    
    # Save updated results
    with open(output_file, "w") as f:
        json.dump(existing_results, f, indent=2)

    print(f"Saved results to {output_file}")


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
