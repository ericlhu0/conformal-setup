"""Hydra-driven entrypoint to run experiments using the existing runner.

Currently implements Q1 (disagreement) via Hydra configs. No visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Ensure the 'experiments' folder is on sys.path so its local imports resolve
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "experiments"))
from experiments.experiment_runner import run_experiment
from safe_feedback_interpretation.models.openai_model import OpenAIModel


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the selected experiment with Hydra-provided configuration.

    Saves outputs under Hydra's run directory.
    """
    # Construct model from config
    model = OpenAIModel(
        model=cfg.model.name,
        system_prompt=cfg.model.sys_prompt,
        temperature=cfg.model.temperature,
    )

    # Hydra run directory for outputs
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Incremental results path (JSONL)
    incremental_file = str(run_dir / "incremental_results.jsonl")

    # Reuse the existing experiment runner for the selected experiment
    results = run_experiment(
        cfg.experiment.config_file,
        model,
        incremental_file,
        cfg.runtime.use_text_descriptions,
    )

    # Save final results (compatible structure with current analysis tools)
    output_data: Dict[str, Any] = {
        "meta": {
            "hydra_run_dir": str(run_dir),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        "raw_results": {
            # Q1 only for now per scope agreement
            "experiment_1": results,
        },
    }

    final_results_path = run_dir / "final_results.json"
    with open(final_results_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"Saved final results to {final_results_path}")
    print(f"Saved incremental results to {incremental_file}")


if __name__ == "__main__":  # pragma: no cover
    main()
