#!/usr/bin/env python3
"""Create final_results.json from incremental_results.jsonl and config
files."""

import json
import os
from typing import Any, Dict, List


def load_incremental_results(incremental_file: str) -> List[Dict[str, Any]]:
    """Load all results from incremental JSONL file."""
    results = []
    if os.path.exists(incremental_file):
        with open(incremental_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results


def load_experiment_config(config_file: str) -> Dict[str, Any]:
    """Load experiment configuration."""
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def organize_results_by_experiment(
    incremental_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Organize incremental results by experiment and scenario."""
    organized: Dict[str, Dict[str, Any]] = {}

    for result in incremental_results:
        # Extract experiment info from query_id or metadata
        metadata = result.get("input_metadata", {})
        scenario_name = metadata.get("scenario_name", "unknown")
        query_type = result.get("query_type", metadata.get("query_type", "unknown"))
        body_part = result.get("body_part", "unknown")

        # Determine experiment from scenario name pattern
        if (
            scenario_name.startswith("v")
            and "f" in scenario_name
            and len(scenario_name.split("_")[0]) == 4
        ):
            experiment_key = "experiment_1"  # disagreement
        elif "ambig" in scenario_name.lower() or "specific" in scenario_name.lower():
            experiment_key = "experiment_2"  # ambiguity
        elif "mild" in scenario_name.lower() or "severe" in scenario_name.lower():
            experiment_key = "experiment_3"  # intensity
        elif "modal" in scenario_name.lower():
            experiment_key = "experiment_4"  # modality
        else:
            experiment_key = "experiment_5"  # uncertainty

        # Initialize experiment structure
        if experiment_key not in organized:
            organized[experiment_key] = {"results": {}}

        if scenario_name not in organized[experiment_key]["results"]:
            organized[experiment_key]["results"][scenario_name] = {
                "predictions_full": {},
                "predictions_single": {},
            }

        # Store result based on query type
        cleaned_probs = result.get("cleaned_probabilities", {})

        if query_type == "body_part_full":
            organized[experiment_key]["results"][scenario_name]["predictions_full"][
                body_part
            ] = cleaned_probs
        elif query_type == "body_part_single":
            organized[experiment_key]["results"][scenario_name]["predictions_single"][
                body_part
            ] = cleaned_probs

    return organized


def create_final_results(
    incremental_file: str,
    config_files: List[str],
    output_file: str = "final_results.json",
):
    """Create final_results.json from incremental results and configs."""

    print(f"📊 Loading incremental results from {incremental_file}...")
    incremental_results = load_incremental_results(incremental_file)
    print(f"✅ Loaded {len(incremental_results)} incremental results")

    print("📋 Organizing results by experiment...")
    organized_results = organize_results_by_experiment(incremental_results)

    # Load experiment configs for metadata
    experiments_meta = {}
    for i, config_file in enumerate(config_files, 1):
        if os.path.exists(config_file):
            config = load_experiment_config(config_file)
            exp_key = f"experiment_{i}"
            experiments_meta[exp_key] = {
                "experiment_name": config.get("experiment_name", f"Experiment {i}"),
                "description": config.get("description", ""),
                "num_scenarios": len(config.get("scenarios", [])),
            }
            print(
                f"✅ Loaded config for {exp_key}: {config.get('experiment_name', '')}"
            )

    # Combine organized results with metadata
    final_data = {
        "raw_results": organized_results,
        "experiment_metadata": experiments_meta,
        "analyses": {},  # Placeholder for analyses
    }

    # Save final results
    print(f"💾 Saving final results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, default=str)

    print("✅ Final results saved successfully!")

    # Print summary
    print("\n📊 SUMMARY:")
    print(f"   Input file: {incremental_file}")
    print(f"   Output file: {output_file}")
    print(f"   Total experiments: {len(organized_results)}")
    for exp_key, exp_data in organized_results.items():
        num_scenarios = len(exp_data.get("results", {}))
        print(f"   {exp_key}: {num_scenarios} scenarios")


def main() -> None:
    """Main function to create final results from incremental data."""

    # Default paths - update these as needed
    incremental_file = "08-16-11:34/incremental_results.jsonl"
    config_files = [
        "experiments/configs/experiment_1_disagreement.json",
        "experiments/configs/experiment_2_ambiguity.json",
        "experiments/configs/experiment_3_intensity.json",
        "experiments/configs/experiment_4_modality.json",
        "experiments/configs/experiment_5_uncertainty.json",
    ]
    output_file = "08-16-11:34/final_results.json"

    # Check if incremental file exists
    if not os.path.exists(incremental_file):
        print(f"❌ Error: Incremental results file not found: {incremental_file}")
        print("Available files:")
        for file in os.listdir("."):
            if file.endswith(".jsonl"):
                print(f"   {file}")
        return

    # Filter existing config files
    existing_configs = [f for f in config_files if os.path.exists(f)]
    if not existing_configs:
        print("❌ Error: No experiment config files found!")
        return

    print("🚀 Creating final results from incremental data...")
    create_final_results(incremental_file, existing_configs, output_file)


if __name__ == "__main__":
    main()
