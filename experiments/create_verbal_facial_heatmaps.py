#!/usr/bin/env python3
"""Create heatmaps with verbal expression vs facial expression axes for Q1
analysis."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_q1_data_from_results_dir(results_dir: str) -> Dict[str, Any]:
    """Load Q1 data from a specific results directory."""
    # Keep imports local to avoid circular deps when this script runs standalone.
    # Use top-level 'sys' to modify path without reimporting.
    sys.path.append(".")

    from analyze_q1_disagreement import (  # pylint: disable=import-outside-toplevel
        analyze_brier_scores_by_experiment_type,
        is_q1_scenario,
        organize_predictions_by_scenario,
        parse_scenario_classification,
    )

    curr_dir = Path(__file__).resolve().parent

    # Load expert labels from experiment config
    config_path = curr_dir / "configs/experiment_1_disagreement.json"
    if not config_path.exists():
        raise FileNotFoundError(f"ERROR: Q1 config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        expert_data = json.load(f)

    # Load model predictions from specified results directory
    final_results_path = curr_dir / "results" / results_dir / "final_results.json"
    if not final_results_path.exists():
        raise FileNotFoundError(
            f"ERROR: Final results file not found at {final_results_path}"
        )

    with open(final_results_path, "r", encoding="utf-8") as f:
        final_results = json.load(f)

    # Extract Q1 experiment results
    q1_results = (
        final_results.get("raw_results", {}).get("experiment_1", {}).get("results", {})
    )
    if not q1_results:
        raise ValueError("ERROR: No experiment_1 results found in final_results.json")

    # Convert to format expected by rest of the code
    final_results_list = []
    for scenario_name, scenario_data in q1_results.items():
        if is_q1_scenario(scenario_name):
            # Extract predictions for each body part and query type
            for body_part in ["entire_arm", "upper_arm", "forearm", "wrist"]:
                if body_part in scenario_data.get("predictions_full", {}):
                    final_results_list.append(
                        {
                            "input_metadata": {
                                "scenario_name": scenario_name,
                                "query_type": "body_part_full",
                            },
                            "query_type": "body_part_full",
                            "body_part": body_part,
                            "cleaned_probabilities": scenario_data["predictions_full"][
                                body_part
                            ],
                        }
                    )

                if body_part in scenario_data.get("predictions_single", {}):
                    final_results_list.append(
                        {
                            "input_metadata": {
                                "scenario_name": scenario_name,
                                "query_type": "body_part_single",
                            },
                            "query_type": "body_part_single",
                            "body_part": body_part,
                            "cleaned_probabilities": scenario_data[
                                "predictions_single"
                            ][body_part],
                        }
                    )

    # Parse scenario classifications
    scenarios_by_type = {}
    for scenario in expert_data["scenarios"]:
        scenario_name = scenario["name"]
        scenario_type = parse_scenario_classification(scenario_name)
        scenarios_by_type[scenario_name] = {
            "expert_labels": scenario["labels"],
            "classification": scenario_type,
            "received_feedback": scenario["received_feedback"],
        }

    organized_predictions = organize_predictions_by_scenario(final_results_list)
    summary_stats = analyze_brier_scores_by_experiment_type(
        organized_predictions, scenarios_by_type
    )

    return summary_stats


def load_q1_analysis_results(
    results_dir: str = "image_facial_expression_results",
) -> Dict[str, Any]:
    """Load and run Q1 analysis to get results from specified directory."""
    return load_q1_data_from_results_dir(results_dir)


def load_both_modalities() -> tuple:
    """Load Q1 analysis results for both image and text modalities."""
    print("  Loading image facial expression results...")
    image_stats = load_q1_analysis_results("image_facial_expression_results")

    print("  Loading text facial expression results...")
    text_stats = load_q1_analysis_results("text_facial_expression_results")

    return image_stats, text_stats


def create_verbal_facial_matrix(
    summary_stats: Dict[str, Any],
) -> tuple[Dict[tuple[str, str], Dict[str, Any]], List[str]]:
    """Create a matrix with verbal (rows) vs facial (columns) expressions."""

    # Define the order for consistent presentation
    levels = ["none", "mid", "high"]

    # Initialize matrix data
    matrix_data = {}

    for stats in summary_stats.values():
        verbal_level = stats.get("verbal_level", "unknown")
        facial_level = stats.get("facial_level", "unknown")

        if verbal_level in levels and facial_level in levels:
            # Calculate entropy differences
            ef = stats.get("entropy_full_mean", np.nan)
            es = stats.get("entropy_single_mean", np.nan)
            el = stats.get("entropy_labels_mean", np.nan)
            entropy_diff_full = (
                (ef - el) if (not np.isnan(ef) and not np.isnan(el)) else np.nan
            )
            entropy_diff_single = (
                (es - el) if (not np.isnan(es) and not np.isnan(el)) else np.nan
            )

            # Store all metrics for this combination
            matrix_data[(verbal_level, facial_level)] = {
                "brier_full_mean": stats.get("brier_scores_full_mean", np.nan),
                "brier_single_mean": stats.get("brier_scores_single_mean", np.nan),
                "brier_full_std": stats.get("brier_scores_full_std", np.nan),
                "brier_single_std": stats.get("brier_scores_single_std", np.nan),
                "entropy_full_mean": stats.get("entropy_full_mean", np.nan),
                "entropy_single_mean": stats.get("entropy_single_mean", np.nan),
                "entropy_labels_mean": stats.get("entropy_labels_mean", np.nan),
                "entropy_full_std": stats.get("entropy_full_std", np.nan),
                "entropy_single_std": stats.get("entropy_single_std", np.nan),
                "entropy_diff_full": entropy_diff_full,
                "entropy_diff_single": entropy_diff_single,
                "label_prob_mass_full_mean": stats.get(
                    "label_prob_mass_full_mean", np.nan
                ),
                "label_prob_mass_single_mean": stats.get(
                    "label_prob_mass_single_mean", np.nan
                ),
                "label_prob_mass_full_std": stats.get(
                    "label_prob_mass_full_std", np.nan
                ),
                "label_prob_mass_single_std": stats.get(
                    "label_prob_mass_single_std", np.nan
                ),
                "f1_full": stats.get("f1_full", np.nan),
                "f1_single": stats.get("f1_single", np.nan),
                "precision_full": stats.get("precision_full", np.nan),
                "precision_single": stats.get("precision_single", np.nan),
                "recall_full": stats.get("recall_full", np.nan),
                "recall_single": stats.get("recall_single", np.nan),
                "accuracy_full": stats.get("accuracy_full", np.nan),
                "accuracy_single": stats.get("accuracy_single", np.nan),
                "mae_full": stats.get("mae_full_mean", np.nan),
                "mae_single": stats.get("mae_single_mean", np.nan),
                "mae_full_std": stats.get("mae_full_std", np.nan),
                "mae_single_std": stats.get("mae_single_std", np.nan),
                "cosine_similarity_full": stats.get(
                    "cosine_similarity_full_mean", np.nan
                ),
                "cosine_similarity_single": stats.get(
                    "cosine_similarity_single_mean", np.nan
                ),
                "cosine_similarity_full_std": stats.get(
                    "cosine_similarity_full_std", np.nan
                ),
                "cosine_similarity_single_std": stats.get(
                    "cosine_similarity_single_std", np.nan
                ),
                "n_scenarios": stats.get("n_scenarios", 0),
            }

    return matrix_data, levels


def create_heatmap(
    matrix_data: Dict,
    levels: list,
    metric: str,
    title: str,
    filename: str,
    reverse_scale: bool = False,
    vmin=None,
    vmax=None,
    grayscale: bool = False,
    cmap_name: str | None = None,
    modality: str = "image",
    subfolder: str | None = None,
):
    """Create a heatmap with verbal (rows) vs facial (columns) expressions."""  # pylint: disable=too-many-positional-arguments

    # Create output directory relative to script location
    script_dir = Path(__file__).resolve().parent
    if subfolder:
        output_dir = script_dir / "results" / "heatmaps" / subfolder
    else:
        output_dir = script_dir / "results" / "heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create matrix and annotation array
    matrix = np.full((len(levels), len(levels)), np.nan)
    annotations = np.full((len(levels), len(levels)), "", dtype=object)

    # Determine std metric name if available
    std_metric = None
    if metric.endswith("_mean"):
        base_metric = metric.replace("_mean", "")
        std_metric = f"{base_metric}_std"
    elif metric in [
        "brier_full",
        "brier_single",
        "entropy_full",
        "entropy_single",
        "mae_full",
        "mae_single",
        "label_prob_mass_full_mean",
        "label_prob_mass_single_mean",
        "cosine_similarity_full",
        "cosine_similarity_single",
    ]:
        if metric.endswith("_mean"):
            std_metric = metric.replace("_mean", "_std")
        else:
            std_metric = f"{metric}_std"

    for i, verbal in enumerate(levels):
        for j, facial in enumerate(levels):
            if (verbal, facial) in matrix_data:
                value = matrix_data[(verbal, facial)].get(metric, np.nan)
                matrix[i, j] = value

                # Create annotation with std if available
                if not np.isnan(value):
                    if std_metric and std_metric in matrix_data[(verbal, facial)]:
                        std_value = matrix_data[(verbal, facial)].get(
                            std_metric, np.nan
                        )
                        if not np.isnan(std_value):
                            annotations[i, j] = f"{value:.3f}\n±{std_value:.3f}"
                        else:
                            annotations[i, j] = f"{value:.3f}"
                    else:
                        annotations[i, j] = f"{value:.3f}"
                else:
                    annotations[i, j] = "N/A"

    # Create DataFrame for better labeling
    df_matrix = pd.DataFrame(
        matrix,
        index=[f"Verbal: {level.title()}" for level in levels],
        columns=[f"Facial: {level.title()}" for level in levels],
    )

    # Create heatmap
    plt.figure(figsize=(8, 6))

    # Choose colormap
    if cmap_name is not None:
        cmap = cmap_name
    else:
        if grayscale:
            # Use grayscale colormap
            cmap = "gray_r" if reverse_scale else "gray"
        else:
            # For metrics where lower is better (Brier, MAE), reverse the colormap
            # so that dark blue = better (lower values)
            cmap = "Blues_r" if reverse_scale else "Blues"

    sns.heatmap(
        df_matrix,
        annot=annotations,
        fmt="",  # Use empty format since we're providing custom annotations
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )

    modality_title = "Images" if modality == "image" else "Text Descriptions"
    plt.title(
        (
            f"Facial Expressions as {modality_title}\n"
            f"{title}\n"
            f"Verbal Expression (rows) vs Facial Expression (columns)"
        ),
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save plot to organized directory
    full_path = output_dir / filename
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved heatmap: {full_path}")

    # Print diagonal (agreement) vs off-diagonal (disagreement) analysis
    diagonal_values = [
        matrix[i, i] for i in range(len(levels)) if not np.isnan(matrix[i, i])
    ]
    off_diagonal_values = [
        matrix[i, j]
        for i in range(len(levels))
        for j in range(len(levels))
        if i != j and not np.isnan(matrix[i, j])
    ]

    if diagonal_values and off_diagonal_values:
        print(f"   {metric}:")
        print(f"     Agreement (diagonal) mean: {np.mean(diagonal_values):.3f}")
        print(
            (
                f"     Disagreement (off-diagonal) mean: "
                f"{np.mean(off_diagonal_values):.3f}"
            )
        )
        print(
            (
                f"     Agreement vs Disagreement difference: "
                f"{np.mean(diagonal_values) - np.mean(off_diagonal_values):.3f}"
            )
        )


def get_combined_value_range(
    image_matrix_data: Dict, text_matrix_data: Dict, levels: list, metric: str
) -> tuple:
    """Calculate combined min/max values across both image and text data for
    unified scaling."""
    all_values = []

    # Collect values from both datasets
    for matrix_data in [image_matrix_data, text_matrix_data]:
        for verbal in levels:
            for facial in levels:
                if (verbal, facial) in matrix_data:
                    value = matrix_data[(verbal, facial)].get(metric, np.nan)
                    if not np.isnan(value):
                        all_values.append(value)

    if not all_values:
        return None, None

    return np.min(all_values), np.max(all_values)


def create_comparison_heatmap(
    image_matrix_data: Dict,
    text_matrix_data: Dict,
    levels: list,
    metric: str,
    title: str,
    filename: str,
    reverse_scale: bool = False,
    vmin=None,
    vmax=None,
    grayscale: bool = False,
    cmap_name: str | None = None,
):
    """Create side-by-side comparison heatmaps for image vs text modalities."""  # pylint: disable=too-many-positional-arguments

    # Create output directory relative to script location
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "results" / "heatmaps" / "modality_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate unified color scale
    if vmin is None or vmax is None:
        combined_min, combined_max = get_combined_value_range(
            image_matrix_data, text_matrix_data, levels, metric
        )
        if combined_min is not None and combined_max is not None:
            if vmin is None:
                vmin = combined_min
            if vmax is None:
                vmax = combined_max

    # Create matrices and annotation arrays for both modalities
    image_matrix = np.full((len(levels), len(levels)), np.nan)
    text_matrix = np.full((len(levels), len(levels)), np.nan)
    image_annotations = np.full((len(levels), len(levels)), "", dtype=object)
    text_annotations = np.full((len(levels), len(levels)), "", dtype=object)

    # Determine std metric name if available
    std_metric = None
    if metric.endswith("_mean"):
        base_metric = metric.replace("_mean", "")
        std_metric = f"{base_metric}_std"
    elif metric in [
        "brier_full",
        "brier_single",
        "entropy_full",
        "entropy_single",
        "mae_full",
        "mae_single",
        "label_prob_mass_full_mean",
        "label_prob_mass_single_mean",
        "cosine_similarity_full",
        "cosine_similarity_single",
    ]:
        if metric.endswith("_mean"):
            std_metric = metric.replace("_mean", "_std")
        else:
            std_metric = f"{metric}_std"

    for i, verbal in enumerate(levels):
        for j, facial in enumerate(levels):
            # Image modality
            if (verbal, facial) in image_matrix_data:
                value = image_matrix_data[(verbal, facial)].get(metric, np.nan)
                image_matrix[i, j] = value

                if not np.isnan(value):
                    if std_metric and std_metric in image_matrix_data[(verbal, facial)]:
                        std_value = image_matrix_data[(verbal, facial)].get(
                            std_metric, np.nan
                        )
                        if not np.isnan(std_value):
                            image_annotations[i, j] = f"{value:.3f}\n±{std_value:.3f}"
                        else:
                            image_annotations[i, j] = f"{value:.3f}"
                    else:
                        image_annotations[i, j] = f"{value:.3f}"
                else:
                    image_annotations[i, j] = "N/A"

            # Text modality
            if (verbal, facial) in text_matrix_data:
                value = text_matrix_data[(verbal, facial)].get(metric, np.nan)
                text_matrix[i, j] = value

                if not np.isnan(value):
                    if std_metric and std_metric in text_matrix_data[(verbal, facial)]:
                        std_value = text_matrix_data[(verbal, facial)].get(
                            std_metric, np.nan
                        )
                        if not np.isnan(std_value):
                            text_annotations[i, j] = f"{value:.3f}\n±{std_value:.3f}"
                        else:
                            text_annotations[i, j] = f"{value:.3f}"
                    else:
                        text_annotations[i, j] = f"{value:.3f}"
                else:
                    text_annotations[i, j] = "N/A"

    # Create DataFrames for better labeling
    df_image = pd.DataFrame(
        image_matrix,
        index=[f"Verbal: {level.title()}" for level in levels],
        columns=[f"Facial: {level.title()}" for level in levels],
    )

    df_text = pd.DataFrame(
        text_matrix,
        index=[f"Verbal: {level.title()}" for level in levels],
        columns=[f"Facial: {level.title()}" for level in levels],
    )

    # Create side-by-side subplots with better spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Choose colormap
    if cmap_name is not None:
        cmap = cmap_name
    else:
        if grayscale:
            cmap = "gray_r" if reverse_scale else "gray"
        else:
            cmap = "Blues_r" if reverse_scale else "Blues"

    # Create heatmaps with unified color scale and custom annotations
    sns.heatmap(
        df_image,
        annot=image_annotations,
        fmt="",  # Use empty format since we're providing custom annotations
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar=False,  # We'll add a single colorbar later
        linewidths=0.5,
        ax=ax1,
    )

    sns.heatmap(
        df_text,
        annot=text_annotations,
        fmt="",  # Use empty format since we're providing custom annotations
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        ax=ax2,
    )

    # Set subplot titles
    ax1.set_title(
        "Facial Expressions as Images", fontsize=12, fontweight="bold", pad=10
    )
    ax2.set_title(
        "Facial Expressions as Text Descriptions",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    # Set main title with better positioning
    fig.suptitle(
        f"Image vs Text Comparison: {title}", fontsize=14, fontweight="bold", y=0.95
    )

    # Adjust spacing - minimal space between plots and make room for titles
    plt.subplots_adjust(left=0.08, right=0.92, top=0.82, bottom=0.15, wspace=0)

    # Save plot to organized directory
    full_path = output_dir / filename
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved comparison heatmap: {full_path}")

    # Print comparison analysis
    print(f"   {metric} Comparison Analysis:")

    # Analyze image data
    image_diagonal = [
        image_matrix[i, i]
        for i in range(len(levels))
        if not np.isnan(image_matrix[i, i])
    ]
    image_off_diagonal = [
        image_matrix[i, j]
        for i in range(len(levels))
        for j in range(len(levels))
        if i != j and not np.isnan(image_matrix[i, j])
    ]

    # Analyze text data
    text_diagonal = [
        text_matrix[i, i] for i in range(len(levels)) if not np.isnan(text_matrix[i, i])
    ]
    text_off_diagonal = [
        text_matrix[i, j]
        for i in range(len(levels))
        for j in range(len(levels))
        if i != j and not np.isnan(text_matrix[i, j])
    ]

    if image_diagonal and image_off_diagonal:
        print(
            (
                f"     Image - Agreement mean: {np.mean(image_diagonal):.3f}, "
                f"Disagreement mean: {np.mean(image_off_diagonal):.3f}"
            )
        )

    if text_diagonal and text_off_diagonal:
        print(
            (
                f"     Text - Agreement mean: {np.mean(text_diagonal):.3f}, "
                f"Disagreement mean: {np.mean(text_off_diagonal):.3f}"
            )
        )

    # Overall comparison
    if image_diagonal and text_diagonal:
        image_avg = np.mean(image_diagonal + image_off_diagonal)
        text_avg = np.mean(text_diagonal + text_off_diagonal)
        better_modality = "Image" if image_avg > text_avg else "Text"
        if (
            reverse_scale
            or metric.startswith("brier")
            or metric.startswith("mae")
            or metric.startswith("entropy")
        ):
            better_modality = "Text" if image_avg > text_avg else "Image"
        print(
            (
                f"     Better overall performance: {better_modality} "
                f"(Image: {image_avg:.3f}, Text: {text_avg:.3f})"
            )
        )


def create_all_heatmaps(matrix_data: Dict, levels: list, modality: str = "image"):
    """Create all heatmaps for different metrics."""

    modality_suffix = f"_{modality}" if modality != "image" else ""
    print(
        f"🔍 Creating verbal vs facial expression heatmaps for {modality} modality..."
    )

    # Key metrics for regular heatmaps - comprehensive set
    metrics_to_create: List[tuple[str, str, bool, float, float, str | None]] = [
        (
            "brier_full_mean",
            "Brier Score - Full Output (Dark = Better)",
            True,
            0.0,
            0.2,
            None,
        ),
        (
            "brier_single_mean",
            "Brier Score - Single Token (Dark = Better)",
            True,
            0.0,
            0.2,
            None,
        ),
        (
            "entropy_full_mean",
            "Entropy - Full Output (Dark = More Certain)",
            True,
            1.176,
            1.584,
            None,
        ),
        (
            "entropy_single_mean",
            "Entropy - Single Token (Dark = More Certain)",
            True,
            0.0,
            0.15,
            None,
        ),
        ("f1_full", "F1 Score - Full Output (Dark = Better)", False, 0.0, 1.0, None),
        ("f1_single", "F1 Score - Single Token (Dark = Better)", False, 0.0, 1.0, None),
        (
            "accuracy_full",
            "Accuracy - Full Output (Dark = Better)",
            False,
            0.0,
            1.0,
            None,
        ),
        (
            "accuracy_single",
            "Accuracy - Single Token (Dark = Better)",
            False,
            0.0,
            1.0,
            None,
        ),
        (
            "mae_full",
            "Mean Absolute Error - Full Output (Dark = Better)",
            True,
            0.0,
            1.0,
            None,
        ),
        (
            "mae_single",
            "Mean Absolute Error - Single Token (Dark = Better)",
            True,
            0.0,
            1.0,
            None,
        ),
        (
            "label_prob_mass_full_mean",
            "Label Probability Mass - Full Output (Dark = Better)",
            False,
            0.5,
            1.0,
            None,
        ),
        (
            "label_prob_mass_single_mean",
            "Label Probability Mass - Single Token (Dark = Better)",
            False,
            0.0,
            1.0,
            None,
        ),
        (
            "cosine_similarity_full",
            "Cosine Similarity - Full Output (Dark = Better)",
            False,
            0.0,
            1.0,
            None,
        ),
        (
            "cosine_similarity_single",
            "Cosine Similarity - Single Token (Dark = Better)",
            False,
            0.0,
            1.0,
            None,
        ),
        (
            "entropy_labels_mean",
            "Entropy - True Labels (Dark = More Certain)",
            True,
            0.469,
            1.406,
            None,
        ),
    ]

    # Compute entropy differences (predicted - labels) for both full and single
    for _, vals in matrix_data.items():
        ef = vals.get("entropy_full_mean", np.nan)
        es = vals.get("entropy_single_mean", np.nan)
        el = vals.get("entropy_labels_mean", np.nan)
        vals["entropy_diff_full"] = (
            (ef - el) if (not np.isnan(ef) and not np.isnan(el)) else np.nan
        )
        vals["entropy_diff_single"] = (
            (es - el) if (not np.isnan(es) and not np.isnan(el)) else np.nan
        )

    # Add entropy difference metrics
    entropy_metrics: List[tuple[str, str, bool, float, float, str | None]] = [
        (
            "entropy_diff_full",
            "Entropy Diff – Full (Pred − Label)",
            False,
            -1.0,
            1.0,
            "RdBu_r",
        ),
        (
            "entropy_diff_single",
            "Entropy Diff – Single (Pred − Label)",
            False,
            -1.0,
            1.0,
            "RdBu_r",
        ),
    ]
    metrics_to_create.extend(entropy_metrics)
    metrics_to_create.extend([])

    # Create all heatmaps
    for metric, title, reverse_scale, vmin, vmax, cmap_name in metrics_to_create:
        # Determine which folder based on metric type
        if (
            metric.endswith("_full")
            or metric.endswith("_full_mean")
            or "diff_full" in metric
        ):
            folder_name = "full_output"
        elif (
            metric.endswith("_single")
            or metric.endswith("_single_mean")
            or "diff_single" in metric
        ):
            folder_name = "single_token"
        else:
            folder_name = "other_metrics"

        create_heatmap(
            matrix_data,
            levels,
            metric,
            title,
            f"q1_heatmap_{metric}{modality_suffix}.png",
            reverse_scale=reverse_scale,
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
            modality=modality,
            subfolder=folder_name,
        )


def create_all_comparison_heatmaps(
    image_matrix_data: Dict, text_matrix_data: Dict, levels: list
):
    """Create all comparison heatmaps for different metrics."""

    print("🔍 Creating image vs text comparison heatmaps...")

    # Key metrics for comparison - comprehensive set
    metrics_to_compare = [
        (
            "brier_full_mean",
            "Brier Score - Full Output (Dark = Better)",
            True,
            0.0,
            0.2,
        ),
        (
            "brier_single_mean",
            "Brier Score - Single Token (Dark = Better)",
            True,
            0.0,
            0.2,
        ),
        (
            "entropy_full_mean",
            "Entropy - Full Output (Dark = More Certain)",
            True,
            1.0,
            1.6,
        ),
        (
            "entropy_single_mean",
            "Entropy - Single Token (Dark = More Certain)",
            True,
            0.0,
            0.15,
        ),
        ("f1_full", "F1 Score - Full Output (Dark = Better)", False, 0.0, 1.0),
        ("f1_single", "F1 Score - Single Token (Dark = Better)", False, 0.0, 1.0),
        ("accuracy_full", "Accuracy - Full Output (Dark = Better)", False, 0.0, 1.0),
        ("accuracy_single", "Accuracy - Single Token (Dark = Better)", False, 0.0, 1.0),
        (
            "mae_full",
            "Mean Absolute Error - Full Output (Dark = Better)",
            True,
            0.0,
            4.0,
        ),
        (
            "mae_single",
            "Mean Absolute Error - Single Token (Dark = Better)",
            True,
            0.0,
            4.0,
        ),
        (
            "label_prob_mass_full_mean",
            "Label Probability Mass - Full Output (Dark = Better)",
            False,
            0.0,
            1.0,
        ),
        (
            "label_prob_mass_single_mean",
            "Label Probability Mass - Single Token (Dark = Better)",
            False,
            0.0,
            1.0,
        ),
        (
            "cosine_similarity_full",
            "Cosine Similarity - Full Output (Dark = Better)",
            False,
            0.0,
            1.0,
        ),
        (
            "cosine_similarity_single",
            "Cosine Similarity - Single Token (Dark = Better)",
            False,
            0.0,
            1.0,
        ),
    ]

    # Compute entropy differences for comparison
    for matrix_data in [image_matrix_data, text_matrix_data]:
        for vals in matrix_data.values():
            ef = vals.get("entropy_full_mean", np.nan)
            es = vals.get("entropy_single_mean", np.nan)
            el = vals.get("entropy_labels_mean", np.nan)
            vals["entropy_diff_full"] = (
                (ef - el) if (not np.isnan(ef) and not np.isnan(el)) else np.nan
            )
            vals["entropy_diff_single"] = (
                (es - el) if (not np.isnan(es) and not np.isnan(el)) else np.nan
            )

    # Add entropy difference metrics
    metrics_to_compare.extend(
        [
            (
                "entropy_diff_full",
                "Entropy Diff – Full (Pred − Label)",
                False,
                -1.0,
                1.0,
            ),
            (
                "entropy_diff_single",
                "Entropy Diff – Single (Pred − Label)",
                False,
                -1.0,
                1.0,
            ),
        ]
    )

    # Create comparison heatmaps
    for metric, title, reverse_scale, vmin, vmax in metrics_to_compare:
        cmap_name = "RdBu_r" if "diff" in metric else None
        create_comparison_heatmap(
            image_matrix_data,
            text_matrix_data,
            levels,
            metric,
            title,
            f"q1_heatmap_{metric}_comparison.png",
            reverse_scale=reverse_scale,
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
        )


def print_pattern_analysis(matrix_data: Dict, levels: list):
    """Analyze patterns in the verbal vs facial matrix."""

    print("\n📊 PATTERN ANALYSIS:")
    print("=" * 50)

    # Collect metrics for analysis
    metrics = [
        "brier_full_mean",
        "brier_single_mean",
        "brier_full_std",
        "brier_single_std",
        "entropy_full_mean",
        "entropy_single_mean",
        "entropy_full_std",
        "entropy_single_std",
        "label_prob_mass_full_mean",
        "label_prob_mass_single_mean",
        "f1_full",
        "f1_single",
        "precision_full",
        "precision_single",
        "recall_full",
        "recall_single",
        "accuracy_full",
        "accuracy_single",
        "mae_full",
        "mae_single",
    ]

    for metric in metrics:
        print(f"\n{metric.upper()}:")

        # Create matrix for this metric
        matrix = np.full((len(levels), len(levels)), np.nan)
        for i, verbal in enumerate(levels):
            for j, facial in enumerate(levels):
                if (verbal, facial) in matrix_data:
                    matrix[i, j] = matrix_data[(verbal, facial)].get(metric, np.nan)

        # Print the matrix
        print("   Facial →")
        print("Verbal ↓    ", end="")
        for facial in levels:
            print(f"{facial:>8}", end="")
        print()

        for i, verbal in enumerate(levels):
            print(f"{verbal:>8}    ", end="")
            for j in range(len(levels)):
                value = matrix[i, j]
                if not np.isnan(value):
                    print(f"{value:>8.3f}", end="")
                else:
                    print(f"{'N/A':>8}", end="")
            print()

        # Analyze agreement vs disagreement
        diagonal = [
            matrix[i, i] for i in range(len(levels)) if not np.isnan(matrix[i, i])
        ]
        off_diagonal = [
            matrix[i, j]
            for i in range(len(levels))
            for j in range(len(levels))
            if i != j and not np.isnan(matrix[i, j])
        ]

        if diagonal and off_diagonal:
            print(f"   Agreement avg: {np.mean(diagonal):.3f}")
            print(f"   Disagreement avg: {np.mean(off_diagonal):.3f}")
            diff = np.mean(diagonal) - np.mean(off_diagonal)
            direction = (
                "better"
                if (
                    metric
                    in [
                        "f1_full",
                        "f1_single",
                        "precision_full",
                        "precision_single",
                        "recall_full",
                        "recall_single",
                        "accuracy_full",
                        "accuracy_single",
                        "label_prob_mass_full_mean",
                        "label_prob_mass_single_mean",
                    ]
                    and diff > 0
                )
                or (
                    metric
                    in [
                        "brier_full_mean",
                        "brier_single_mean",
                        "brier_full_std",
                        "brier_single_std",
                        "entropy_full_mean",
                        "entropy_single_mean",
                        "entropy_full_std",
                        "entropy_single_std",
                        "mae_full",
                        "mae_single",
                    ]
                    and diff < 0
                )
                else "worse"
            )
            print(f"   Agreement is {direction} (diff: {diff:.3f})")


def create_output_comparison_heatmaps(
    matrix_data: Dict, levels: list, modality: str = "image"
):
    """Create comparison heatmaps between single token and full output for a
    single modality."""

    # Create output directory relative to script location
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "results" / "heatmaps" / "output_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define metrics for comparison
    metrics_to_compare = [
        (
            "brier_full_mean",
            "brier_single_mean",
            "Brier Score (Dark = Better)",
            True,
            None,
            None,
        ),
        (
            "entropy_full_mean",
            "entropy_single_mean",
            "Entropy (Dark = More Certain)",
            True,
            None,
            None,
        ),
        ("f1_full", "f1_single", "F1 Score (Dark = Better)", False, 0.0, 1.0),
        (
            "accuracy_full",
            "accuracy_single",
            "Accuracy (Dark = Better)",
            False,
            0.0,
            1.0,
        ),
        (
            "mae_full",
            "mae_single",
            "Mean Absolute Error (Dark = Better)",
            True,
            0.0,
            1.0,
        ),
        (
            "label_prob_mass_full_mean",
            "label_prob_mass_single_mean",
            "Label Probability Mass (Dark = Better)",
            False,
            0.0,
            1.0,
        ),
        (
            "entropy_diff_full",
            "entropy_diff_single",
            "Entropy Diff (Pred - Label)",
            False,
            -1.0,
            1.0,
        ),
        (
            "cosine_similarity_full",
            "cosine_similarity_single",
            "Cosine Similarity (Higher = Better)",
            False,
            0.0,
            1.0,
        ),
    ]

    modality_title = "Images" if modality == "image" else "Text Descriptions"

    for (
        full_metric,
        single_metric,
        title,
        reverse_scale,
        vmin,
        vmax,
    ) in metrics_to_compare:
        # Create matrices for both output types
        full_matrix = np.full((len(levels), len(levels)), np.nan)
        single_matrix = np.full((len(levels), len(levels)), np.nan)
        full_annotations = np.full((len(levels), len(levels)), "", dtype=object)
        single_annotations = np.full((len(levels), len(levels)), "", dtype=object)

        # Determine std metric names if available
        full_std_metric = None
        single_std_metric = None
        if full_metric.endswith("_mean"):
            full_std_metric = full_metric.replace("_mean", "_std")
        elif full_metric in [
            "brier_full",
            "entropy_full",
            "mae_full",
            "label_prob_mass_full_mean",
            "entropy_diff_full",
        ]:
            if full_metric.endswith("_mean"):
                full_std_metric = full_metric.replace("_mean", "_std")
            else:
                full_std_metric = f"{full_metric}_std"

        if single_metric.endswith("_mean"):
            single_std_metric = single_metric.replace("_mean", "_std")
        elif single_metric in [
            "brier_single",
            "entropy_single",
            "mae_single",
            "label_prob_mass_single_mean",
            "entropy_diff_single",
        ]:
            if single_metric.endswith("_mean"):
                single_std_metric = single_metric.replace("_mean", "_std")
            else:
                single_std_metric = f"{single_metric}_std"

        # Debug: Check available keys for entropy_diff
        if "entropy_diff" in full_metric:
            sample_key = next(iter(matrix_data))
            available_keys = list(matrix_data[sample_key].keys())
            print(
                (
                    f"   DEBUG: Available keys in matrix_data: "
                    f"{[k for k in available_keys if 'entropy' in k]}"
                )
            )

        for i, verbal in enumerate(levels):
            for j, facial in enumerate(levels):
                if (verbal, facial) in matrix_data:
                    # Full output
                    full_value = matrix_data[(verbal, facial)].get(full_metric, np.nan)
                    full_matrix[i, j] = full_value

                    if not np.isnan(full_value):
                        if (
                            full_std_metric
                            and full_std_metric in matrix_data[(verbal, facial)]
                        ):
                            full_std_value = matrix_data[(verbal, facial)].get(
                                full_std_metric, np.nan
                            )
                            if not np.isnan(full_std_value):
                                full_annotations[i, j] = (
                                    f"{full_value:.3f}\n±{full_std_value:.3f}"
                                )
                            else:
                                full_annotations[i, j] = f"{full_value:.3f}"
                        else:
                            full_annotations[i, j] = f"{full_value:.3f}"
                    else:
                        full_annotations[i, j] = "N/A"

                    # Single token
                    single_value = matrix_data[(verbal, facial)].get(
                        single_metric, np.nan
                    )
                    single_matrix[i, j] = single_value

                    if not np.isnan(single_value):
                        if (
                            single_std_metric
                            and single_std_metric in matrix_data[(verbal, facial)]
                        ):
                            single_std_value = matrix_data[(verbal, facial)].get(
                                single_std_metric, np.nan
                            )
                            if not np.isnan(single_std_value):
                                single_annotations[i, j] = (
                                    f"{single_value:.3f}\n±{single_std_value:.3f}"
                                )
                            else:
                                single_annotations[i, j] = f"{single_value:.3f}"
                        else:
                            single_annotations[i, j] = f"{single_value:.3f}"
                    else:
                        single_annotations[i, j] = "N/A"

        # Calculate unified color scale
        combined_values = []
        for matrix in [full_matrix, single_matrix]:
            valid_values = matrix[~np.isnan(matrix)]
            if len(valid_values) > 0:
                combined_values.extend(valid_values)

        if len(combined_values) > 0:
            combined_min, combined_max = np.min(combined_values), np.max(
                combined_values
            )
            if vmin is None:
                vmin = combined_min
            if vmax is None:
                vmax = combined_max

        # Create DataFrames for better labeling
        df_full = pd.DataFrame(
            full_matrix,
            index=[f"Verbal: {level.title()}" for level in levels],
            columns=[f"Facial: {level.title()}" for level in levels],
        )

        df_single = pd.DataFrame(
            single_matrix,
            index=[f"Verbal: {level.title()}" for level in levels],
            columns=[f"Facial: {level.title()}" for level in levels],
        )

        # Create side-by-side subplots with better spacing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        # Choose colormap
        if "entropy_diff" in full_metric:
            cmap = "RdBu_r"  # Diverging colormap for entropy difference
        else:
            cmap = "Blues_r" if reverse_scale else "Blues"

        # Create heatmaps with unified color scale and custom annotations
        sns.heatmap(
            df_full,
            annot=full_annotations,
            fmt="",  # Use empty format since we're providing custom annotations
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=False,  # We'll add a single colorbar later
            linewidths=0.5,
            ax=ax1,
        )

        sns.heatmap(
            df_single,
            annot=single_annotations,
            fmt="",  # Use empty format since we're providing custom annotations
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar_kws={"shrink": 0.8},
            linewidths=0.5,
            ax=ax2,
        )

        # Set subplot titles
        ax1.set_title("Full Output", fontsize=12, fontweight="bold", pad=10)
        ax2.set_title("Single Token", fontsize=12, fontweight="bold", pad=10)

        # Set main title with better positioning
        fig.suptitle(
            (
                f"Facial Expressions as {modality_title}\n"
                f"Full vs Single Output Comparison: {title}"
            ),
            fontsize=14,
            fontweight="bold",
            y=0.95,
        )

        # Adjust spacing - minimal space between plots and make room for titles
        plt.subplots_adjust(left=0.08, right=0.92, top=0.82, bottom=0.15, wspace=0)

        # Save plot to organized directory
        metric_name = full_metric.replace("_mean", "").replace("_full", "")
        filename = f"q1_heatmap_{metric_name}_output_comparison_{modality}.png"
        full_path = output_dir / filename
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved output comparison heatmap: {full_path}")

        # Print comparison analysis
        full_values = full_matrix[~np.isnan(full_matrix)]
        single_values = single_matrix[~np.isnan(single_matrix)]

        # Debug: Check if we have any values for entropy_diff
        if "entropy_diff" in full_metric:
            print(f"   DEBUG: Full matrix has {len(full_values)} non-NaN values")
            print(f"   DEBUG: Single matrix has {len(single_values)} non-NaN values")
            if len(full_values) > 0:
                print(
                    (
                        f"   DEBUG: Full values range: {np.min(full_values):.3f} "
                        f"to {np.max(full_values):.3f}"
                    )
                )
            if len(single_values) > 0:
                print(
                    (
                        f"   DEBUG: Single values range: {np.min(single_values):.3f} "
                        f"to {np.max(single_values):.3f}"
                    )
                )

        if len(full_values) > 0 and len(single_values) > 0:
            full_mean = np.mean(full_values)
            single_mean = np.mean(single_values)

            # Determine which is better based on metric type
            if reverse_scale:  # Lower is better (Brier, MAE, Entropy)
                better_output = "Full" if full_mean < single_mean else "Single"
            else:  # Higher is better (F1, Accuracy, Label Prob Mass)
                better_output = "Full" if full_mean > single_mean else "Single"

            print(f"   {metric_name} Output Comparison Analysis:")
            print(f"     Full Output mean: {full_mean:.3f}")
            print(f"     Single Token mean: {single_mean:.3f}")
            print(f"     Better overall performance: {better_output}")
        else:
            print(f"   {metric_name} Output Comparison Analysis: No valid data found")


def main(
    create_comparison: bool = False,
    use_image: bool = False,
    use_text: bool = False,
    comparison_type: str = "modality",
):
    """Main function to create verbal vs facial expression heatmaps.

    Args:
        create_comparison: If True, create comparison heatmaps
        use_image: If True, create heatmaps for image modality only
        use_text: If True, create heatmaps for text modality only
        comparison_type: Type of comparison - "modality" (image vs text)
            or "output" (single vs full)
    """

    print("🚀 Starting verbal vs facial expression heatmap analysis...")

    if create_comparison:
        if comparison_type == "modality":
            print(
                "📊 Loading Q1 analysis results for both image and text modalities..."
            )
            image_stats, text_stats = load_both_modalities()

            # Create matrices for both modalities
            print("🔄 Creating verbal vs facial expression matrices...")
            image_matrix_data, levels = create_verbal_facial_matrix(image_stats)
            text_matrix_data, _ = create_verbal_facial_matrix(text_stats)

            # Create modality comparison heatmaps
            create_all_comparison_heatmaps(image_matrix_data, text_matrix_data, levels)

            print("\n✅ Modality comparison heatmap analysis completed!")
            print("   Generated comparison heatmaps showing image vs text performance")
            print(
                (
                    "   Side-by-side plots with unified color scaling "
                    "for direct comparison"
                )
            )

        elif comparison_type == "output":
            print(
                "📊 Loading Q1 analysis results for single vs full output comparison..."
            )
            # Default to image modality for output comparison, but could be
            # extended to support --modality flag
            modality = "image"  # Could make this configurable in the future
            results_dir = f"{modality}_facial_expression_results"

            summary_stats = load_q1_analysis_results(results_dir)

            # Create verbal vs facial matrix
            print("🔄 Creating verbal vs facial expression matrix...")
            matrix_data, levels = create_verbal_facial_matrix(summary_stats)

            # Create output comparison heatmaps
            print(
                (
                    f"\n🔍 Creating single vs full output comparison heatmaps "
                    f"for {modality} modality..."
                )
            )
            create_output_comparison_heatmaps(matrix_data, levels, modality)

            print("\n✅ Output comparison heatmap analysis completed!")
            print(
                (
                    f"   Generated comparison heatmaps showing single vs full "
                    f"output for {modality} modality"
                )
            )
            print(
                (
                    "   Side-by-side plots with unified color scaling "
                    "for direct comparison"
                )
            )

    elif use_image:
        # Load analysis results for image modality
        print("📊 Loading Q1 analysis results for image modality...")
        summary_stats = load_q1_analysis_results("image_facial_expression_results")

        # Create verbal vs facial matrix
        print("🔄 Creating verbal vs facial expression matrix...")
        matrix_data, levels = create_verbal_facial_matrix(summary_stats)

        # Print pattern analysis
        print_pattern_analysis(matrix_data, levels)

        # Create all heatmaps
        create_all_heatmaps(matrix_data, levels, modality="image")

        print("\n✅ Image modality heatmap analysis completed!")
        print(
            (
                "   Generated heatmaps showing performance across "
                "verbal/facial combinations"
            )
        )
        print(
            "   Diagonal = agreement scenarios, off-diagonal = disagreement scenarios"
        )

    elif use_text:
        # Load analysis results for text modality
        print("📊 Loading Q1 analysis results for text modality...")
        summary_stats = load_q1_analysis_results("text_facial_expression_results")

        # Create verbal vs facial matrix
        print("🔄 Creating verbal vs facial expression matrix...")
        matrix_data, levels = create_verbal_facial_matrix(summary_stats)

        # Print pattern analysis
        print_pattern_analysis(matrix_data, levels)

        # Create all heatmaps
        create_all_heatmaps(matrix_data, levels, modality="text")

        print("\n✅ Text modality heatmap analysis completed!")
        print(
            (
                "   Generated heatmaps showing performance across "
                "verbal/facial combinations"
            )
        )
        print(
            "   Diagonal = agreement scenarios, off-diagonal = disagreement scenarios"
        )

    else:
        print(
            "❌ Error: Please specify a modality using --image, --text, or --comparison"
        )
        print("   Examples:")
        print("     python create_verbal_facial_heatmaps.py --image")
        print("     python create_verbal_facial_heatmaps.py --text")
        print("     python create_verbal_facial_heatmaps.py --comparison")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create heatmaps for Q1 verbal vs facial expression analysis"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Create side-by-side comparison heatmaps",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Create heatmaps for image facial expression modality only",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Create heatmaps for text facial expression modality only",
    )
    parser.add_argument(
        "--type",
        choices=["modality", "output"],
        default="modality",
        help=(
            "Type of comparison: 'modality' (image vs text) "
            "or 'output' (single vs full)"
        ),
    )

    args = parser.parse_args()

    # Check for conflicting arguments
    flags_set = sum([args.comparison, args.image, args.text])
    if flags_set > 1:
        print("❌ Error: Please specify only one of --comparison, --image, or --text")
        parser.print_help()
        sys.exit(1)

    # Validate argument combination
    if args.type == "output" and not args.comparison:
        print("❌ Error: --type output requires --comparison flag")
        parser.print_help()
        sys.exit(1)

    main(
        create_comparison=args.comparison,
        use_image=args.image,
        use_text=args.text,
        comparison_type=args.type,
    )
