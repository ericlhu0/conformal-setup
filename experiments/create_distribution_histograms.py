#!/usr/bin/env python3
"""Create histograms showing average prediction distributions for each
verbal/facial combination."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append(".")

from analyze_q1_disagreement import (
    analyze_brier_scores_by_experiment_type,
    is_q1_scenario,
    organize_predictions_by_scenario,
    parse_scenario_classification,
)


def load_q1_data_from_results_dir(results_dir: str) -> tuple:
    """Load Q1 data from a specific results directory."""
    curr_dir = Path(__file__).resolve().parent

    # Load expert labels from experiment config
    config_path = curr_dir / "configs/experiment_1_disagreement.json"
    if not config_path.exists():
        raise FileNotFoundError(f"ERROR: Q1 config file not found at {config_path}")

    with open(config_path, "r") as f:
        expert_data = json.load(f)

    # Load model predictions from specified results directory
    final_results_path = curr_dir / "results" / results_dir / "final_results.json"
    if not final_results_path.exists():
        raise FileNotFoundError(
            f"ERROR: Final results file not found at {final_results_path}"
        )

    with open(final_results_path, "r") as f:
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

    return expert_data, final_results_list, scenarios_by_type


def load_q1_analysis_results(
    results_dir: str = "image_facial_expression_results",
) -> Dict[str, Any]:
    """Load and run Q1 analysis to get results from specified directory."""
    # Load data and run analysis
    expert_data, final_results_list, scenarios_by_type = load_q1_data_from_results_dir(
        results_dir
    )
    organized_predictions = organize_predictions_by_scenario(final_results_list)
    summary_stats = analyze_brier_scores_by_experiment_type(
        organized_predictions, scenarios_by_type
    )

    return summary_stats, organized_predictions, scenarios_by_type


def calculate_average_distributions(
    organized_predictions: Dict, scenarios_by_type: Dict
) -> Dict[str, Dict]:
    """Calculate average prediction distributions and label distributions for
    each experiment type."""

    distributions_by_type = {}
    levels = ["none", "mid", "high"]

    # Group scenarios by verbal/facial combination
    for scenario_name, predictions in organized_predictions.items():
        if scenario_name not in scenarios_by_type:
            continue

        scenario_info = scenarios_by_type[scenario_name]
        classification = scenario_info["classification"]
        experiment_type = classification["scenario_description"]
        verbal_level = classification["verbal_level"]
        facial_level = classification["facial_level"]
        expert_labels = scenario_info["expert_labels"]

        # Only process the 9 main combinations
        if verbal_level in levels and facial_level in levels:
            exp_key = f"verbal_{verbal_level}_facial_{facial_level}"

            if exp_key not in distributions_by_type:
                distributions_by_type[exp_key] = {
                    "full_distributions": [],
                    "single_distributions": [],
                    "label_distributions": [],
                    "verbal_level": verbal_level,
                    "facial_level": facial_level,
                }

            # Collect wrist labels (same for both query types, so add once)
            if "wrist" in expert_labels:
                label_probs = expert_labels["wrist"]
                # Convert to consistent format (class 1-5 with probabilities)
                label_array = [0.0] * 5  # Initialize for classes 1-5
                for class_str, prob in label_probs.items():
                    class_idx = int(class_str) - 1  # Convert to 0-4 index
                    if 0 <= class_idx < 5:
                        label_array[class_idx] = float(prob)
                distributions_by_type[exp_key]["label_distributions"].append(
                    label_array
                )

            # Collect wrist predictions for both query types
            for query_type in ["body_part_full", "body_part_single"]:
                if query_type in predictions and "wrist" in predictions[query_type]:
                    prediction_data = predictions[query_type]["wrist"]
                    predicted_probs = prediction_data["cleaned_probabilities"]

                    # Convert to consistent format (class 1-5 with probabilities)
                    prob_array = [0.0] * 5  # Initialize for classes 1-5
                    for class_str, prob in predicted_probs.items():
                        class_idx = int(class_str) - 1  # Convert to 0-4 index
                        if 0 <= class_idx < 5:
                            prob_array[class_idx] = float(prob)

                    if query_type == "body_part_full":
                        distributions_by_type[exp_key]["full_distributions"].append(
                            prob_array
                        )
                    else:
                        distributions_by_type[exp_key]["single_distributions"].append(
                            prob_array
                        )

    # Calculate average distributions
    avg_distributions = {}
    for exp_key, data in distributions_by_type.items():
        avg_distributions[exp_key] = {
            "verbal_level": data["verbal_level"],
            "facial_level": data["facial_level"],
        }

        # Calculate averages and standard deviations for labels
        if data["label_distributions"]:
            label_arrays = np.array(data["label_distributions"])
            avg_distributions[exp_key]["label_avg"] = np.mean(label_arrays, axis=0)
            avg_distributions[exp_key]["label_std"] = (
                np.std(label_arrays, axis=0, ddof=1)
                if len(label_arrays) > 1
                else np.zeros(5)
            )
            avg_distributions[exp_key]["label_count"] = len(data["label_distributions"])
        else:
            avg_distributions[exp_key]["label_avg"] = np.zeros(5)
            avg_distributions[exp_key]["label_std"] = np.zeros(5)
            avg_distributions[exp_key]["label_count"] = 0

        # Calculate averages and standard deviations for full output
        if data["full_distributions"]:
            full_arrays = np.array(data["full_distributions"])
            avg_distributions[exp_key]["full_avg"] = np.mean(full_arrays, axis=0)
            avg_distributions[exp_key]["full_std"] = (
                np.std(full_arrays, axis=0, ddof=1)
                if len(full_arrays) > 1
                else np.zeros(5)
            )
            avg_distributions[exp_key]["full_count"] = len(data["full_distributions"])
        else:
            avg_distributions[exp_key]["full_avg"] = np.zeros(5)
            avg_distributions[exp_key]["full_std"] = np.zeros(5)
            avg_distributions[exp_key]["full_count"] = 0

        # Calculate averages and standard deviations for single token
        if data["single_distributions"]:
            single_arrays = np.array(data["single_distributions"])
            avg_distributions[exp_key]["single_avg"] = np.mean(single_arrays, axis=0)
            avg_distributions[exp_key]["single_std"] = (
                np.std(single_arrays, axis=0, ddof=1)
                if len(single_arrays) > 1
                else np.zeros(5)
            )
            avg_distributions[exp_key]["single_count"] = len(
                data["single_distributions"]
            )
        else:
            avg_distributions[exp_key]["single_avg"] = np.zeros(5)
            avg_distributions[exp_key]["single_std"] = np.zeros(5)
            avg_distributions[exp_key]["single_count"] = 0

    return avg_distributions


def create_single_plot(
    avg_distributions: Dict, output_type: str, modality: str = "image"
):
    """Create a single histogram plot."""

    # Create output directory relative to script location
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "results" / "histograms"
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = ["none", "mid", "high"]
    class_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    if output_type == "labels":
        fig.suptitle(
            f"Average True Label Distributions\n",
            fontsize=16,
            fontweight="bold",
        )
    else:
        modality_title = "Images" if modality == "image" else "Text Descriptions"
        fig.suptitle(
            f"Facial Expressions as {modality_title}\n{output_type.title()} Output\n",
            fontsize=16,
            fontweight="bold"
        )

    for i, verbal_level in enumerate(levels):
        for j, facial_level in enumerate(levels):
            ax = axes[i, j]
            exp_key = f"verbal_{verbal_level}_facial_{facial_level}"

            if exp_key in avg_distributions:
                if output_type == "labels":
                    probabilities = avg_distributions[exp_key]["label_avg"]
                    std_values = avg_distributions[exp_key]["label_std"]
                    count = avg_distributions[exp_key]["label_count"]
                    color = "orange"

                    # Create bar plot for labels only with error bars
                    bars = ax.bar(
                        class_labels,
                        probabilities,
                        width=2 * 0.8,
                        color=color,
                        alpha=0.7,
                        edgecolor="navy",
                        yerr=std_values,
                        capsize=3,
                        error_kw={"linewidth": 1, "capthick": 1},
                    )

                    # Add value labels on bars
                    for bar, prob in zip(bars, probabilities):
                        if prob > 0.01:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.01,
                                f"{prob:.2f}",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                            )
                else:
                    # Get prediction probabilities and standard deviations
                    if output_type == "full":
                        pred_probabilities = avg_distributions[exp_key]["full_avg"]
                        pred_std_values = avg_distributions[exp_key]["full_std"]
                        count = avg_distributions[exp_key]["full_count"]
                        color = "skyblue"
                    else:
                        pred_probabilities = avg_distributions[exp_key]["single_avg"]
                        pred_std_values = avg_distributions[exp_key]["single_std"]
                        count = avg_distributions[exp_key]["single_count"]
                        color = "lightgreen"

                    # Get label probabilities
                    label_probabilities = avg_distributions[exp_key]["label_avg"]

                    # Create bar plot for predictions with error bars
                    pred_bars = ax.bar(
                        class_labels,
                        pred_probabilities,
                        color=color,
                        alpha=0.7,
                        edgecolor="navy",
                        label="Predictions",
                        yerr=pred_std_values,
                        capsize=3,
                        error_kw={"linewidth": 1, "capthick": 1},
                    )

                    # Draw red horizontal lines (tick marks) for true labels at each class position
                    for k, prob in enumerate(label_probabilities):
                        if prob > 0.001:  # Only draw lines for non-zero probabilities
                            # Draw horizontal line spanning the width of the bar
                            ax.hlines(
                                y=prob,
                                xmin=k - 0.4,
                                xmax=k + 0.4,
                                colors="red",
                                linewidth=4,
                                alpha=0.8,
                            )

                    # Add a single invisible line for legend
                    ax.hlines(
                        y=[],
                        xmin=[],
                        xmax=[],
                        colors="red",
                        linewidth=4,
                        alpha=0.8,
                        label="True Labels",
                    )

                    # Add value labels on prediction bars
                    for bar, prob in zip(pred_bars, pred_probabilities):
                        if prob > 0.01:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.01,
                                f"{prob:.2f}",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                            )

                    # Add legend
                    ax.legend(loc="upper right", fontsize=8)

                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Average Probability")
                ax.set_xlabel("Comfort Threshold")
                ax.set_title(
                    f"Verbal: {verbal_level.title()}, Facial: {facial_level.title()}\n(n={count})",
                    fontsize=10,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Average Probability")
                ax.set_xlabel("Comfort Threshold")
                ax.set_ylabel("Average Probability")
                ax.set_xlabel("Comfort Threshold")

    # Add axis labels for the grid structure
    fig.text(0.02, 0.5, 'Verbal Expressed Discomfort', va='center', rotation='vertical', 
             fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02, 'Facial Expressed Discomfort', ha='center', 
             fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.08, hspace=0.4)  # Make room for axis labels
    filename = f"q1_distribution_histograms_{output_type}_{modality}.png"
    full_path = output_dir / filename
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved histogram plot: {full_path}")


def create_comparison_plot(
    avg_distributions_image: Dict, avg_distributions_text: Dict, output_type: str
):
    """Create side-by-side comparison histogram plot between image and text
    modalities."""

    # Create output directory relative to script location
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "results" / "histograms" / "modality_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = ["none", "mid", "high"]
    class_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(
        f"Image vs Text Facial Expression Comparison\n{output_type.title()} Output\n",
        fontsize=16,
        fontweight="bold",
    )

    for i, verbal_level in enumerate(levels):
        for j, facial_level in enumerate(levels):
            ax = axes[i, j]
            exp_key = f"verbal_{verbal_level}_facial_{facial_level}"

            if exp_key in avg_distributions_image and exp_key in avg_distributions_text:
                # Get prediction probabilities and standard deviations for both modalities
                if output_type == "full":
                    image_pred = avg_distributions_image[exp_key]["full_avg"]
                    image_std = avg_distributions_image[exp_key]["full_std"]
                    text_pred = avg_distributions_text[exp_key]["full_avg"]
                    text_std = avg_distributions_text[exp_key]["full_std"]
                    image_count = avg_distributions_image[exp_key]["full_count"]
                    text_count = avg_distributions_text[exp_key]["full_count"]
                else:
                    image_pred = avg_distributions_image[exp_key]["single_avg"]
                    image_std = avg_distributions_image[exp_key]["single_std"]
                    text_pred = avg_distributions_text[exp_key]["single_avg"]
                    text_std = avg_distributions_text[exp_key]["single_std"]
                    image_count = avg_distributions_image[exp_key]["single_count"]
                    text_count = avg_distributions_text[exp_key]["single_count"]

                # Get label probabilities (same for both)
                label_probabilities = avg_distributions_image[exp_key]["label_avg"]

                # Create side-by-side bar plot
                x = np.arange(len(class_labels))
                width = 0.35  # Width of the bars

                # Create bars with error bars
                image_bars = ax.bar(
                    x - width / 2,
                    image_pred,
                    width,
                    color="skyblue",
                    alpha=0.7,
                    edgecolor="navy",
                    label="Image",
                    yerr=image_std,
                    capsize=2,
                    error_kw={"linewidth": 1, "capthick": 1},
                )
                text_bars = ax.bar(
                    x + width / 2,
                    text_pred,
                    width,
                    color="lightcoral",
                    alpha=0.7,
                    edgecolor="darkred",
                    label="Text",
                    yerr=text_std,
                    capsize=2,
                    error_kw={"linewidth": 1, "capthick": 1},
                )

                # Draw red horizontal lines (tick marks) for true labels at each class position
                for k, prob in enumerate(label_probabilities):
                    if prob > 0.001:  # Only draw lines for non-zero probabilities
                        # Draw horizontal line spanning both bars
                        ax.hlines(
                            y=prob,
                            xmin=k - width * 0.6,
                            xmax=k + width * 0.6,
                            colors="red",
                            linewidth=4,
                            alpha=0.8,
                        )

                # Add a single invisible line for legend
                ax.hlines(
                    y=[],
                    xmin=[],
                    xmax=[],
                    colors="red",
                    linewidth=4,
                    alpha=0.8,
                    label="True Labels",
                )

                # Add value labels on bars (smaller font for comparison plots)
                for bar, prob in zip(image_bars, image_pred):
                    if prob > 0.01:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{prob:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=6,
                        )

                for bar, prob in zip(text_bars, text_pred):
                    if prob > 0.01:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{prob:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=6,
                        )

                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Average Probability")
                ax.set_xlabel("Comfort Threshold")
                ax.set_title(
                    f"Verbal: {verbal_level.title()}, Facial: {facial_level.title()}\n"
                    f"Image(n={image_count}), Text(n={text_count})",
                    fontsize=10,
                )
                ax.set_xticks(x)
                ax.set_xticklabels(class_labels)
                ax.legend(loc="upper right", fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Average Probability")
                ax.set_xlabel("Comfort Threshold")

    # Add axis labels for the grid structure
    fig.text(0.02, 0.5, 'Verbal Expressed Discomfort', va='center', rotation='vertical', 
             fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02, 'Facial Expressed Discomfort', ha='center', 
             fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.08, hspace=0.4)  # Make room for axis labels
    filename = f"q1_distribution_histograms_{output_type}_comparison.png"
    full_path = output_dir / filename
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved comparison histogram plot: {full_path}")


def create_output_comparison_plot(avg_distributions: Dict, modality: str = "image"):
    """Create side-by-side comparison histogram plot between single and full
    output."""

    # Create output directory relative to script location
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "results" / "histograms" / "output_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = ["none", "mid", "high"]
    class_labels = ["Very Low", "Low", "Medium", "High", "Very High"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    modality_title = "Images" if modality == "image" else "Text Descriptions"
    fig.suptitle(
        f"Facial Expressions as {modality_title}\nSingle vs Full Output Comparison\n",
        fontsize=16,
        fontweight="bold",
    )

    for i, verbal_level in enumerate(levels):
        for j, facial_level in enumerate(levels):
            ax = axes[i, j]
            exp_key = f"verbal_{verbal_level}_facial_{facial_level}"

            if exp_key in avg_distributions:
                # Get prediction probabilities and standard deviations for both output types
                single_pred = avg_distributions[exp_key]["single_avg"]
                single_std = avg_distributions[exp_key]["single_std"]
                full_pred = avg_distributions[exp_key]["full_avg"]
                full_std = avg_distributions[exp_key]["full_std"]
                single_count = avg_distributions[exp_key]["single_count"]
                full_count = avg_distributions[exp_key]["full_count"]

                # Get label probabilities
                label_probabilities = avg_distributions[exp_key]["label_avg"]

                # Create side-by-side bar plot
                x = np.arange(len(class_labels))
                width = 0.35  # Width of the bars

                # Create bars with error bars
                single_bars = ax.bar(
                    x - width / 2,
                    single_pred,
                    width,
                    color="lightgreen",
                    alpha=0.7,
                    edgecolor="darkgreen",
                    label="Single Token",
                    yerr=single_std,
                    capsize=2,
                    error_kw={"linewidth": 1, "capthick": 1},
                )
                full_bars = ax.bar(
                    x + width / 2,
                    full_pred,
                    width,
                    color="skyblue",
                    alpha=0.7,
                    edgecolor="navy",
                    label="Full Output",
                    yerr=full_std,
                    capsize=2,
                    error_kw={"linewidth": 1, "capthick": 1},
                )

                # Draw red horizontal lines (tick marks) for true labels at each class position
                for k, prob in enumerate(label_probabilities):
                    if prob > 0.001:  # Only draw lines for non-zero probabilities
                        # Draw horizontal line spanning both bars
                        ax.hlines(
                            y=prob,
                            xmin=k - width * 0.6,
                            xmax=k + width * 0.6,
                            colors="red",
                            linewidth=4,
                            alpha=0.8,
                        )

                # Add a single invisible line for legend
                ax.hlines(
                    y=[],
                    xmin=[],
                    xmax=[],
                    colors="red",
                    linewidth=4,
                    alpha=0.8,
                    label="True Labels",
                )

                # Add value labels on bars (smaller font for comparison plots)
                for bar, prob in zip(single_bars, single_pred):
                    if prob > 0.01:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{prob:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=6,
                        )

                for bar, prob in zip(full_bars, full_pred):
                    if prob > 0.01:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{prob:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=6,
                        )

                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Average Probability")
                ax.set_xlabel("Comfort Threshold")
                ax.set_title(
                    f"Verbal: {verbal_level.title()}, Facial: {facial_level.title()}\n"
                    f"Single(n={single_count}), Full(n={full_count})",
                    fontsize=10,
                )
                ax.set_xticks(x)
                ax.set_xticklabels(class_labels)
                ax.legend(loc="upper right", fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("Average Probability")
                ax.set_xlabel("Comfort Threshold")

    # Add axis labels for the grid structure
    fig.text(0.02, 0.5, 'Verbal Expressed Discomfort', va='center', rotation='vertical', 
             fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02, 'Facial Expressed Discomfort', ha='center', 
             fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.08, hspace=0.4)  # Make room for axis labels
    filename = f"q1_distribution_histograms_output_comparison_{modality}.png"
    full_path = output_dir / filename
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved output comparison histogram plot: {full_path}")


def main(create_comparison: bool = False, comparison_type: str = "modality"):
    """Main function to create distribution histograms.

    Args:
        create_comparison: If True, create comparison plots
        comparison_type: Type of comparison - "modality" (image vs text) or "output" (single vs full)
    """

    print("🚀 Starting distribution histogram analysis...")

    if create_comparison:
        if comparison_type == "modality":
            print(
                "📊 Loading Q1 analysis results for both image and text modalities..."
            )

            # Load image results
            print("  Loading image facial expression results...")
            (
                summary_stats_image,
                organized_predictions_image,
                scenarios_by_type_image,
            ) = load_q1_analysis_results("image_facial_expression_results")
            avg_distributions_image = calculate_average_distributions(
                organized_predictions_image, scenarios_by_type_image
            )

            # Load text results
            print("  Loading text facial expression results...")
            summary_stats_text, organized_predictions_text, scenarios_by_type_text = (
                load_q1_analysis_results("text_facial_expression_results")
            )
            avg_distributions_text = calculate_average_distributions(
                organized_predictions_text, scenarios_by_type_text
            )

            # Create modality comparison plots
            print("\n🔍 Creating image vs text comparison histograms...")
            for output_type in ["full", "single"]:
                create_comparison_plot(
                    avg_distributions_image, avg_distributions_text, output_type
                )

            print(f"\n✅ Modality comparison histogram analysis completed!")
            print(
                f"   Generated 2 comparison histogram plots with side-by-side bars and red tick marks for labels"
            )

        elif comparison_type == "output":
            print(
                "📊 Loading Q1 analysis results for single vs full output comparison..."
            )
            # Default to image modality for output comparison, but could be extended to support --modality flag
            modality = "image"  # Could make this configurable in the future
            results_dir = f"{modality}_facial_expression_results"

            summary_stats, organized_predictions, scenarios_by_type = (
                load_q1_analysis_results(results_dir)
            )
            avg_distributions = calculate_average_distributions(
                organized_predictions, scenarios_by_type
            )

            # Create output comparison plot
            print(
                f"\n🔍 Creating single vs full output comparison histogram for {modality} modality..."
            )
            create_output_comparison_plot(avg_distributions, modality)

            print(f"\n✅ Output comparison histogram analysis completed!")
            print(
                f"   Generated 1 comparison histogram plot showing single vs full output for {modality} modality"
            )

    else:
        # Load analysis results for default (image) modality
        print("📊 Loading Q1 analysis results...")
        summary_stats, organized_predictions, scenarios_by_type = (
            load_q1_analysis_results()
        )

        # Calculate average distributions
        print("🔄 Calculating average distributions...")
        avg_distributions = calculate_average_distributions(
            organized_predictions, scenarios_by_type
        )

        # Create histogram plots
        print("\n🔍 Creating distribution histograms...")
        for output_type in ["full", "single"]:
            create_single_plot(avg_distributions, output_type, "image")

        print(f"\n✅ Distribution histogram analysis completed!")
        print(f"   Generated 2 histogram plots with red tick marks for labels")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create distribution histograms for Q1 analysis"
    )
    parser.add_argument(
        "--comparison", action="store_true", help="Create side-by-side comparison plots"
    )
    parser.add_argument(
        "--type",
        choices=["modality", "output"],
        default="modality",
        help="Type of comparison: 'modality' (image vs text) or 'output' (single vs full)",
    )

    args = parser.parse_args()

    # Validate argument combination
    if args.type == "output" and not args.comparison:
        print("❌ Error: --type output requires --comparison flag")
        parser.print_help()
        exit(1)

    main(create_comparison=args.comparison, comparison_type=args.type)
