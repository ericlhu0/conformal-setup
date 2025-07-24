"""Vizualization Utilities."""

import json
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_experiments(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group experiments by model and text_input (base prompt)"""
    groups = defaultdict(list)

    for experiment in results:
        model = experiment["config"]["model"]
        text_input = experiment["config"]["text_input"]
        sensitivity_spec = experiment["config"]["sensitivity_spec"]
        key = f"{model}_{hash(text_input)}_{hash(sensitivity_spec)}"
        groups[key].append(experiment)

    print(f"Grouped {len(groups)} unique experiments based on model and text input.")

    return groups


def plot_probability_distributions(group_data: List[Dict[str, Any]]) -> Figure:
    """Create visualization for a group of experiments."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 2 * len(group_data)))

    for i, experiment in enumerate(group_data):
        results = experiment["results"]
        config = experiment["config"]

        # Get all tokens and their probabilities
        tokens = tuple(results.keys())
        probs = tuple(results.values())

        # Sort by probability
        sorted_pairs = sorted(zip(tokens, probs), key=lambda x: x[1], reverse=True)
        tokens, probs = tuple(zip(*sorted_pairs)) if sorted_pairs else ((), ())

        # Create horizontal stacked bar representing the full probability distribution
        left = 0
        y_pos = i

        for token, prob in zip(tokens, probs):
            ax.barh(
                y_pos,
                prob,
                left=left,
                height=0.9,
                color="lightgray",
                edgecolor="black",
                linewidth=0.5,
            )

            # Add text label in the middle of each segment if segment is large enough
            if prob > 0.01:  # Only show text for segments larger than 1%
                text_x = left + prob / 2
                ax.text(
                    text_x,
                    y_pos,
                    f"{token}\n{prob:.3e}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    weight="bold",
                )

            left += prob

        # Create label for each bar based on modality condition
        if config.get("expression_image"):
            modality = "Text"
            expression_desc = (
                config.get("expression_text", "N/A")
                + "\n"
                + "Image: "
                + config.get("expression_image", "")
            )
        else:
            modality = "Text"
            expression_desc = (
                config.get("expression_text", "N/A") + "\n" + "Image: " + "No Image"
            )

        bar_label = f"{modality}: {expression_desc}"
        ax.text(-0.05, y_pos, bar_label, ha="right", va="center", fontsize=10)

    # Set axis properties
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(group_data) - 0.5)
    ax.set_xlabel("Probability")
    ax.set_yticks([])
    ax.set_ylabel("")

    # Create overall title from first experiment's text_input
    base_prompt = group_data[0]["config"]["text_input"]
    model = group_data[0]["config"]["model"]

    wrapped_prompt = textwrap.fill(base_prompt, width=80)
    title_text = f"Model: {model}\nPrompt: {wrapped_prompt}"

    plt.title(title_text, fontsize=12, loc="left", pad=20)
    plt.tight_layout()
    return fig


def visualize_results(results_file: str) -> List[Tuple[str, Figure]]:
    """Main function to create visualizations."""
    results = load_results(results_file)
    groups = group_experiments(results)

    figures = []
    for group_name, group_data in groups.items():
        fig = plot_probability_distributions(group_data)
        figures.append((group_name, fig))

    return figures


if __name__ == "__main__":
    # Example usage
    example_results_file = "/multirun/2025-07-23/17-01-40/results.json"
    example_figures = visualize_results(example_results_file)

    # Save and show figures
    for idx, (example_group_name, example_fig) in enumerate(example_figures):
        example_fig.savefig(f"results_{idx+1}.png")
        plt.show()
