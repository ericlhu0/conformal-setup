"""Vizualization Utilities."""

import json
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Tuple

import colorhash
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_experiments(
    results: List[Dict[str, Any]], grouping_keys: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """Group experiments by model and text_input (base prompt)"""
    groups = defaultdict(list)

    for experiment in results:
        # model = experiment["config"]["model"]
        # sensitivity_spec = experiment["config"]["sensitivity_spec"]
        # prompt_id = experiment["config"]["prompt_id"]
        # context_id = experiment["config"]["context_id"]
        # key = f"{hash(prompt_id)}_{hash(context_id)}_{hash(sensitivity_spec)}"
        key = reduce(
            lambda a, b: a + b,
            [f"{hash(experiment['config'][group_key])}" for group_key in grouping_keys],
        )
        groups[key].append(experiment)

    print(f"Grouped {len(groups)} groups")

    return groups


def plot_probability_distributions(
    group_data: List[Dict[str, Any]], grouping_keys: List[str]
) -> Figure:
    """Create visualization for a group of experiments."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 1.5 * len(group_data)))

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

        # def colormap(token: str) -> str:
        #     """Generate color based on token."""
        #     if token in ["1", "2", "3", "low"]:
        #         return "green"
        #     elif token in ["4", "5", "6", "mid"]:
        #         return "yellow"
        #     elif token in ["7", "8", "9", "high"]:
        #         return "red"

        for token, prob in zip(tokens, probs):
            ax.barh(
                y_pos,
                prob,
                left=left,
                height=0.8,
                # color=colormap(token),
                color=colorhash.ColorHash(token).hex,
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
                    fontsize=12,
                )

            left += prob

        bar_keys = [key for key in config if key not in grouping_keys]

        bar_label = reduce(
            lambda a, b: a + b,
            [f"{key}: {config[key]}\n" for key in bar_keys],
        )

        # bar_label = ""

        # bar_label = expression_desc
        ax.text(-0.05, y_pos, bar_label, ha="right", va="center", fontsize=10)

    # Set axis properties
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(group_data) - 0.5)
    ax.set_xlabel("Probability")
    ax.set_yticks([])
    ax.set_ylabel("")

    config = group_data[0]["config"]

    # wrapped_prompt = textwrap.fill(base_prompt, width=80)
    title_text = reduce(
        lambda a, b: a + b,
        [f"{key}: {str(config[key])[:40]}\n" for key in grouping_keys],
    )

    plt.title(title_text, fontsize=12, loc="left", pad=20)
    plt.tight_layout()
    return fig


def visualize_results(
    results_file: str, grouping_keys: List[str]
) -> List[Tuple[str, Figure]]:
    """Main function to create visualizations."""
    results = load_results(results_file)
    groups = group_experiments(results, grouping_keys)

    figures = []
    for group_name, group_data in groups.items():
        fig = plot_probability_distributions(group_data, grouping_keys)
        figures.append((group_name, fig))

    return figures


if __name__ == "__main__":
    # Example usage
    path = Path("multirun/2025-07-28/contextstrength/")
    example_results_file = path / "results.json"
    group_keys = [
        "model",
        "temperature",
        "expression_text",
        "expression_image",
        "prompt_id",
        # "context_id",
        "sensitivity_spec",
    ]
    example_figures = visualize_results(str(example_results_file), group_keys)

    # Save and show figures
    for idx, (example_group_name, example_fig) in enumerate(example_figures):
        example_fig.savefig(path / f"results_{idx+1}.png")
        # plt.show()
