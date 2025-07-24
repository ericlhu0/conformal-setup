"Visualize results for modality disagreement."

import matplotlib.pyplot as plt

from safe_feedback_interpretation.viz_utils import visualize_results

if __name__ == "__main__":
    # Example usage
    results_file = "multirun/2025-07-24/12-19-38/results.json"
    figures = visualize_results(results_file)

    # Save and show figures
    for i, (group_name, fig) in enumerate(figures):
        # fig.savefig(f"results_{i+1}.png")
        plt.show()
