"""
Q1 Disagreement Analysis: Speech vs Facial Expression Disagreement

This module analyzes how model uncertainty and classification accuracy change
when speech and facial expressions disagree in assistive robotics
feedback interpretation.

Research Question: How unconfident is the model when there is disagreement
between speech and facial expression?
"""

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Pylint tuning for this analysis script (safe: names reused across sections)
# - We intentionally reuse names across analysis stages.
# - We fix the expert_data name issue below; the rest are benign redefinitions.
# pylint: disable=redefined-outer-name

# Manual implementation of classification metrics


def calculate_entropy(probs: Dict[Any, float]) -> float:
    """Calculate entropy as a measure of uncertainty."""
    entropy = 0.0
    for prob in probs.values():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy


def calculate_label_probability_mass(
    predicted_probs: Dict[Any, float], true_labels: Dict[Any, float]
) -> float:
    """Calculate sum of min(predicted_prob, true_label_prob) for each class.

    This measures how much predicted probability 'fits' into the true
    label 'buckets', where each bucket has capacity equal to the true
    label probability.
    """
    total_mass = 0.0
    for class_label, true_prob in true_labels.items():
        predicted_prob = predicted_probs.get(class_label, 0.0)
        total_mass += min(predicted_prob, true_prob)
    return total_mass


warnings.filterwarnings("ignore")


def load_q1_data() -> (
    Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]
):
    """Load Q1 disagreement data from both expert labels and model predictions.

    Returns:
        Tuple of (expert_labels_dict, final_results_list)
    """
    print("🔍 Loading Q1 disagreement data...")

    curr_dir = Path(__file__).resolve().parent

    # Load expert labels from experiment config
    config_path = curr_dir / "configs/experiment_1_disagreement.json"
    if not config_path.exists():
        raise FileNotFoundError(f"ERROR: Q1 config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # Use the loaded config_data (expert_data was incorrect and unused here)
    print(f"✅ Loaded expert labels for {len(config_data['scenarios'])} scenarios")

    # Load model predictions from final results
    final_results_path = (
        curr_dir
        / "results"
        / "text_facial_expression_results"
        / Path("final_results.json")
    )
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

    print(
        f"✅ Loaded {len(final_results_list)} Q1 model prediction queries "
        f"from final results"
    )

    # Parse scenario classifications
    scenarios_by_type = {}
    for scenario in config_data["scenarios"]:
        scenario_name = scenario["name"]
        scenario_type = parse_scenario_classification(scenario_name)
        scenarios_by_type[scenario_name] = {
            "expert_labels": scenario["labels"],
            "classification": scenario_type,
            "received_feedback": scenario["received_feedback"],
        }

    print(f"✅ Parsed {len(scenarios_by_type)} scenario classifications")
    return config_data, final_results_list, scenarios_by_type


def is_q1_scenario(scenario_name: str) -> bool:
    """Check if scenario belongs to Q1 disagreement experiment."""
    if not scenario_name:
        return False

    # Remove trailing numbers and underscores
    clean_name = scenario_name.rstrip("_0123456789")

    # Check if it matches disagreement pattern: v[hmn]f[hmn]
    if len(clean_name) == 4 and clean_name.startswith("v") and clean_name[2] == "f":
        verbal_level = clean_name[1]
        facial_level = clean_name[3]
        return verbal_level in ["h", "m", "n"] and facial_level in ["h", "m", "n"]

    return False


def parse_scenario_classification(scenario_name: str) -> Dict[str, str]:
    """Parse scenario name to extract disagreement classification.

    Args:
        scenario_name: e.g., 'vhfh_1', 'vmfn_2'

    Returns:
        Classification dict with verbal_level, facial_level, agreement_type
    """
    clean_name = scenario_name.rstrip("_0123456789")

    if len(clean_name) == 4 and clean_name.startswith("v") and clean_name[2] == "f":
        verbal_code = clean_name[1]  # h, m, or n
        facial_code = clean_name[3]  # h, m, or n

        level_map = {"h": "high", "m": "mid", "n": "none"}
        verbal_level = level_map.get(verbal_code, verbal_code)
        facial_level = level_map.get(facial_code, facial_code)

        # Determine agreement type
        if verbal_code == facial_code:
            agreement_type = "agreement"
        else:
            agreement_type = "disagreement"

        return {
            "scenario_category": clean_name,
            "verbal_level": verbal_level,
            "facial_level": facial_level,
            "agreement_type": agreement_type,
            "scenario_description": f"verbal_{verbal_level}_facial_{facial_level}",
        }

    return {
        "scenario_category": clean_name,
        "verbal_level": "unknown",
        "facial_level": "unknown",
        "agreement_type": "unknown",
        "scenario_description": clean_name,
    }


def organize_predictions_by_scenario(
    final_results_list: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Organize model predictions by scenario and prediction type.

    Returns:
        Dict[scenario_name][query_type][body_part] = prediction_data
    """
    print("📊 Organizing model predictions by scenario...")

    organized: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for result in final_results_list:
        try:
            scenario_name = result["input_metadata"]["scenario_name"]
            query_type = result["input_metadata"]["query_type"]

            if scenario_name not in organized:
                organized[scenario_name] = {}

            if query_type not in organized[scenario_name]:
                organized[scenario_name][query_type] = {}

            # Store prediction data
            if query_type in ["body_part_full", "body_part_single"]:
                body_part = result.get("body_part")
                if body_part:
                    organized[scenario_name][query_type][body_part] = {
                        "cleaned_probabilities": result.get(
                            "cleaned_probabilities", {}
                        ),
                        "raw_output": result.get("full_output")
                        or result.get("single_token_logits", {}),
                    }

        except KeyError as e:
            print(f"🔸 Warning: Missing key {e} in result, skipping...")
            continue

    print(f"✅ Organized predictions for {len(organized)} scenarios")

    return organized


def calculate_brier_score(
    predicted_probs: Dict[str, float], true_labels: Dict[str, float]
) -> float:
    """Calculate Brier score between predicted probability distribution and
    true labels.

    Args:
        predicted_probs: Dict mapping categories to predicted probabilities
        true_labels: Dict mapping categories to true probability values

    Returns:
        Brier score (lower is better)
    """
    # Convert to numpy arrays aligned by keys
    all_keys = sorted(set(predicted_probs.keys()) | set(true_labels.keys()))

    pred_array = np.array([predicted_probs.get(k, 0.0) for k in all_keys])
    true_array = np.array([true_labels.get(k, 0.0) for k in all_keys])

    # Brier score = mean squared difference
    return np.mean((pred_array - true_array) ** 2)


def get_predicted_class(probs_dict: Dict[str, float]) -> str:
    """Get the class with highest probability."""
    return max(probs_dict.items(), key=lambda x: x[1])[0]


def get_true_class(labels_dict: Dict[str, float]) -> str:
    """Get the true class (highest probability in labels)."""
    return max(labels_dict.items(), key=lambda x: x[1])[0]


def calculate_classification_metrics(
    y_true: List[str], y_pred: List[str]
) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 score manually with macro
    averaging."""
    # Get unique classes
    classes = sorted(set(y_true + y_pred))

    # Calculate metrics for each class
    class_metrics = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        class_metrics[cls] = {"precision": precision, "recall": recall, "f1": f1}

    # Calculate macro averages
    if classes:
        macro_precision = sum(
            metrics["precision"] for metrics in class_metrics.values()
        ) / len(classes)
        macro_recall = sum(
            metrics["recall"] for metrics in class_metrics.values()
        ) / len(classes)
        macro_f1 = sum(metrics["f1"] for metrics in class_metrics.values()) / len(
            classes
        )
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    return macro_precision, macro_recall, macro_f1


def calculate_precision_for_true_class(y_true: List[str], y_pred: List[str]) -> float:
    """Calculate precision for the true class in a single-class experiment."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return np.nan

    # Get the true class (should be the same for all y_true in single-class experiments)
    true_class = y_true[0]  # All should be the same

    # Count predictions of the true class
    predictions_of_true_class = sum(1 for p in y_pred if p == true_class)

    if predictions_of_true_class == 0:
        return 0.0  # No predictions of true class = precision 0

    # Count correct predictions of the true class
    correct_predictions_of_true_class = sum(
        1 for t, p in zip(y_true, y_pred) if t == true_class and p == true_class
    )

    return correct_predictions_of_true_class / predictions_of_true_class


def calculate_mae(y_true: List[str], y_pred: List[str]) -> float:
    """Calculate Mean Absolute Error between predicted and true integer
    classes."""
    if len(y_true) != len(y_pred):
        return np.nan

    # Convert classes to integers for distance calculation
    try:
        # Map class labels to integers for ordinal calculation
        class_map = {"comfortable": 0, "mildly_uncomfortable": 1, "uncomfortable": 2}
        true_ints = [
            class_map.get(cls, int(cls) if cls.isdigit() else 0) for cls in y_true
        ]
        pred_ints = [
            class_map.get(cls, int(cls) if cls.isdigit() else 0) for cls in y_pred
        ]

        # Calculate absolute differences and return mean
        absolute_errors = [abs(t - p) for t, p in zip(true_ints, pred_ints)]
        return float(np.mean(absolute_errors))
    except (ValueError, TypeError):
        return np.nan


def calculate_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Calculate simple accuracy (proportion of correct predictions)."""
    if len(y_true) != len(y_pred):
        return np.nan

    if len(y_true) == 0:
        return np.nan

    # Count correct predictions
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def calculate_cosine_similarity(
    predicted_probs: Dict[str, float], true_labels: Dict[str, float]
) -> float:
    """Calculate cosine similarity between predicted probability distribution
    and true labels.

    Args:
        predicted_probs: Dict mapping categories to predicted probabilities
        true_labels: Dict mapping categories to true probability values

    Returns:
        Cosine similarity (higher is better, range [-1, 1])
    """
    # Convert to numpy arrays aligned by keys
    all_keys = sorted(set(predicted_probs.keys()) | set(true_labels.keys()))

    pred_array = np.array([predicted_probs.get(k, 0.0) for k in all_keys])
    true_array = np.array([true_labels.get(k, 0.0) for k in all_keys])

    # Calculate cosine similarity
    pred_norm = np.linalg.norm(pred_array)
    true_norm = np.linalg.norm(true_array)

    if pred_norm == 0 or true_norm == 0:
        return 0.0  # Handle zero vectors

    dot_product = np.dot(pred_array, true_array)
    cosine_sim = dot_product / (pred_norm * true_norm)

    return cosine_sim


def analyze_brier_scores_by_experiment_type(
    organized_predictions: Dict[str, Dict[str, Dict[str, Any]]],
    scenarios_by_type: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Calculate Brier scores and F1 scores for each prediction and group by
    experiment type (v<x>f<x>).

    Returns:
        Dict mapping experiment type to aggregated statistics
    """
    print("📊 Calculating Brier scores and F1 scores by experiment type...")

    results_by_type = {}

    for scenario_name, predictions in organized_predictions.items():
        if scenario_name not in scenarios_by_type:
            print(f"🔸 Warning: Scenario {scenario_name} not found in expert data")
            continue

        scenario_info = scenarios_by_type[scenario_name]
        expert_labels = scenario_info["expert_labels"]
        classification = scenario_info["classification"]
        experiment_type = classification["scenario_description"]

        if experiment_type not in results_by_type:
            results_by_type[experiment_type] = {
                "brier_scores_full": [],
                "brier_scores_single": [],
                "predicted_classes_full": [],
                "predicted_classes_single": [],
                "entropy_full": [],
                "entropy_single": [],
                "entropy_labels": [],
                "label_prob_mass_full": [],
                "label_prob_mass_single": [],
                "mae_full": [],
                "mae_single": [],
                "cosine_similarity_full": [],
                "cosine_similarity_single": [],
                "true_classes": [],
                "scenario_names": [],
                "agreement_type": classification["agreement_type"],
                "verbal_level": classification["verbal_level"],
                "facial_level": classification["facial_level"],
            }

        results_by_type[experiment_type]["scenario_names"].append(scenario_name)

        # Calculate Brier scores and classifications for wrist only
        for query_type in ["body_part_full", "body_part_single"]:
            if query_type in predictions:
                # Focus only on wrist predictions
                if "wrist" in predictions[query_type] and "wrist" in expert_labels:
                    prediction_data = predictions[query_type]["wrist"]
                    predicted_probs = prediction_data["cleaned_probabilities"]
                    true_labels = expert_labels["wrist"]

                    # Convert string keys to match
                    predicted_probs_str = {
                        str(k): float(v) for k, v in predicted_probs.items()
                    }
                    true_labels_str = {str(k): float(v) for k, v in true_labels.items()}

                    # Calculate Brier score
                    brier_score = calculate_brier_score(
                        predicted_probs_str, true_labels_str
                    )

                    # Calculate entropy
                    entropy = calculate_entropy(predicted_probs_str)

                    # Calculate label probability mass
                    label_prob_mass = calculate_label_probability_mass(
                        predicted_probs_str, true_labels_str
                    )

                    # Calculate cosine similarity
                    cosine_sim = calculate_cosine_similarity(
                        predicted_probs_str, true_labels_str
                    )

                    # Calculate label entropy once per scenario (on full path)
                    if query_type == "body_part_full":
                        label_entropy = calculate_entropy(true_labels_str)
                        results_by_type[experiment_type]["entropy_labels"].append(
                            label_entropy
                        )

                    # Get predicted and true classes
                    predicted_class = get_predicted_class(predicted_probs_str)
                    true_class = get_true_class(true_labels_str)

                    # Calculate MAE for this scenario
                    mae = calculate_mae([true_class], [predicted_class])

                    if query_type == "body_part_full":
                        results_by_type[experiment_type]["brier_scores_full"].append(
                            brier_score
                        )
                        results_by_type[experiment_type][
                            "predicted_classes_full"
                        ].append(predicted_class)
                        results_by_type[experiment_type]["entropy_full"].append(entropy)
                        results_by_type[experiment_type]["label_prob_mass_full"].append(
                            label_prob_mass
                        )
                        results_by_type[experiment_type]["mae_full"].append(mae)
                        results_by_type[experiment_type][
                            "cosine_similarity_full"
                        ].append(cosine_sim)
                    else:
                        results_by_type[experiment_type]["brier_scores_single"].append(
                            brier_score
                        )
                        results_by_type[experiment_type][
                            "predicted_classes_single"
                        ].append(predicted_class)
                        results_by_type[experiment_type]["entropy_single"].append(
                            entropy
                        )
                        results_by_type[experiment_type][
                            "label_prob_mass_single"
                        ].append(label_prob_mass)
                        results_by_type[experiment_type]["mae_single"].append(mae)
                        results_by_type[experiment_type][
                            "cosine_similarity_single"
                        ].append(cosine_sim)

                    if query_type == "body_part_full":
                        results_by_type[experiment_type]["true_classes"].append(
                            true_class
                        )

    # Calculate summary statistics
    summary_stats = {}
    for exp_type, data in results_by_type.items():
        summary_stats[exp_type] = {
            "agreement_type": data["agreement_type"],
            "verbal_level": data["verbal_level"],
            "facial_level": data["facial_level"],
            "n_scenarios": len(data["scenario_names"]),
            "scenario_names": data["scenario_names"],
        }

        for score_type in [
            "brier_scores_full",
            "brier_scores_single",
            "entropy_full",
            "entropy_single",
            "label_prob_mass_full",
            "label_prob_mass_single",
            "mae_full",
            "mae_single",
            "cosine_similarity_full",
            "cosine_similarity_single",
        ]:
            scores = data[score_type]
            if scores:
                summary_stats[exp_type][f"{score_type}_mean"] = np.mean(scores)

                # Check for very low standard deviation (debugging)
                if score_type == "brier_scores_single" and len(scores) > 1:
                    std_val = np.std(scores, ddof=1)
                    if std_val < 0.001:  # Very small or zero std
                        unique_scores = len({round(s, 8) for s in scores})
                        print(
                            f"⚠️  {exp_type}: Single token std={std_val:.6f}, "
                            f"{unique_scores} unique values (n={len(scores)})"
                        )

                summary_stats[exp_type][f"{score_type}_std"] = (
                    np.std(scores, ddof=1) if len(scores) > 1 else 0.0
                )
                summary_stats[exp_type][f"{score_type}_n"] = len(scores)
            else:
                summary_stats[exp_type][f"{score_type}_mean"] = np.nan
                summary_stats[exp_type][f"{score_type}_std"] = np.nan
                summary_stats[exp_type][f"{score_type}_n"] = 0

        # Add label entropy mean (no std requested)
        if data.get("entropy_labels"):
            summary_stats[exp_type]["entropy_labels_mean"] = np.mean(
                data["entropy_labels"]
            )
        else:
            summary_stats[exp_type]["entropy_labels_mean"] = np.nan

        # Calculate F1, precision, and recall scores
        for pred_type in ["full", "single"]:
            predicted_classes = data[f"predicted_classes_{pred_type}"]
            true_classes = data["true_classes"]

            if (
                predicted_classes
                and true_classes
                and len(predicted_classes) == len(true_classes)
            ):
                try:
                    # Debug: Print class distributions for several experiment types
                    if pred_type == "full" and exp_type in [
                        "verbal_high_facial_high",
                        "verbal_none_facial_none",
                        "verbal_mid_facial_high",
                    ]:
                        true_counts = Counter(true_classes)
                        pred_counts = Counter(predicted_classes)
                        print(f"🔍 DEBUG {exp_type}:")
                        print(f"    True class counts: {dict(true_counts)}")
                        print(f"    Pred class counts: {dict(pred_counts)}")

                    # Calculate metrics
                    _, macro_recall, macro_f1 = calculate_classification_metrics(
                        true_classes, predicted_classes
                    )
                    true_class_precision = calculate_precision_for_true_class(
                        true_classes, predicted_classes
                    )
                    mae = calculate_mae(true_classes, predicted_classes)
                    accuracy = calculate_accuracy(true_classes, predicted_classes)

                    summary_stats[exp_type][f"f1_{pred_type}"] = macro_f1
                    summary_stats[exp_type][
                        f"precision_{pred_type}"
                    ] = true_class_precision
                    summary_stats[exp_type][f"recall_{pred_type}"] = macro_recall
                    summary_stats[exp_type][f"mae_{pred_type}"] = mae
                    summary_stats[exp_type][f"accuracy_{pred_type}"] = accuracy
                except Exception as e:
                    print(
                        f"🔸 Warning: Could not calculate metrics for "
                        f"{exp_type} {pred_type}: {e}"
                    )
                    summary_stats[exp_type][f"f1_{pred_type}"] = np.nan
                    summary_stats[exp_type][f"precision_{pred_type}"] = np.nan
                    summary_stats[exp_type][f"recall_{pred_type}"] = np.nan
                    summary_stats[exp_type][f"mae_{pred_type}"] = np.nan
                    summary_stats[exp_type][f"accuracy_{pred_type}"] = np.nan
            else:
                print(
                    f"🔸 Data mismatch for {exp_type} {pred_type}: "
                    f"pred_len={len(predicted_classes) if predicted_classes else 0}, "
                    f"true_len={len(true_classes) if true_classes else 0}"
                )
                summary_stats[exp_type][f"f1_{pred_type}"] = np.nan
                summary_stats[exp_type][f"precision_{pred_type}"] = np.nan
                summary_stats[exp_type][f"recall_{pred_type}"] = np.nan
                summary_stats[exp_type][f"mae_{pred_type}"] = np.nan
                summary_stats[exp_type][f"accuracy_{pred_type}"] = np.nan

    return summary_stats


def print_brier_score_summary(summary_stats: Dict[str, Dict[str, Any]]) -> None:
    """Print formatted summary of Brier score and classification metrics
    analysis."""
    print("\n" + "=" * 100)
    print("PERFORMANCE ANALYSIS BY EXPERIMENT TYPE (WRIST ONLY)")
    print("=" * 100)

    # Group by agreement type for better organization
    agreement_groups: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}
    for exp_type, stats in summary_stats.items():
        agreement_type = stats["agreement_type"]
        if agreement_type not in agreement_groups:
            agreement_groups[agreement_type] = []
        agreement_groups[agreement_type].append((exp_type, stats))

    for agreement_type in ["agreement", "disagreement"]:
        if agreement_type in agreement_groups:
            print(f"\n{agreement_type.upper()} SCENARIOS:")
            print("-" * 40)

            for exp_type, stats in sorted(agreement_groups[agreement_type]):
                print(f"\n{exp_type} (n={stats['n_scenarios']} scenarios):")
                print("  " + "-" * 70)

                # Full output scores
                print("  Full Output:")
                if not np.isnan(stats["brier_scores_full_mean"]):
                    print(
                        f"    Brier Score: {stats['brier_scores_full_mean']:.4f} ± "
                        f"{stats['brier_scores_full_std']:.4f}"
                    )
                    if not np.isnan(stats.get("f1_full", np.nan)):
                        print(f"    F1 Score:    {stats['f1_full']:.4f}")
                        print(f"    Precision:   {stats['precision_full']:.4f}")
                        print(f"    Recall:      {stats['recall_full']:.4f}")
                        print(f"    Accuracy:    {stats['accuracy_full']:.4f}")
                        print(f"    MAE:         {stats['mae_full']:.4f}")
                else:
                    print("    No data")

                # Single token scores
                print("  Single Token:")
                if not np.isnan(stats["brier_scores_single_mean"]):
                    print(
                        f"    Brier Score: {stats['brier_scores_single_mean']:.4f} ± "
                        f"{stats['brier_scores_single_std']:.4f}"
                    )
                    if not np.isnan(stats.get("f1_single", np.nan)):
                        print(f"    F1 Score:    {stats['f1_single']:.4f}")
                        print(f"    Precision:   {stats['precision_single']:.4f}")
                        print(f"    Recall:      {stats['recall_single']:.4f}")
                        print(f"    Accuracy:    {stats['accuracy_single']:.4f}")
                        print(f"    MAE:         {stats['mae_single']:.4f}")
                else:
                    print("    No data")


if __name__ == "__main__":
    print("🚀 Starting Q1 Disagreement Analysis")
    print("=" * 50)

    try:
        # Load data
        expert_data, final_results_list, scenarios_by_type = load_q1_data()

        # Organize predictions
        organized_predictions = organize_predictions_by_scenario(final_results_list)

        # Calculate Brier scores by experiment type
        brier_summary = analyze_brier_scores_by_experiment_type(
            organized_predictions, scenarios_by_type
        )

        # Print results
        print_brier_score_summary(brier_summary)

        print("\n✅ Analysis completed successfully!")

        # print(scenarios_by_type)

    except Exception as e:
        print(f"🚨 ERROR: Failed to complete analysis: {e}")
        raise
