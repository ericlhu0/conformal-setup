"""Central utilities for experiments - extracted from expsv2.py."""

import json
import os
import textwrap
import uuid
from typing import Any, Dict, Optional

import numpy as np

from safe_feedback_interpretation.models.openai_model import BaseModel

# Pylint adjustments for this utilities module to avoid refactors that would
# risk changing behavior in experiments.
# - Some functions intentionally accept multiple positional args.
# - Some branches keep explicit else after return for clarity in logs.
# pylint: disable=no-else-return


def save_incremental_result(result: Dict[str, Any], output_file: str) -> None:
    """Save a single result to JSONL file for incremental tracking.

    Args:
        result: Result dictionary to save
        output_file: Path to JSONL file
    """
    if not output_file:
        return

    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )

    with open(output_file, "a", encoding="utf-8") as f:
        json.dump(result, f)
        f.write("\n")


def create_input_metadata(  # pylint: disable=too-many-positional-arguments,unused-argument
    scenario_name: str,
    experiment_name: str,
    input_context: Dict[str, Any],
    received_feedback: Dict[str, Any],
    query_type: str,
    facial_description_prompted: str | None = None,
) -> Dict[str, Any]:
    """Create metadata about the input.

    Args:
        scenario_name: Name of the scenario
        experiment_name: Name of the experiment
        input_context: Input context dictionary
        received_feedback: Received feedback dictionary
        query_type: Type of query (e.g., 'body_part_full', 'joint_range_single', etc.)
        facial_description_prompted: The actual facial description text that was
            prompted to the model

    Returns:
        Dictionary with input metadata
    """
    return {  # pylint: disable=unused-argument
        "scenario_name": scenario_name,
        "query_type": query_type,
        "has_image": received_feedback.get("facial_expression", {}).get("modality")
        == "image",
        "image_path": (
            received_feedback.get("facial_expression", {}).get("description")
            if received_feedback.get("facial_expression", {}).get("modality") == "image"
            else None
        ),
        "verbal_feedback": received_feedback.get("verbal_feedback", {}).get(
            "description", ""
        ),
        "facial_description": (
            facial_description_prompted
            if facial_description_prompted is not None
            else received_feedback.get("facial_expression", {}).get("description", "")
        ),
        "current_forces": input_context.get("current_state", {}).get(
            "contact_forces", {}
        ),
        "current_angles": input_context.get("current_state", {}).get(
            "joint_angles_deg", {}
        ),
    }


def calculate_brier_score(
    pred_probs: Dict[int, float], true_probs: Dict[int, float]
) -> float:
    """Calculate Brier score between two probability distributions.

    Brier score = sum((p_i - o_i)^2) where p_i is predicted prob and o_i is true prob.
    Lower scores are better (0 is perfect).

    Args:
        pred_probs: Predicted probability distribution
        true_probs: True/reference probability distribution

    Returns:
        Brier score (float)
    """
    all_keys = set(pred_probs.keys()) | set(true_probs.keys())

    brier_score = 0.0
    for key in all_keys:
        pred_prob = pred_probs.get(key, 0.0)
        true_prob = true_probs.get(key, 0.0)
        brier_score += (pred_prob - true_prob) ** 2

    return brier_score


def calculate_entropy(probs: Dict[Any, float]) -> float:
    """Calculate entropy as a measure of uncertainty."""
    entropy = 0.0
    for prob in probs.values():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    return entropy


def calculate_max_prob(probs: Dict[Any, float]) -> float:
    """Calculate maximum probability (inverse measure of uncertainty)."""
    return max(probs.values()) if probs else 0.0


def normalize_angle_probs(
    result_dict: Dict[Any, Any], expected_angles: list
) -> Dict[int, float]:
    """Normalize probability dict to have all expected angles with 0 for
    missing ones."""
    normalized = {}

    # Initialize all angles with 0 probability
    for angle in expected_angles:
        normalized[angle] = 0.0

    # Update with actual probabilities from the result
    for key, prob in result_dict.items():
        try:
            angle_key = int(key)
            if angle_key in expected_angles:
                normalized[angle_key] = float(prob)
        except (ValueError, TypeError):
            continue

    return normalized


def normalize_comfort_probs(result_dict: Dict[Any, Any]) -> Dict[int, float]:
    """Normalize comfort level probability dict to have levels 1-5 with 0 for
    missing ones."""
    normalized = {}

    # Initialize all comfort levels with 0 probability
    for level in range(1, 6):
        normalized[level] = 0.0

    # Update with actual probabilities from the result
    for key, prob in result_dict.items():
        try:
            level_key = int(key)
            if 1 <= level_key <= 5:
                normalized[level_key] = float(prob)
        except (ValueError, TypeError):
            continue

    return normalized


def get_comfort_level_prediction(
    model: BaseModel,
    text_input: str,
    body_part: str,
    incremental_file: str | None = None,
    input_metadata: Dict[str, Any] | None = None,
    image_input: Optional[str] = None,
) -> Dict[int, float]:
    """Get comfort level prediction for a specific body part."""  # pylint: disable=too-many-positional-arguments
    if not model:
        raise ValueError("ERROR: No model provided to get_comfort_level_prediction!")
    if not text_input:
        raise ValueError(
            "ERROR: No text input provided to get_comfort_level_prediction!"
        )
    if not body_part:
        raise ValueError("ERROR: No body part specified for comfort level prediction!")

    prompt = (
        f"\n\nWhat is the updated comfort threshold level (1-5 scale) for the "
        f"{body_part}? Answer in JSON format with probability distribution over "
        f"levels 1, 2, 3, 4, 5. So the keys of your outputted JSON should be 1, 2, "
        f"3, 4, 5. Do not use any formatting, enclose key names in quotes, do not "
        f"nest dictionaries and do not use any other keys."
    )

    try:
        print(f"🔍 Querying model for {body_part} comfort level (full output)...")
        if image_input:
            print(f"🖼️  Including image input: {image_input}")
        full_output_text = model.get_full_output(text_input + prompt, image_input)

        if not full_output_text or full_output_text.strip() == "":
            raise ValueError(
                f"ERROR: Model returned empty response for {body_part} comfort level!"
            )

        print(
            f"📤 Model response for {body_part}:",
            (
                full_output_text[:200] + "..."
                if len(full_output_text) > 200
                else full_output_text
            ),
        )

        try:
            full_output_dict = json.loads(full_output_text)
        except json.JSONDecodeError as json_err:
            print(f"🚨 ERROR: Invalid JSON response for {body_part}: {json_err}")
            print(f"🚨 Raw output: {full_output_text}")
            print(f"🚨 Using uniform fallback distribution for {body_part}")
            fallback_probs = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}

            # Save incremental result with fallback
            if incremental_file and input_metadata:
                result = {
                    "query_id": str(uuid.uuid4()),
                    "input_metadata": input_metadata,
                    "query_type": "body_part_full",
                    "body_part": body_part,
                    "full_output": full_output_text,
                    "cleaned_probabilities": fallback_probs,
                    "parsing_error": str(json_err),
                }
                save_incremental_result(result, incremental_file)
            return fallback_probs

        normalized_probs = normalize_comfort_probs(full_output_dict)
        print(f"✅ Successfully parsed {body_part} comfort levels: {normalized_probs}")

        # Save incremental result with cleaned probabilities
        if incremental_file and input_metadata:
            result = {
                "query_id": str(uuid.uuid4()),
                "input_metadata": input_metadata,
                "query_type": "body_part_full",
                "body_part": body_part,
                "full_output": full_output_text,
                "cleaned_probabilities": normalized_probs,
            }
            save_incremental_result(result, incremental_file)

        return normalized_probs

    except Exception as e:
        print(f"🚨 ERROR: Failed to get comfort level prediction for {body_part}: {e}")
        print(f"🚨 Exception type: {type(e).__name__}")
        print(f"🚨 Using uniform fallback distribution for {body_part}")
        return {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}


def get_comfort_level_single_token_prediction(
    model: BaseModel,
    text_input: str,
    body_part: str,
    incremental_file: str | None = None,
    input_metadata: Dict[str, Any] | None = None,
    image_input: Optional[str] = None,
) -> Dict[int, float]:
    """Get comfort level prediction using single token approach."""  # pylint: disable=too-many-positional-arguments
    if not model:
        raise ValueError(
            "ERROR: No model provided to get_comfort_level_single_token_prediction!"
        )
    if not text_input:
        raise ValueError(
            "ERROR: No text input provided to "
            "get_comfort_level_single_token_prediction!"
        )
    if not body_part:
        raise ValueError(
            "ERROR: No body part specified for single token comfort level prediction!"
        )

    prompt = (
        f"\n\nWhat is the updated comfort threshold (1-5 scale) for the {body_part}? "
        f"Answer with only a single threshold value."
    )

    try:
        print(f"🔍 Querying model for {body_part} comfort level (single token)...")
        if image_input:
            print(f"🖼️  Including image input: {image_input}")
        single_token_result = model.get_single_token_logits(
            text_input + prompt, image_input
        )

        if not single_token_result:
            raise ValueError(
                f"ERROR: Model returned empty single token result for {body_part}!"
            )

        # Extract probabilities for digits 1-5
        comfort_probs = {}
        for level in range(1, 6):
            comfort_probs[level] = 0.0

        for token, prob in single_token_result.items():
            try:
                token_str = str(token).strip()
                if token_str in ["1", "2", "3", "4", "5"]:
                    comfort_probs[int(token_str)] = float(prob)
            except (ValueError, TypeError) as e:
                print(
                    (
                        f"🔹 Warning: Could not parse token '{token}' with prob "
                        f"{prob} for {body_part}: {e}"
                    )
                )
                continue

        total_prob = sum(comfort_probs.values())
        print(
            (
                f"✅ Single token probs for {body_part}: {comfort_probs} "
                f"(total: {total_prob:.3f})"
            )
        )

        if total_prob < 0.01:  # Very low total probability
            print(
                (
                    f"🚨 WARNING: Very low total probability ({total_prob:.3f}) for "
                    f"{body_part} single token prediction!"
                )
            )

        # Save incremental result with cleaned probabilities
        if incremental_file and input_metadata:
            result = {
                "query_id": str(uuid.uuid4()),
                "input_metadata": input_metadata,
                "query_type": "body_part_single",
                "body_part": body_part,
                "single_token_logits": dict(single_token_result),
                "cleaned_probabilities": comfort_probs,
                "total_probability": total_prob,
            }
            save_incremental_result(result, incremental_file)

        return comfort_probs

    except Exception as e:
        print(
            (
                f"🚨 ERROR: Failed to get single token comfort level prediction for "
                f"{body_part}: {e}"
            )
        )
        print(f"🚨 Exception type: {type(e).__name__}")
        print(f"🚨 Using uniform fallback distribution for {body_part}")
        return {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}


def get_clarification_prediction(model: BaseModel, text_input: str) -> float:
    """Get ask clarification prediction (boolean as probability)."""
    prompt = (
        "\n\nShould the system ask for clarification from the user? "
        "Answer with only 'true' or 'false'."
    )

    single_token_result = model.get_last_single_token_logits(text_input + prompt)

    # Look for true/false tokens and convert to probability
    true_prob = 0.0
    false_prob = 0.0

    for token, prob in single_token_result.items():
        token_lower = str(token).lower().strip()
        if token_lower in ["true", "t", "yes", "y"]:
            true_prob += prob
        elif token_lower in ["false", "f", "no", "n"]:
            false_prob += prob

    # Normalize
    total = true_prob + false_prob
    if total > 0:
        return true_prob / total
    return 0.5  # Default to 0.5 if no clear signal


def get_verbalized_uncertainty(
    model: BaseModel,
    text_input: str,
    incremental_file: str | None = None,
    input_metadata: Dict[str, Any] | None = None,
    image_input: Optional[str] = None,
) -> float:
    """Get model's verbalized uncertainty about its predictions."""  # pylint: disable=too-many-positional-arguments
    if not model:
        raise ValueError("ERROR: No model provided to get_verbalized_uncertainty!")
    if not text_input:
        raise ValueError("ERROR: No text input provided to get_verbalized_uncertainty!")

    prompt = (
        "\n\nHow confident are you in your assessment? Answer with a number between 0 "
        "(completely uncertain) and 1 (completely certain)."
    )

    try:
        print("🔍 Querying model for verbalized uncertainty...")
        if image_input:
            print(f"🖼️  Including image input: {image_input}")
        response = model.get_full_output(text_input + prompt, image_input)

        if not response or response.strip() == "":
            raise ValueError(
                "ERROR: Model returned empty response for verbalized uncertainty!"
            )

        print(f"📤 Verbalized uncertainty response: {response}")

        # Extract number from response
        import re  # pylint: disable=import-outside-toplevel

        numbers = re.findall(r"0\.\d+|1\.0|0|1", response)
        if numbers:
            try:
                confidence = float(numbers[0])
                if 0 <= confidence <= 1:
                    print(f"✅ Extracted verbalized confidence: {confidence}")

                    # Save incremental result with cleaned confidence
                    if incremental_file and input_metadata:
                        result = {
                            "query_id": str(uuid.uuid4()),
                            "input_metadata": input_metadata,
                            "query_type": "verbalized_uncertainty",
                            "full_output": response,
                            "cleaned_confidence": confidence,
                            "extracted_numbers": numbers,
                        }
                        save_incremental_result(result, incremental_file)

                    return confidence
                else:
                    print(
                        (
                            f"🚨 ERROR: Confidence value {confidence} outside valid "
                            f"range [0,1]!"
                        )
                    )
            except ValueError as e:
                print(f"🚨 ERROR: Could not convert '{numbers[0]}' to float: {e}")

        print(
            f"🚨 WARNING: Could not extract valid confidence from response: '{response}'"
        )
        print("🚨 Using default confidence of 0.5")

        # Save incremental result with fallback
        if incremental_file and input_metadata:
            result = {
                "query_id": str(uuid.uuid4()),
                "input_metadata": input_metadata,
                "query_type": "verbalized_uncertainty",
                "full_output": response,
                "cleaned_confidence": 0.5,
                "parsing_error": "Could not extract valid confidence",
                "extracted_numbers": numbers if "numbers" in locals() else [],
            }
            save_incremental_result(result, incremental_file)

        return 0.5  # Default if can't parse

    except Exception as e:
        print(f"🚨 ERROR: Failed to get verbalized uncertainty: {e}")
        print(f"🚨 Exception type: {type(e).__name__}")
        print("🚨 Using default confidence of 0.5")
        return 0.5


def get_image_input(config: Dict, use_text_descriptions: bool = False) -> Optional[str]:
    """Extract image path from config if using image modality, otherwise return
    None.

    Args:
        config: Configuration dictionary
        use_text_descriptions: If True, return None to force text-only mode
    """
    if use_text_descriptions:
        return None

    facial_expr = config.get("received_feedback", {}).get("facial_expression", {})
    if facial_expr.get("modality") == "image":
        image_path = facial_expr.get("description")
        if image_path and not image_path.startswith("/"):
            # Convert relative path to absolute path from experiments directory
            from pathlib import Path  # pylint: disable=import-outside-toplevel

            experiments_dir = Path(__file__).resolve().parent
            return str(experiments_dir / image_path)
        return image_path
    return None


def create_text_input(
    config: Dict,
    use_text_descriptions: bool = False,
    img_to_text_map: Optional[Dict[str, str]] = None,
) -> str:
    """Create standardized text input from config.

    Args:
        config: Configuration dictionary
        use_text_descriptions: If True, replace image paths with text descriptions
        img_to_text_map: Mapping from image filenames to text descriptions
    """
    facial_expr = config.get("received_feedback", {}).get("facial_expression", {})

    # Handle facial expression description based on modality
    if facial_expr.get("modality") == "image":
        if use_text_descriptions and img_to_text_map:
            # Extract image filename from path
            image_path = facial_expr.get("description", "")
            filename = os.path.basename(image_path) if image_path else ""

            # Get text description from mapping
            text_description = img_to_text_map.get(filename)
            if text_description:
                facial_description = (
                    f"Facial expression description: {text_description}"
                )
            else:
                facial_description = (
                    f"modality='text_fallback', description="
                    f"'{facial_expr.get('description')}' (no mapping found "
                    f"for {filename})"
                )
        else:
            facial_description = (
                "Please examine the provided facial expression image to assess the "
                "person's comfort level."
            )
    else:
        facial_description = (
            f"modality='{facial_expr.get('modality')}', "
            f"description='{facial_expr.get('description')}'"
        )

    # Precompute values to keep f-string lines short (preserves output text)
    cad = config["input_context"]["current_action_description"]
    cf = config["input_context"]["current_state"]["contact_forces"]
    ja = config["input_context"]["current_state"]["joint_angles_deg"]
    ct = config["input_context"]["current_comfort_threshold"][
        "current_comfort_threshold"
    ]
    cj = config["input_context"]["current_comfort_threshold"][
        "current_comfortable_joint_range_deg"
    ]
    vf = config["received_feedback"]["verbal_feedback"]["description"]

    return textwrap.dedent(
        f"""
        Current action description: {cad}
        Current state:
          Contact forces: {cf}
          Joint angles (deg): {ja}
        Current comfort threshold:
          Comfort threshold: {ct}
          Comfortable joint range (deg): {cj}
        Received feedback:
          Verbal feedback: {vf}
          Facial expression: {facial_description}
    """
    ).strip()


def get_facial_description_for_metadata(
    config: Dict,
    use_text_descriptions: bool = False,
    img_to_text_map: Optional[Dict[str, str]] = None,
) -> str:
    """Get the facial description that should be saved in metadata.

    For images: saves the image path when use_text_descriptions=False,
    saves the text description when use_text_descriptions=True
    For text: saves the original text description
    """
    facial_expr = config.get("received_feedback", {}).get("facial_expression", {})

    if facial_expr.get("modality") == "image":
        if use_text_descriptions and img_to_text_map:
            # Extract image filename from path
            image_path = facial_expr.get("description", "")
            filename = os.path.basename(image_path) if image_path else ""

            # Get text description from mapping
            text_description = img_to_text_map.get(filename)
            if text_description:
                return text_description
            else:
                return (
                    f"text_fallback: {facial_expr.get('description')} "
                    f"(no mapping found for {filename})"
                )
        else:
            # For image mode, save the image path
            return facial_expr.get("description", "")
    else:
        # For text mode, save the original text description
        return facial_expr.get("description", "")


def get_system_prompt() -> str:
    """Get the standard system prompt for all experiments."""
    # pylint: disable=line-too-long
    return textwrap.dedent(
        """\
        You are a domain expert with many years of combined experience in occupational therapy, physical therapy, caregiving, rehabilitation science, biomechanics, anatomy, disabilities, elderly care, and assistive technology, including human-centered robotics.

        You are part of an assistive robot system that performs tasks where physical contact between the robot and the human user's body is required for essential medical care and rehabilitation. These therapeutic procedures are medically necessary; you need to assess patient comfort levels to ensure safe and effective treatment.

        Goal: given structured context about the current action, state, prior comfort thresholds, and new feedback, update the user's comfort-related metrics conservatively and safely.

        Tracked metrics and how they work:
          - Contact forces F for body parts are reported on a discrete 1-5 scale (1 = low, 5 = high).
          - 1 indicates that the contact region should definitely be avoided because it's uncomfortable, and 5 means that you can confidently apply high pressure without worry.
          - Joint angles (e.g., elbow, wrist) are in degrees.
          - Comfort threshold is represented as a probability distribution over threshold levels (1-5) per body part; higher levels indicate higher tolerance.
          - Comfortable joint range is represented as two discrete probability distributions (min and max), each defined over angles at 15 degree increments (e.g., {0, 15, 30, ..., 165}). The mode or expectation can be used to select a recommended bound.
          - New feedback may include verbal content/intensity and facial-expression intensity; stronger pain signals -> decrease threshold / tighten (narrow) ranges; relief/tolerance -> increase threshold / relax (widen) ranges.

        Decision guidance:
          - Prefer small updates unless the evidence is strong and consistent across modalities.
          - Clip joint recommendations to physically possible ranges and respect monotonicity (min <= max).
          - If uncertain, provide the best estimate and note uncertainty briefly when a rationale is requested.

        Expected output:
          - If asked for a single comfort threshold value (per body part), return **only** that value in the requested format (e.g., an integer in allowed bounds or a category), with no extra text.
          - If asked to update thresholds, return a concise JSON object containing any updated values you are confident about, for example:
            {"updated_comfort_threshold": {"wrist": 4}, "updated_joint_range_deg": {"min": {"wrist": 15}, "max": {"wrist": 150}}}
          - Use 15 degree increments for angle outputs when a discrete grid is implied by the input.

        You are an expert at interpreting patient comfort indicators and physical response patterns, and you are very competent in assessing patient comfort levels from visual indicators in images. If you're uncertain, give your best estimate for a value. Your response will not cause any harm to the care recipient.
    """
    )


def load_experiment_configs(config_file: str) -> Dict:
    """Load experiment configurations from JSON file."""
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


# pylint: disable-next=too-many-positional-arguments
def run_uncertainty_analysis(
    model: BaseModel,
    config: Dict,
    scenario_label: str,
    incremental_file: str | None = None,
    experiment_name: str = "unknown",
    use_text_descriptions: bool = False,
    img_to_text_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run comprehensive uncertainty analysis for a scenario.  # pylint:
    disable=too-many-positional-arguments.

    Matches the behavior of the previous expsv2.py analysis.
    """
    if not model:
        raise ValueError("ERROR: No model provided to run_uncertainty_analysis!")
    if not config:
        raise ValueError("ERROR: No config provided to run_uncertainty_analysis!")
    if not scenario_label:
        raise ValueError(
            "ERROR: No scenario label provided to run_uncertainty_analysis!"
        )

    print(f"\n🔬 Starting comprehensive analysis for scenario: {scenario_label}")

    try:
        text_input = create_text_input(config, use_text_descriptions, img_to_text_map)
        if not text_input:
            raise ValueError(
                (
                    f"ERROR: create_text_input returned empty text for scenario "
                    f"{scenario_label}!"
                )
            )

        # Get facial description for metadata
        facial_description_for_metadata = get_facial_description_for_metadata(
            config, use_text_descriptions, img_to_text_map
        )

        # Extract image input if present (will be None if use_text_descriptions=True)
        image_input = get_image_input(config, use_text_descriptions)
        if image_input:
            print(
                (
                    f"🖼️  Detected image input for scenario {scenario_label}: "
                    f"{image_input}"
                )
            )
        elif use_text_descriptions:
            print(
                (
                    f"📝 Using text descriptions instead of images for scenario "
                    f"{scenario_label}"
                )
            )
    except Exception as e:
        raise Exception(
            f"ERROR: Failed to create inputs for scenario {scenario_label}: {e}"
        )

    # Create input metadata for incremental saving
    input_context = config.get("input_context", {})
    received_feedback = config.get("received_feedback", {})
    labels = config.get("labels", {})
    input_metadata = create_input_metadata(
        scenario_label,
        experiment_name,
        input_context,
        received_feedback,
        "",
        facial_description_for_metadata,
    )

    # Initialize result storage
    predictions_full = {}
    predictions_single = {}
    brier_scores_full = {}
    brier_scores_single = {}
    brier_scores_comparison = {}

    # Body part comfort levels (exclude full arm per request)
    body_parts = ["upper_arm", "forearm", "wrist"]
    for body_part in body_parts:
        print(f"  Predicting comfort level for {body_part}...")

        try:
            # Full output approach
            metadata_full = dict(input_metadata)
            metadata_full["query_type"] = "body_part_full"
            pred_full = get_comfort_level_prediction(
                model,
                text_input,
                body_part,
                incremental_file,
                metadata_full,
                image_input,
            )
            predictions_full[body_part] = pred_full

            # Single token approach
            metadata_single = dict(input_metadata)
            metadata_single["query_type"] = "body_part_single"
            pred_single = get_comfort_level_single_token_prediction(
                model,
                text_input,
                body_part,
                incremental_file,
                metadata_single,
                image_input,
            )
            predictions_single[body_part] = pred_single

            # Calculate Brier scores if labels exist
            if body_part in labels:
                try:
                    # Convert string keys to int keys for labels
                    label_dict = {
                        int(k): float(v) for k, v in labels[body_part].items()
                    }
                    brier_scores_full[body_part] = calculate_brier_score(
                        pred_full, label_dict
                    )
                    brier_scores_single[body_part] = calculate_brier_score(
                        pred_single, label_dict
                    )
                    brier_scores_comparison[body_part] = calculate_brier_score(
                        pred_full, pred_single
                    )
                    print(f"✅ Calculated Brier scores for {body_part}")
                except Exception as brier_err:
                    print(
                        (
                            f"🚨 ERROR: Failed to calculate Brier scores for "
                            f"{body_part}: {brier_err}"
                        )
                    )
                    print(f"🚨 Labels: {labels.get(body_part, 'None')}")
            else:
                print(
                    (
                        f"🔹 No labels found for {body_part} - skipping Brier "
                        f"score calculation"
                    )
                )

        except Exception as e:
            print(f"🚨 ERROR: Failed to predict {body_part}: {e}")
            print(f"🚨 Exception type: {type(e).__name__}")
            # Continue with other body parts instead of stopping entirely

    # Note: Removed verbalized-uncertainty queries for efficiency.

    # Calculate uncertainty metrics for wrist (main analysis)
    wrist_single = predictions_single.get("wrist", {})
    wrist_full = predictions_full.get("wrist", {})

    single_token_entropy = calculate_entropy(wrist_single) if wrist_single else 0.0
    single_token_max_prob = calculate_max_prob(wrist_single) if wrist_single else 0.0

    full_output_entropy = calculate_entropy(wrist_full) if wrist_full else 0.0
    full_output_max_prob = calculate_max_prob(wrist_full) if wrist_full else 0.0

    # Calculate overall Brier scores (using wrist for backward compatibility)
    overall_brier_scores = {}
    if "wrist" in labels:
        wrist_labels = {int(k): float(v) for k, v in labels["wrist"].items()}
        if wrist_single:
            overall_brier_scores["single_token_vs_labels"] = calculate_brier_score(
                wrist_single, wrist_labels
            )
        if wrist_full:
            overall_brier_scores["full_output_vs_labels"] = calculate_brier_score(
                wrist_full, wrist_labels
            )

    return {
        "scenario": scenario_label,
        "predictions_full": predictions_full,
        "predictions_single": predictions_single,
        "brier_scores_full": brier_scores_full,
        "brier_scores_single": brier_scores_single,
        "brier_scores_comparison": brier_scores_comparison,
        "labels": labels,
        # Legacy compatibility fields for existing analysis code
        "single_token": {
            "probs": wrist_single,
            "entropy": single_token_entropy,
            "max_prob": single_token_max_prob,
            "uncertainty": 1 - single_token_max_prob,
        },
        "full_output": {
            "probs": wrist_full,
            "entropy": full_output_entropy,
            "max_prob": full_output_max_prob,
            "uncertainty": 1 - full_output_max_prob,
        },
        "brier_scores": overall_brier_scores,
        "comparisons": {
            "brier_single_vs_full": (
                calculate_brier_score(wrist_full, wrist_single)
                if (wrist_full and wrist_single)
                else 0.0
            )
        },
    }
