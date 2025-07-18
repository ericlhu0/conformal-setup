#!/usr/bin/env python3
"""Test script for conformal prediction functionality."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conformal_setup.conformal_prediction.openai_conformal import (
    OpenAIConformalPredictor,
)
from conformal_setup.models.openai_model import OpenAIModel
from conformal_setup.utils import parse_calibration_data


def test_data_parsing():
    """Test parsing calibration data into ConformalDataset."""
    print("Testing data parsing...")

    # Parse calibration data
    data_path = Path(__file__).parent.parent / "data" / "calibration_sets" / "25.txt"
    dataset = parse_calibration_data(str(data_path))

    print(f"Loaded {len(dataset)} calibration examples")

    # Show first example
    if len(dataset) > 0:
        input_text, label = dataset[0]
        print(f"\nFirst example:")
        print(f"Input: {input_text}")
        print(f"Label: {label}")

    return dataset


def test_conformal_prediction():
    """Test conformal prediction with OpenAI model."""
    print("\nTesting conformal prediction...")

    # sys_prompt = "You are part of an assistive robot system that performs tasks where physical contact between the robot and the human user's body is required. In order to minimize discomfort for the user, we want to maintain a maximum force threshold for each part of the user's body, so that the robot doesn't exceed it.\nIn order to build your understanding of the maximum forces permissible at each part of the user's body, the user will provide feedback in natural language intermittently while the robot system performs its task. Your job is to translate human language feedback into parameters for the planner for the robot system.\nTaking into account the history of feedback provided by the user and forces sensed by the user at each body part, make your best guess for the maximum force allowable for each body part.\nAny query sent by me will contain a piece of feedback by the user during the assistive robot's execution, and the current forces sensed at each body part. Reply in strictly the following format:\n[body part 1]: [max force]\n...\n[body part n]: [max force]"

    sys_prompt = "You are part of an assistive robot system that can help reposition the user's limbs. Your job is to translate the user's language feedback and current parameter values into your best estimate of the delta (change) for each parameter that would make the user as comfortbale as possible.\nIn order to build your understanding of the user's desired parameter changes, the user will provide feedback in natural language intermittently while the robot system performs its task.\nAny query sent by me will contain a piece of feedback by the user during the assistive robot's execution, and the current parameter values. I will only ask you for the desired change for one parameter at a time. The range of possible parameter values is [1, 20]. Reply with only the desired change and nothing else. E.g.: +x or 0 or -x, where x is an integer."

    # Create OpenAI model
    model = OpenAIModel(model="gpt-4o-mini", system_prompt=sys_prompt, temperature=0.1)

    # Create conformal predictor
    predictor = OpenAIConformalPredictor(model, confidence_level=0.8)

    # Load calibration data
    data_path = Path(__file__).parent.parent / "data" / "calibration_sets" / "25.txt"
    calibration_dataset = parse_calibration_data(str(data_path))

    # Use first 10 examples for calibration
    if len(calibration_dataset) >= 10:
        # Create smaller calibration set
        cal_inputs = []
        cal_labels = []
        for i in range(10):
            input_text, label = calibration_dataset[i]
            cal_inputs.append(input_text)
            cal_labels.append(label)

        from conformal_setup.conformal_prediction.base_conformal import ConformalDataset

        small_cal_dataset = ConformalDataset(cal_inputs, cal_labels)

        print(f"Calibrating with {len(small_cal_dataset)} examples...")

        try:
            # Calibrate predictor
            predictor.calibrate(small_cal_dataset)
            print("✓ Calibration completed")

            # Test prediction
            test_input = "Current Parameters: {'scale of sampled actions': '10', 'maximum velocity': '15'}\nFeedback: \"Too fast, please slow down\"\nWhat parameter changes are needed?"

            print(f"\nTest input: {test_input}")
            prediction_sets = predictor.predict(test_input)

            print(f"Prediction set size: {len(prediction_sets[0])}")
            print(
                f"Prediction set: {prediction_sets[0][:3]}..."
            )  # Show first 3 predictions

        except Exception as e:
            print(f"Error during conformal prediction: {e}")
    else:
        print("Not enough calibration data (need at least 10 examples)")


def main():
    """Run all tests."""
    print("=== Conformal Prediction Test Script ===\n")

    dataset = test_data_parsing()
    print(dataset)

    # try:
    #     # Test data parsing
    #     dataset = test_data_parsing()

    #     # Test conformal prediction
    #     test_conformal_prediction()

    #     print("\n✓ All tests completed!")

    # except Exception as e:
    #     print(f"❌ Test failed: {e}")
    #     import traceback

    #     traceback.print_exc()


if __name__ == "__main__":
    main()
