"""Test LLM."""

from typing import Any, Dict, List

from conformal_setup.models.openai_model import OpenAIModel


def test_openai_model() -> List[Dict[Any, Any]]:
    """Test OpenAI model."""

    sys_prompt = (
        "You are a system that translates human language feedback into parameters for a planner. "
        "The planner wants to help a robot perform tasks for the human, while making the "
        "human as comfortable as possible. Reply with only one number. "
        "Previously, the robot sensed force 5 on the human's chest, and force 1 on "
        "the human's arm. Currently, the robot senses force 6 on the human chest, and force 1 on "
        'the human\'s arm. The human says "ow, that hurt".'
    )

    model = OpenAIModel(
        model="gpt-4.1-nano",
        system_prompt=sys_prompt,
        temperature=0.2,
        max_tokens=3,
    )

    # Test single input
    input_text = [
        "What should the maximum force threshold be for the human's chest?",
        "What should the maximum force threshold be for the human's arm?",
    ]

    response = model(input_text)

    return response


if __name__ == "__main__":
    result = test_openai_model()
    for r in result:
        print()
        for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True):
            print(f"{k}: {v}")
