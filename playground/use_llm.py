"""Use LLM."""

from typing import Any, Dict

from safe_feedback_interpretation.models.openai_model import OpenAIModel


def get_single_token_probs() -> Dict[Any, Any]:
    """Test single token probs."""

    sys_prompt = "Answer with yes or no."

    model = OpenAIModel(
        model="chatgpt-4o-latest",
        system_prompt=sys_prompt,
    )

    # Test single input
    # input_text = "What should the maximum force threshold be for the human's arm?"
    # text_input = (
    #     "I flip a fair 6-sided die. What number do I get? Answer using only one
    #      number."
    # )
    # text_input = "what is the facial expression in the image? answer in only one word"
    text_input = (
        "You are looking at a patient with whom a physical therapist is "
        "doing a physical movement. Here's the feedback you get from the patient:"
        "Patient speech: \"i'm ok, it doesn't hurt\". "
        # "Patient facial expression: [attached image]. "
        "Patient facial expression: wincing. "
        "Is the patient comfortable or not?"
    )
    print(text_input)

    # image_input = "playground/straight.png"
    image_input = "playground/wince.png"

    response = model.get_single_token_logits(text_input, image_input)
    # response = model.get_single_token_logits(text_input)

    return response


def get_full_output() -> str:
    """Test full output."""
    # sys_prompt = "Answer with yes or no."
    sys_prompt = "You are a medical expert."

    model = OpenAIModel(
        model="gpt-5",
        system_prompt=sys_prompt,
    )

    text_input = (
        "You are looking at a patient with whom a physical therapist is "
        "doing a physical movement. Here's the feedback you get from the patient:"
        "Patient speech: \"i'm ok, it doesn't hurt\". "
        "Patient facial expression: [attached image]. "
        "Patient facial expression: wincing. "
        "In the past, the patient has winced but we know that they were comfortable. "
        "They insist that they are comfortable. "
        "Based on their verbal feedback and facial expression, is the patient comfortable? "
        "Explain in depth how certain you are, and give a percentage for certainty."
    )
    print(text_input)

    # image_input = "playground/straight.png"
    image_input = "playground/wince.png"

    response = model.get_full_output(text_input, image_input)
    # response = model.get_full_output(text_input)

    return response


if __name__ == "__main__":
    result = get_single_token_probs()
    for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True):
        print(f"{k}: {v}")

    print(f"Full output: {get_full_output()}")
