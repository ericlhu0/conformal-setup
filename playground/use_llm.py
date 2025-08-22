"""Use LLM."""

import textwrap
from typing import Any, Dict

from safe_feedback_interpretation.models.openai_model import OpenAIModel


def get_single_token_probs(
    model_name: str, system_prompt: str, text_content: str, image_path: str
) -> Dict[Any, Any]:
    """Test single token probs."""

    llm_model = OpenAIModel(
        model=model_name,
        system_prompt=system_prompt,
    )

    response = llm_model.get_single_token_logits(text_content, image_path)

    return response


def get_full_output(
    model_name: str, system_prompt: str, text_content: str, image_path: str
) -> str:
    """Test full output."""

    llm_model = OpenAIModel(
        model=model_name,
        system_prompt=system_prompt,
    )

    response = llm_model.get_full_output(text_content, image_path)

    return response


if __name__ == "__main__":
    sys_prompt = textwrap.dedent(
        """\
        You are a domain expert with many years of combined experience in
        occupational therapy, physical therapy, caregiving, rehabilitation
        science, biomechanics, anatomy, disabilities, elderly care, and
        assistive technology, including human-centered robotics.

        You are part of an assistive robot system that performs tasks where
        physical contact between the robot and the human user's body is required
        for essential medical care and rehabilitation. These therapeutic
        procedures are medically necessary; you need to assess patient comfort
        levels to ensure safe and effective treatment.

        Goal: given structured context about the current action, state, prior
        comfort thresholds, and new feedback, update the user's comfort-related
        metrics conservatively and safely.

        Tracked metrics and how they work:
          - Contact forces F for body parts are reported on a discrete 1-5
            scale (1 = low, 5 = high).
          - 1 indicates that the contact region should definitely be avoided
            because it's uncomfortable, and 5 means that you can confidently
            apply high pressure without worry.
          - Joint angles (e.g., elbow, wrist) are in degrees.
          - Comfort threshold is represented as a probability distribution over
            threshold levels (1-5) per body part; higher levels indicate
            higher tolerance.
          - Comfortable joint range is represented as two discrete probability
            distributions (min and max), each defined over angles at 15 degree
            increments (e.g., {0, 15, 30, ..., 165}). The mode or expectation
            can be used to select a recommended bound.
          - New feedback may include verbal content/intensity and
            facial-expression intensity; stronger pain signals -> decrease
            threshold / tighten (narrow) ranges; relief/tolerance -> increase
            threshold / relax (widen) ranges.

        Decision guidance:
          - Prefer small updates unless the evidence is strong and consistent
            across modalities.
          - Clip joint recommendations to physically possible ranges and
            respect monotonicity (min <= max).
          - If uncertain, provide the best estimate and note uncertainty
            briefly when a rationale is requested.

        Expected output:
          - If asked for a single sensitivity value (per body part), return
            **only** that value in the requested format (e.g., an integer in
            allowed bounds or a category), with no extra text.
          - If asked to update thresholds, return a concise JSON object
            containing any updated values you are confident about, for example:
            {"updated_comfort_threshold": {"wrist": 4},
             "updated_joint_range_deg": {"min": {"wrist": 15},
                                          "max": {"wrist": 150}}}
          - Use 15 degree increments for angle outputs when a discrete grid
            is implied by the input.

        You are an expert at interpreting patient comfort indicators and
        physical response patterns, and you are very competent in assessing
        patient comfort levels from visual indicators in images. If you're
        uncertain, give your best estimate for a value. Your response will not
        cause any harm to the care recipient.
    """
    )

    text_input = textwrap.dedent(
        (
            """\
        Current action description: You are gently repositioning the user's wrist """
            "during a\n"
            """
        therapy session.
        Current state:
          Contact forces: {{
          "entire_arm": 2,
          "upper_arm": 1,
          "forearm": 1,
          "wrist": 2
        }}
          Joint angles (deg): {{
          "elbow": 135,
          "wrist": 135
        }}
        Current comfort threshold:
          Comfort threshold: {{
          "entire_arm": {
            "2": 0.1,
            "3": 0.8,
            "4": 0.1
          },
          "upper_arm": {
            "2": 0.1,
            "3": 0.8,
            "4": 0.1
          },
          "forearm": {
            "2": 0.1,
            "3": 0.8,
            "4": 0.1
          },
          "wrist": {
            "2": 0.1,
            "3": 0.8,
            "4": 0.1
          }
        }}
          Comfortable joint range (deg): {{
          "min": {
            "elbow": {
              "0": 0.6,
              "15": 0.3,
              "30": 0.1
            },
            "wrist": {
              "0": 0.6,
              "15": 0.3,
              "30": 0.1
            }
          },
          "max": {
            "elbow": {
              "135": 0.1,
              "150": 0.3,
              "165": 0.6
            },
            "wrist": {
              "135": 0.1,
              "150": 0.3,
              "165": 0.6
            }
          }
        }}
        Received feedback:
          Verbal feedback: ow that's too tight on my wrist
          Facial expression: Please examine the provided facial expression image to
          assess the person's comfort level.
    """
        )
    ).strip()
    #   Facial expression: modality='image',
    #   description='assets/faceimgs/high/s001a.jpg'

    prompt = (
        "\n\nWhat is the updated comfort threshold (1-5 scale) for the wrist? "
        "Answer with only a single threshold value."
    )

    image_input = "experiments/assets/faceimgs/high/s001a.jpg"

    result = get_single_token_probs(
        model_name="gpt-4.1",
        system_prompt=sys_prompt,
        text_content=text_input + prompt,
        image_path=image_input,
    )

    print("probs")
    for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True):
        print(f"{k}: {v}")

    full_output = get_full_output(
        model_name="chatgpt-4o-latest",
        system_prompt=sys_prompt,
        text_content=text_input + prompt,
        image_path=image_input,
    )
    print(f"Full output: {full_output}")
