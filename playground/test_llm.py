"""Test LLM."""

from typing import Any, Dict, List
import numpy as np

from conformal_setup.models.openai_model import OpenAIModel


def test_openai_model() -> List[Dict[Any, Any]]:
    """Test OpenAI model."""

    # contact pref
    # sys_prompt = "You are part of an assistive robot system that performs tasks where physical contact between the robot and the human user's body is required. In order to minimize discomfort for the user, we want to maintain a maximum force threshold for each part of the user's body, so that the robot doesn't exceed it.\nIn order to build your understanding of the maximum forces permissible at each part of the user's body, the user will provide feedback in natural language intermittently while the robot system performs its task. Your job is to translate human language feedback into parameters for the planner for the robot system.\nTaking into account the history of feedback provided by the user and forces sensed by the user at each body part, make your best guess for the maximum force allowable for each body part.\nAny query sent by me will contain a piece of feedback by the user during the assistive robot's execution, and the current forces sensed at each body part. Reply in strictly the following format:\n[body part 1]: [max force]\n...\n[body part n]: [max force]"
    
    # limb repo
    sys_prompt = (
        "You are part of an assistive robot system that helps reposition the user’s limbs. "
        "Your role is to interpret the user’s natural language feedback, along with the current parameter values, "
        "and determine the best delta (change) for a specific parameter to maximize the user’s comfort.\n\n"
        "Here’s how this process works:\n"
        "- I will send you a query that contains:\n"
        "  1. A piece of feedback given by the user during the assistive task.\n"
        "  2. The current parameter values.\n"
        "- I will only ever ask you for the desired change for one parameter at a time.\n\n"
        "Descriptions of each parameter:\n"
        "- sampled action scale: the scale of the noise used to sample torques in the MPC (higher gives higher torques).\n"
        "- max velocity: the maximum velocity allowed in the MPC rollouts (higher allows for higher velocities).\n"
        "- distance-from-collision cost: weighs how important collision avoidance is in the MPC evaluation (higher is more careful about collisions).\n"
        "- goal-reaching cost: weighs how much the evaluation values going directly to the goal (higher goes more directly to the goal).\n\n"
        "Important details about your response:\n"
        "- The range of possible parameter values is [1, 20].\n"
        "- Reply with only a single integer, representing how the feedback translates into a numerical change for the parameter, using the following encoding:\n\n"
        "    - If the parameter is good and needs no change (0), reply with 20.\n"
        "    - If the user feedback suggests increasing the parameter by +x, reply with 20 + x.\n"
        "    - If the feedback suggests decreasing the parameter by -x, reply with x (just x, no addition to 20).\n\n"
        "- Always shift your predictions by +20 following these rules.\n\n"
        "Do not include any explanation, comments, or additional text—only the integer result."
    )



    model = OpenAIModel(
        model="gpt-4o",
        system_prompt=sys_prompt,
        temperature=0.2,
        max_tokens=3,
    )

    # Test single input
    # input_text = [
    #     "What should the maximum force threshold be for the human's chest?",
    #     "What should the maximum force threshold be for the human's arm?",
    # ]

    base = "These are the current parameter values: {'scale of sampled actions': '6', 'maximum velocity': '13', 'MPC weight from distance-to-collision cost': '12', 'MPC weight from goal-reaching cost': '4'}. The user says \"that almost hit me\"."

    input_text = [
        base + " What should the change for the 'scale of sampled actions' parameter be?",
        base + " What should the change for the 'maximum velocity' parameter be?",
        base + " What should the change for the 'MPC weight from distance-to-collision cost' parameter be?",
        base + " What should the change for the 'MPC weight from goal-reaching cost' parameter be?",
    ]

    response = model(input_text)

    return response


if __name__ == "__main__":
    base = "These are the current parameter values: {'scale of sampled actions': '6', 'maximum velocity': '13', 'MPC weight from distance-to-collision cost': '12', 'MPC weight from goal-reaching cost': '4'}. The user says \" that was a bit close to the armrest\"."

    input_text = [
        base + " What should the change for the 'scale of sampled actions' parameter be?",
        base + " What should the change for the 'maximum velocity' parameter be?",
        base + " What should the change for the 'MPC weight from distance-to-collision cost' parameter be?",
        base + " What should the change for the 'MPC weight from goal-reaching cost' parameter be?",
    ]
    result = test_openai_model()
    for i in range(len(result)):
        print(input_text[i])
        r = result[i]
        for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True):
            print(f"{k}: {np.exp(v)}")
