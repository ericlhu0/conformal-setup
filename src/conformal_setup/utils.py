"""Utility functions."""

import re
from pathlib import Path
from typing import List, Tuple

from conformal_setup.benchmarks.base_benchmark import Benchmark
from conformal_setup.conformal_prediction.base_conformal import ConformalDataset
from conformal_setup.structs import Action, Task


def plan_is_valid(plan: list[Action], task: Task, benchmark: Benchmark) -> bool:
    """Checks if the plan solves the task."""
    state = task.init
    for action in plan:
        state = benchmark.get_next_state(state, action)
    return benchmark.check_goal(state, task.goal)


def get_plan_cost(plan: list[Action], task: Task, benchmark: Benchmark) -> float:
    """Get the total plan cost."""
    cost = 0.0
    state = task.init
    for action in plan:
        next_state = benchmark.get_next_state(state, action)
        cost += benchmark.get_cost(state, action, next_state)
        state = next_state
    return cost


def parse_calibration_data(file_path: str) -> ConformalDataset:
    """Parse calibration data file into ConformalDataset.

    Expected format:
    Entry #N

    Current Parameter Values:
    parameter1: value1
    parameter2: value2
    ...

    Feedback: "feedback text"

    Desired Changes in Parameter Values:
    parameter1: change1
    parameter2: change2
    ...

    ---

    Args:
        file_path: Path to calibration data file

    Returns:
        ConformalDataset with (input, label) pairs where:
        - input: formatted context with current parameters and feedback
        - label: string representation of desired parameter changes
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {file_path}")

    content = path.read_text()
    entries = content.split("---")

    inputs = []
    labels = []

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Extract entry components using regex
        entry_match = re.search(r"Entry #(\d+)", entry)
        if not entry_match:
            continue

        # Extract current parameters
        current_params = {}
        current_section = re.search(
            r"Current Parameter Values:\s*\n(.*?)\n\nFeedback:", entry, re.DOTALL
        )
        if current_section:
            for line in current_section.group(1).split("\n"):
                if ":" in line:
                    param, value = line.split(":", 1)
                    current_params[param.strip()] = value.strip()

        # Extract feedback
        feedback_match = re.search(r'Feedback: "([^"]*)"', entry)
        feedback = feedback_match.group(1) if feedback_match else ""

        # Extract desired changes
        desired_changes = {}
        changes_section = re.search(
            r"Desired Changes in Parameter Values:\s*\n(.*?)(?:\n---|\Z)",
            entry,
            re.DOTALL,
        )
        if changes_section:
            for line in changes_section.group(1).split("\n"):
                if ":" in line:
                    param, change = line.split(":", 1)
                    desired_changes[param.strip()] = change.strip()

        # Format input as context for the model
        input_text = f'Current Parameters: {current_params}\nFeedback: "{feedback}"\nWhat parameter changes are needed?'

        # Format label as string representation of changes
        label_text = str(desired_changes)

        inputs.append(input_text)
        labels.append(label_text)

    return ConformalDataset(inputs, labels)
