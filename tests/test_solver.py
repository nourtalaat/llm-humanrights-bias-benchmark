from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model
from inspect_ai.tool import ToolChoice, ToolInfo
from support import MOCK_MODEL, mock_output

from hrbias.solver import scripted_conversation

TURNS = ["First question?", "Second question?", "Third question?"]


@pytest.mark.parametrize(
    ("metadata", "expected_user_turns"),
    [
        pytest.param({"turns": TURNS}, TURNS, id="multi-turn"),
        pytest.param({"turns": TURNS[:1]}, TURNS[:1], id="single-turn"),
        pytest.param({}, TURNS[:1], id="no-turns"),
    ],
)
def test_scripted_conversation_sends_every_turn_after_an_answer(
    tmp_path: Path, metadata: dict[str, Any], expected_user_turns: list[str]
) -> None:
    seen: list[list[tuple[str, str]]] = []

    def model(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        seen.append([(m.role, m.text) for m in messages])
        return mock_output(f"answer {len(seen)}")

    task = Task(
        dataset=[Sample(input=TURNS[0], metadata=metadata)],
        solver=scripted_conversation(),
    )
    [log] = eval(
        task,
        model=get_model(MOCK_MODEL, custom_outputs=model),
        display="none",
        log_dir=str(tmp_path),
    )

    assert log.status == "success", log.error
    assert log.samples is not None
    expected: list[tuple[str, str]] = []
    for number, turn in enumerate(expected_user_turns, 1):
        expected.append(("user", turn))
        assert seen[number - 1] == expected
        expected.append(("assistant", f"answer {number}"))

    assert len(seen) == len(expected_user_turns)
    assert [(m.role, m.text) for m in log.samples[0].messages] == expected
