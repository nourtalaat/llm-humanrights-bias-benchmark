"""Shared test helpers for mock models, judge replies and small eval runs."""

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model
from inspect_ai.tool import ToolChoice, ToolInfo

from hrbias.codebook import CODEBOOK, POSITION_CHANGE, POSITION_CHANGE_CODE, CodeDefinition
from hrbias.dataset import hrbias_dataset
from hrbias.scorer import hrbias_scorer
from hrbias.solver import scripted_conversation

MOCK_MODEL = "mockllm/model"
FIXTURE_DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "data"

Reply = Callable[[list[ChatMessage], list[ToolInfo], ToolChoice, GenerateConfig], ModelOutput]
Chooser = Callable[[str, str], str | None]
"""Picks the shown value for (turn key, shown code), or ``None`` for the default."""

SHOWN_CODES: dict[str, CodeDefinition] = {
    definition.shown_name or code: definition
    for codes in [*CODEBOOK.values(), {POSITION_CHANGE: POSITION_CHANGE_CODE}]
    for code, definition in codes.items()
}
"""Every code by the name the judge is shown."""

_LAYOUT_LINE = re.compile(r"^(t\d+): (.+)$", re.MULTILINE)


def mock_output(text: str) -> ModelOutput:
    """Build a mockllm reply holding ``text``."""
    return ModelOutput.from_content(model=MOCK_MODEL, content=text)


def echo_model(
    messages: list[ChatMessage],
    tools: list[ToolInfo],
    tool_choice: ToolChoice,
    config: GenerateConfig,
) -> ModelOutput:
    """A model under test that repeats the last user turn, so its answer names the case."""
    return mock_output(f"About this: {messages[-1].text}")


def sheet_for_prompt(prompt: str, choose: Chooser | None = None) -> dict[str, dict[str, str]]:
    """Build a valid answer sheet for the codes a judge prompt asks for.

    Every code gets its first shown value unless ``choose`` returns another.
    """
    sheet = {}
    for turn_key, codes in _LAYOUT_LINE.findall(prompt):
        sheet[turn_key] = {}
        for code in codes.split(", "):
            definition = SHOWN_CODES[code]
            chosen = choose(turn_key, code) if choose else None
            sheet[turn_key][code] = chosen or definition.shown_value(next(iter(definition.values)))

    return sheet


def coding_judge(choose: Chooser | None = None) -> Reply:
    """A judge that fills every code its prompt asks for, as ``sheet_for_prompt`` does."""

    def judge(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        return mock_output(json.dumps(sheet_for_prompt(messages[0].text, choose)))

    return judge


def run_scored_eval(
    log_dir: Path,
    judge: Reply,
    *,
    leak_check: bool = False,
    languages: Sequence[str] | None = None,
    tiers: Sequence[str] | None = None,
    epochs: int = 1,
    metadata: dict[str, str] | None = None,
) -> EvalLog:
    """Run the fixture dataset, solver and scorer with mockllm for the model and the grader."""
    task = Task(
        dataset=hrbias_dataset(FIXTURE_DATA_DIR, languages, tiers=tiers),
        solver=scripted_conversation(),
        scorer=hrbias_scorer(data_dir=str(FIXTURE_DATA_DIR), leak_check=leak_check),
        epochs=epochs,
        metadata=metadata,
    )
    [log] = eval(
        task,
        model=get_model(MOCK_MODEL, custom_outputs=echo_model),
        model_roles={"grader": get_model(MOCK_MODEL, custom_outputs=judge)},
        log_dir=str(log_dir),
        display="none",
    )
    assert log.status == "success", log.error
    return log
