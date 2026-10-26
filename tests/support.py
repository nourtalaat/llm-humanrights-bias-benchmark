"""Shared test helpers for mock models, codebook replies and small eval runs."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model
from inspect_ai.tool import ToolChoice, ToolInfo

from hrbias.dataset import hrbias_dataset
from hrbias.scorer import hrbias_scorer
from hrbias.solver import scripted_conversation

MOCK_MODEL = "mockllm/model"
FIXTURE_DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "data"

VALID_CODES = {
    "refusal": "none",
    "stance": "affirms",
    "hedging": "light",
    "both_sides": "no",
    "blames_group": "no",
    "rights_conditional_on_security": "yes",
}
VALID_SHEET_JSON = json.dumps(VALID_CODES)

Reply = Callable[[list[ChatMessage], list[ToolInfo], ToolChoice, GenerateConfig], ModelOutput]


def mock_output(text: str) -> ModelOutput:
    """Build a mockllm reply holding ``text``."""
    return ModelOutput.from_content(model=MOCK_MODEL, content=text)


def echo_model(
    messages: list[ChatMessage],
    tools: list[ToolInfo],
    tool_choice: ToolChoice,
    config: GenerateConfig,
) -> ModelOutput:
    """A model under test that repeats the prompt, so its answer names the group."""
    return mock_output(f"About this: {messages[-1].text}")


def run_scored_eval(
    log_dir: Path,
    judge: Reply,
    *,
    leak_check: bool = False,
    languages: Sequence[str] | None = None,
    epochs: int = 1,
    metadata: dict[str, str] | None = None,
) -> EvalLog:
    """Run the hrbias dataset and scorer with mockllm for the model and the grader."""
    task = Task(
        dataset=hrbias_dataset(FIXTURE_DATA_DIR, languages),
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
