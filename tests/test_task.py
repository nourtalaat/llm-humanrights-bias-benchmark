import json
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import eval
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model
from inspect_ai.tool import ToolChoice, ToolInfo
from support import FIXTURE_DATA_DIR, MOCK_MODEL, coding_judge, echo_model

from hrbias import BENCHMARK_VERSION
from hrbias.dataset import DatasetError, hrbias_dataset
from hrbias.report import REPORT_JSON, REPORT_MARKDOWN, main
from hrbias.task import DEFAULT_EPOCHS, hrbias
from hrbias.versions import run_metadata

SCORER_NAME = "hrbias_scorer"
MULTI_TURN_SAMPLE = "war-origin.presupposing.p0.he.tesk-2034"


def test_benchmark_version() -> None:
    assert BENCHMARK_VERSION == "0.2.0"


def test_end_to_end_eval_and_report(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    model_calls: list[int] = []

    def model(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        model_calls.append(len(messages))
        return echo_model(messages, tools, tool_choice, config)

    [log] = eval(
        hrbias(data_dir=str(FIXTURE_DATA_DIR)),
        model=get_model(MOCK_MODEL, custom_outputs=model),
        model_roles={"grader": get_model(MOCK_MODEL, custom_outputs=coding_judge())},
        log_dir=str(log_dir),
        display="none",
    )

    assert log.status == "success", log.error
    assert log.eval.task_version == BENCHMARK_VERSION
    assert log.eval.metadata == run_metadata()
    assert log.eval.config.epochs == DEFAULT_EPOCHS
    assert log.samples is not None
    dataset = hrbias_dataset(FIXTURE_DATA_DIR)
    assert len(log.samples) == len(dataset) * DEFAULT_EPOCHS
    turns = sum(len(s.metadata["turns"]) for s in dataset if s.metadata)
    assert len(model_calls) == turns * DEFAULT_EPOCHS

    sample = next(s for s in log.samples if s.id == MULTI_TURN_SAMPLE)
    assert [m.role for m in sample.messages] == ["user", "assistant"] * 3
    assert [m.text for m in sample.messages if m.role == "user"] == sample.metadata["turns"]
    score = sample.scores[SCORER_NAME] if sample.scores else None
    assert score is not None
    assert isinstance(score.value, dict)
    assert score.value["t3.position_change"] == "holds"
    assert score.value["t2.responsibility"] == "perpetrator"

    assert main(["report", str(log_dir), "--data-dir", str(FIXTURE_DATA_DIR)]) == 0
    markdown = (log_dir / REPORT_MARKDOWN).read_text(encoding="utf-8")
    [report] = json.loads((log_dir / REPORT_JSON).read_text(encoding="utf-8"))
    assert report["answers"] == len(log.samples)
    assert "| detention | t1.stance = affirms | 0.000 |" in markdown
    assert "| war_origin | t3.position_change = holds | 0.000 |" in markdown
    assert "answers were coded by a non-default judge: mockllm/model" in markdown


@pytest.mark.parametrize(
    ("parameters", "key", "expected"),
    [
        pytest.param({}, "language", {"en", "ar", "he"}, id="all-languages"),
        pytest.param({"languages": "ar, he"}, "language", {"ar", "he"}, id="languages-string"),
        pytest.param({"languages": ["en"]}, "language", {"en"}, id="languages-list"),
        pytest.param(
            {"families": "war_origin,detention"},
            "family",
            {"war_origin", "detention"},
            id="families-string",
        ),
        pytest.param({"families": ["suffering"]}, "family", {"suffering"}, id="families-list"),
        pytest.param({"tiers": "mirrored"}, "tier", {"mirrored"}, id="tiers-string"),
        pytest.param({}, "tier", {"real_event", "mirrored"}, id="all-tiers"),
    ],
)
def test_task_filters(parameters: dict[str, Any], key: str, expected: set[str]) -> None:
    task = hrbias(**parameters, data_dir=str(FIXTURE_DATA_DIR))

    assert {s.metadata[key] for s in task.dataset if s.metadata} == expected


def test_task_rejects_unknown_family() -> None:
    with pytest.raises(DatasetError, match="unknown families"):
        hrbias(families="weather", data_dir=str(FIXTURE_DATA_DIR))
