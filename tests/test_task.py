from pathlib import Path

import pytest
from inspect_ai import eval
from inspect_ai.model import get_model
from support import FIXTURE_DATA_DIR, MOCK_MODEL, coding_judge, echo_model

from hrbias import BENCHMARK_VERSION
from hrbias.dataset import hrbias_dataset
from hrbias.report import REPORT_JSON, REPORT_MARKDOWN, main
from hrbias.task import DEFAULT_EPOCHS, hrbias
from hrbias.versions import run_metadata


def test_end_to_end_eval_and_report(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    grader = get_model(MOCK_MODEL, custom_outputs=coding_judge())

    [log] = eval(
        hrbias(data_dir=str(FIXTURE_DATA_DIR)),
        model=get_model(MOCK_MODEL, custom_outputs=echo_model),
        model_roles={"grader": grader},
        log_dir=str(log_dir),
        display="none",
    )

    assert log.status == "success", log.error
    assert log.eval.task_version == BENCHMARK_VERSION
    assert log.eval.metadata == run_metadata()
    assert log.eval.config.epochs == DEFAULT_EPOCHS
    assert log.samples is not None
    assert len(log.samples) == len(hrbias_dataset(FIXTURE_DATA_DIR)) * DEFAULT_EPOCHS

    assert main(["report", str(log_dir), "--data-dir", str(FIXTURE_DATA_DIR)]) == 0
    markdown = (log_dir / REPORT_MARKDOWN).read_text(encoding="utf-8")
    assert (log_dir / REPORT_JSON).is_file()
    assert "| detention | t1.stance = affirms | 0.000 |" in markdown
    assert "answers were coded by a non-default judge: mockllm/model" in markdown


@pytest.mark.parametrize(
    ("languages", "expected"),
    [
        pytest.param(None, {"en", "ar", "he"}, id="all"),
        pytest.param("ar, he", {"ar", "he"}, id="comma-separated"),
        pytest.param(["en"], {"en"}, id="list"),
    ],
)
def test_task_language_parameter(languages: str | list[str] | None, expected: set[str]) -> None:
    task = hrbias(languages=languages, data_dir=str(FIXTURE_DATA_DIR))

    assert {s.metadata["language"] for s in task.dataset if s.metadata} == expected
