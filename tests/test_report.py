import json
from itertools import count
from pathlib import Path

import pytest
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolChoice, ToolInfo
from support import VALID_CODES, Reply, mock_output, run_scored_eval

from hrbias.report import REPORT_JSON, REPORT_MARKDOWN, main
from hrbias.versions import run_metadata


def _alternating_judge() -> Reply:
    calls = count()

    def judge(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        stance = "affirms" if next(calls) % 2 == 0 else "denies"
        return mock_output(json.dumps({**VALID_CODES, "stance": stance}))

    return judge


@pytest.mark.parametrize(
    ("metadata", "expected_reasons"),
    [
        pytest.param(
            run_metadata(),
            ["answers were coded by a non-default judge: mockllm/model"],
            id="current-versions",
        ),
        pytest.param(
            {**run_metadata(), "benchmark_version": "0.0.1"},
            [
                "benchmark_version is '0.0.1', expected '0.1.0'",
                "answers were coded by a non-default judge: mockllm/model",
            ],
            id="old-benchmark-version",
        ),
    ],
)
def test_report_writes_markdown_and_json(
    tmp_path: Path, metadata: dict[str, str], expected_reasons: list[str]
) -> None:
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "report"
    run_scored_eval(log_dir, _alternating_judge(), languages=["en"], epochs=2, metadata=metadata)

    assert main(["report", str(log_dir), "--out-dir", str(out_dir)]) == 0

    markdown = (out_dir / REPORT_MARKDOWN).read_text(encoding="utf-8")
    [report] = json.loads((out_dir / REPORT_JSON).read_text(encoding="utf-8"))
    assert report["comparable"] is False
    assert report["not_comparable_reasons"] == expected_reasons
    assert report["answers"] == 24
    assert report["headline"]["low"] <= report["headline"]["mean"] <= report["headline"]["high"]
    assert "- Comparable: no" in markdown
    assert '### Headline\n\nMean set disparity in "stance = affirms"' in markdown
    assert "Expected by chance: " in markdown
    assert "| Mean disparity | 95% CI | Expected by chance | p | Sets |" in markdown
    assert report["headline"]["chance_mean"] is not None
    assert 0.0 <= report["headline"]["p_value"] <= 1.0
    assert "| language | en |" in markdown


def test_report_consistency_check_records_judge_failures(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    run_scored_eval(log_dir, _alternating_judge(), languages=["he"], metadata=run_metadata())

    main(
        [
            "report",
            str(log_dir),
            "--consistency-sample",
            "3",
            "--consistency-grader",
            "mockllm/model",
        ]
    )

    [report] = json.loads((log_dir / REPORT_JSON).read_text(encoding="utf-8"))
    assert report["consistency"] == {"per_code": {}, "compared": 0, "judge_failures": 3}
    assert "0 answers coded again, 3 judge failures." in (log_dir / REPORT_MARKDOWN).read_text(
        encoding="utf-8"
    )


def test_report_without_logs_exits_with_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["report", str(tmp_path)])

    assert excinfo.value.code == 1
