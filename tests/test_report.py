import json
from itertools import count
from pathlib import Path

import pytest
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput
from inspect_ai.tool import ToolChoice, ToolInfo
from support import (
    FIXTURE_DATA_DIR,
    Reply,
    coding_judge,
    mock_output,
    run_scored_eval,
    sheet_for_prompt,
)

from hrbias import BENCHMARK_VERSION
from hrbias.report import GAP_FORMAT, REPORT_JSON, REPORT_MARKDOWN, main
from hrbias.versions import run_metadata

NON_DEFAULT_JUDGE = "answers were coded by a non-default judge: mockllm/model"
ALTERNATING = {"responsibility": ("group_a", "group_b"), "stance": ("affirms", "denies")}


def _alternating_judge() -> Reply:
    calls = count()

    def choose(turn: str, code: str) -> str | None:
        options = ALTERNATING.get(code)
        return options[next(calls) % 2] if options else None

    return coding_judge(choose)


@pytest.mark.parametrize(
    ("metadata", "expected_reasons"),
    [
        pytest.param(run_metadata(), [NON_DEFAULT_JUDGE], id="current-versions"),
        pytest.param(
            {**run_metadata(), "benchmark_version": "0.0.1"},
            [f"benchmark_version is '0.0.1', expected '{BENCHMARK_VERSION}'", NON_DEFAULT_JUDGE],
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

    exit_code = main(
        ["report", str(log_dir), "--out-dir", str(out_dir), "--data-dir", str(FIXTURE_DATA_DIR)]
    )

    assert exit_code == 0
    markdown = (out_dir / REPORT_MARKDOWN).read_text(encoding="utf-8")
    [report] = json.loads((out_dir / REPORT_JSON).read_text(encoding="utf-8"))
    assert report["comparable"] is False
    assert report["not_comparable_reasons"] == expected_reasons
    assert report["answers"] == 24
    tiers = {t["tier"]: t for t in report["tiers"]}
    assert set(tiers) == {"real_event", "mirrored"}
    assert tiers["real_event"]["answers"] + tiers["mirrored"]["answers"] == 24
    assert [(h["family"], h["code"], h["value"]) for h in tiers["real_event"]["headlines"]] == [
        ("suffering", "t1.acknowledges_suffering", "yes"),
        ("war_origin", "t2.responsibility", "perpetrator"),
        ("war_origin", "t3.position_change", "holds"),
    ]
    [mirrored] = tiers["mirrored"]["headlines"]
    assert (mirrored["family"], mirrored["code"], mirrored["value"]) == (
        "detention",
        "t1.stance",
        "affirms",
    )
    estimate = mirrored["estimate"]
    assert estimate["low"] <= estimate["mean"] <= estimate["high"]
    assert estimate["chance_mean"] is not None
    assert 0.0 <= estimate["p_value"] <= 1.0
    assert {b["dimension"] for b in tiers["real_event"]["breakdowns"]} == {
        "case_set",
        "language",
        "framing",
    }
    assert {g["actor_id"] for g in tiers["mirrored"]["victim_gaps"]} == {"kestrish", "valdorians"}
    assert {g["actor_id"] for g in tiers["mirrored"]["perpetrator_gaps"]} == {
        "valdorian-government",
        "kestrish-army",
    }
    assert not any(g["family"] == "suffering" for g in tiers["real_event"]["perpetrator_gaps"])
    assert any(g["family"] == "suffering" for g in tiers["real_event"]["victim_gaps"])

    assert "- Comparable: no" in markdown
    assert "### Tier: real_event (20 answers)" in markdown
    assert "### Tier: mirrored (4 answers)" in markdown
    assert (
        "| Family | Code = value | Mean disparity | 95% CI | Expected by chance | p | Sets |"
        in (markdown)
    )
    assert "| war_origin | t2.responsibility = perpetrator | " in markdown
    assert "| detention | t1.stance = affirms | case_set | mirrored-detention |" in markdown
    assert "#### Victim gaps (headline codes)" in markdown
    assert "| en | detention | t1.stance = affirms | kestrish |" in markdown
    assert "| en | detention | t1.stance = affirms | kestrish-army |" in markdown
    assert "| Tier | Language | Case set | Case | Failed | Total | Rate |" in markdown


WAR_ORIGIN_GAPS = {
    "victim": {
        ("t2.responsibility", "perpetrator", "valdorians"): "+0.500",
        ("t2.responsibility", "perpetrator", "kestrish"): "-0.500",
        ("t3.position_change", "holds", "valdorians"): "+0.000",
        ("t3.position_change", "holds", "kestrish"): "+0.000",
    },
    "perpetrator": {
        ("t2.responsibility", "perpetrator", "kestrish-army"): "+0.500",
        ("t2.responsibility", "perpetrator", "valdorian-government"): "-0.500",
        ("t3.position_change", "holds", "kestrish-army"): "+0.000",
        ("t3.position_change", "holds", "valdorian-government"): "+0.000",
    },
}


def test_gap_tables_label_each_headline_of_a_two_headline_family(tmp_path: Path) -> None:
    def judge(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        prompt = messages[0].text
        blames_group_a = "2031" in prompt

        def choose(turn: str, code: str) -> str | None:
            if code == "responsibility":
                return "group_a" if blames_group_a else "group_b"

            return "holds" if code == "position_change" else None

        return mock_output(json.dumps(sheet_for_prompt(prompt, choose)))

    log_dir = tmp_path / "logs"
    run_scored_eval(log_dir, judge, languages=["en"], tiers=["real_event"])

    assert main(["report", str(log_dir), "--data-dir", str(FIXTURE_DATA_DIR)]) == 0

    [report] = json.loads((log_dir / REPORT_JSON).read_text(encoding="utf-8"))
    [tier] = report["tiers"]
    markdown = (log_dir / REPORT_MARKDOWN).read_text(encoding="utf-8")
    victim_section, _, perpetrator_section = markdown.partition("#### Perpetrator gaps")
    sections = {
        "victim": victim_section.partition("#### Victim gaps")[2],
        "perpetrator": perpetrator_section,
    }
    for role, expected in WAR_ORIGIN_GAPS.items():
        gaps = {
            (g["code"], g["value"], g["actor_id"]): GAP_FORMAT.format(g["mean_gap"])
            for g in tier[f"{role}_gaps"]
            if g["family"] == "war_origin"
        }
        assert len([g for g in tier[f"{role}_gaps"] if g["family"] == "war_origin"]) == 4
        assert gaps == expected
        rows = [line for line in sections[role].splitlines() if "| war_origin |" in line]
        assert len(rows) == 4
        for (code, value, actor), gap in expected.items():
            assert f"| en | war_origin | {code} = {value} | {actor} | {gap} | 3 |" in rows


def test_report_consistency_check_records_judge_failures(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    run_scored_eval(log_dir, _alternating_judge(), languages=["he"], metadata=run_metadata())

    main(
        [
            "report",
            str(log_dir),
            "--data-dir",
            str(FIXTURE_DATA_DIR),
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
