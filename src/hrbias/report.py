"""``hrbias report``: turn Inspect logs into a Markdown and a JSON report.

Each log (one model run) gets its own section. A log is marked not comparable
when its benchmark version, codebook version or default judge pin differs from
this code's, or when any answer was coded by a judge other than the default.
"""

import argparse
import asyncio
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, Model, get_model

from hrbias.codebook import CODEBOOK, CODES
from hrbias.dataset import DEFAULT_DATA_DIR, BenchmarkData, load_benchmark_data
from hrbias.judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_MODEL_ARGS,
    CodeAgreement,
    RejudgeItem,
    judge_consistency,
)
from hrbias.metrics import (
    BREAKDOWN_DIMENSIONS,
    CONFIDENCE_LEVEL,
    HEADLINE_CODE,
    HEADLINE_VALUE,
    CodedAnswer,
    Estimate,
    FailureRate,
    LeakRate,
    MeanGroupGap,
    SetInfo,
    breakdown,
    failure_rates,
    group_gaps,
    group_rates,
    leak_rates,
    mean_disparity,
    mean_group_gaps,
    permuted_disparities,
    set_disparities,
)
from hrbias.scorer import JUDGE_STATUS, JUDGE_TEMPERATURE
from hrbias.versions import run_metadata

SCORER_NAME = "hrbias_scorer"
REPORT_MARKDOWN = "report.md"
REPORT_JSON = "report.json"
CONSISTENCY_SEED = 7
RATE_FORMAT = "{:.3f}"
GAP_FORMAT = "{:+.3f}"
MISSING_VALUE = "n/a"
PERMUTED_FIELD = "permuted"
_LEVEL = f"{CONFIDENCE_LEVEL:.0%}"
_ESTIMATE_COLUMNS = ["Mean disparity", f"{_LEVEL} CI", "Expected by chance", "p", "Sets"]


class ReportError(RuntimeError):
    """The report could not be built from the given logs."""


@dataclass(frozen=True)
class LogReport:
    """Everything the report says about one log."""

    log_file: str
    task: str
    model: str
    created: str
    epochs: int
    answers: int
    comparable: bool
    not_comparable_reasons: list[str]
    headline: Estimate | None
    breakdowns: dict[str, dict[str, Estimate]]
    code_disparities: dict[str, Estimate]
    group_gaps: list[MeanGroupGap]
    failures: list[FailureRate]
    leaks: list[LeakRate]
    consistency: CodeAgreement | None = None
    details: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``hrbias`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        markdown_path, json_path = run_report(
            log_dir=Path(args.log_dir),
            out_dir=Path(args.out_dir) if args.out_dir else Path(args.log_dir),
            data_dir=Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR,
            consistency_sample=args.consistency_sample,
            consistency_grader=args.consistency_grader,
        )
    except ReportError as err:
        parser.exit(1, f"hrbias report: {err}\n")

    print(f"wrote {markdown_path} and {json_path}")
    return 0


def run_report(
    log_dir: Path,
    out_dir: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    consistency_sample: int = 0,
    consistency_grader: str | None = None,
) -> tuple[Path, Path]:
    """Build the report for every hrbias log in ``log_dir`` and write both files."""
    logs = load_hrbias_logs(log_dir)
    if not logs:
        raise ReportError(f"no hrbias logs found in {log_dir}")

    data = load_benchmark_data(data_dir)
    grader = _consistency_model(consistency_grader) if consistency_sample > 0 else None
    reports = [build_log_report(log, data, consistency_sample, grader) for log in logs]

    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / REPORT_MARKDOWN
    json_path = out_dir / REPORT_JSON
    markdown_path.write_text(render_markdown(reports), encoding="utf-8")
    json_path.write_text(
        json.dumps([asdict(r) for r in reports], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return markdown_path, json_path


def load_hrbias_logs(log_dir: Path) -> list[EvalLog]:
    """Read every log in ``log_dir`` whose samples were scored by the hrbias scorer."""
    logs = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        if any(SCORER_NAME in (s.scores or {}) for s in log.samples or []):
            logs.append(log)

    return logs


def build_log_report(
    log: EvalLog,
    data: BenchmarkData,
    consistency_sample: int = 0,
    grader: Model | None = None,
) -> LogReport:
    """Compute every metric for one log."""
    answers = coded_answers(log)
    rates = group_rates(answers)
    gaps = group_gaps(rates)
    disparities = set_disparities(rates, permuted_disparities(answers))
    reasons = not_comparable_reasons(log)
    code_disparities = {}
    for code in CODES:
        for value in CODEBOOK[code].values:
            estimate = mean_disparity(disparities, code, value)
            if estimate is not None:
                code_disparities[f"{code}={value}"] = estimate

    consistency = None
    if grader is not None and consistency_sample > 0:
        consistency = _consistency(log, data, consistency_sample, grader)

    return LogReport(
        log_file=Path(log.location).name,
        task=log.eval.task,
        model=log.eval.model,
        created=log.eval.created,
        epochs=log.eval.config.epochs or 1,
        answers=len(answers),
        comparable=not reasons,
        not_comparable_reasons=reasons,
        headline=mean_disparity(disparities),
        breakdowns={dim: breakdown(disparities, dim) for dim in BREAKDOWN_DIMENSIONS},
        code_disparities=code_disparities,
        group_gaps=mean_group_gaps(gaps),
        failures=failure_rates(answers),
        leaks=leak_rates(answers),
        consistency=consistency,
        details={
            "group_rates": [asdict(r) for r in rates],
            "set_disparities": [
                {k: v for k, v in asdict(d).items() if k != PERMUTED_FIELD} for d in disparities
            ],
        },
    )


def coded_answers(log: EvalLog) -> list[CodedAnswer]:
    """Turn every scored sample of ``log`` into a ``CodedAnswer``."""
    answers = []
    for sample in log.samples or []:
        score = (sample.scores or {}).get(SCORER_NAME)
        if score is None or not isinstance(score.value, dict):
            continue

        meta = sample.metadata
        score_meta = score.metadata or {}
        leak_options = score_meta.get("leak_options")
        answers.append(
            CodedAnswer(
                sample_id=str(sample.id),
                epoch=sample.epoch,
                set_info=SetInfo(
                    set_id=meta["set_id"],
                    template_id=meta["template_id"],
                    right_id=meta["right_id"],
                    category=meta["category"],
                    language=meta["language"],
                    framing=meta["framing"],
                ),
                group_id=meta["group_id"],
                judge_status=str(score.value.get(JUDGE_STATUS)),
                codes={code: score.value.get(code) for code in CODES},
                leak_correct=score_meta.get("leak_correct"),
                leak_options=len(leak_options) if leak_options else None,
                redacted_question=score_meta.get("redacted_question"),
                redacted_answer=score_meta.get("redacted_answer"),
            )
        )

    return answers


def not_comparable_reasons(log: EvalLog) -> list[str]:
    """Why ``log`` cannot be compared with runs of this benchmark version."""
    stored = log.eval.metadata or {}
    reasons = [
        f"{key} is {stored.get(key)!r}, expected {expected!r}"
        for key, expected in run_metadata().items()
        if stored.get(key) != expected
    ]
    other_judges = sorted(
        {
            str(score.metadata.get("judge_model"))
            for sample in log.samples or []
            if (score := (sample.scores or {}).get(SCORER_NAME)) is not None
            and score.metadata is not None
            and score.metadata.get("judge_is_default") is not True
        }
    )
    if other_judges:
        reasons.append(f"answers were coded by a non-default judge: {', '.join(other_judges)}")

    return reasons


def render_markdown(reports: Sequence[LogReport]) -> str:
    """Render all log reports as one Markdown document."""
    lines = ["# LLM human rights bias benchmark report", ""]
    for r in reports:
        lines += _summary_lines(r)
        lines += _headline_lines(r)
        lines += _metric_tables(r)
        lines += _optional_tables(r)

    return "\n".join(lines)


def _summary_lines(r: LogReport) -> list[str]:
    lines = [f"## {r.model} ({r.log_file})", ""]
    lines += [f"- Task: {r.task}, created {r.created}, {r.epochs} epochs, {r.answers} answers"]
    if r.comparable:
        return [*lines, "- Comparable: yes", ""]

    reasons = [f"  - {reason}" for reason in r.not_comparable_reasons]
    return [*lines, "- Comparable: no", *reasons, ""]


def _headline_lines(r: LogReport) -> list[str]:
    lines = ["### Headline", ""]
    if r.headline is None:
        return [*lines, "No set had enough coded answers to compare groups.", ""]

    h = r.headline
    return [
        *lines,
        f'Mean set disparity in "{HEADLINE_CODE} = {HEADLINE_VALUE}": {_rate(h.mean)} '
        f"({_LEVEL} CI {_interval(h)}), over {h.sets} sets from {h.templates} templates. "
        f"Expected by chance: {_optional_rate(h.chance_mean)} (p = {_optional_rate(h.p_value)}).",
        "",
    ]


def _metric_tables(r: LogReport) -> list[str]:
    lines = [f"### Breakdown ({HEADLINE_CODE} = {HEADLINE_VALUE})", ""]
    lines += _table(
        ["Dimension", "Level", *_ESTIMATE_COLUMNS],
        [
            [dim, lvl, *_estimate_cells(e)]
            for dim, levels in r.breakdowns.items()
            for lvl, e in levels.items()
        ],
    )
    lines += ["### Disparity by code", ""]
    lines += _table(
        ["Code = value", *_ESTIMATE_COLUMNS],
        [[name, *_estimate_cells(e)] for name, e in r.code_disparities.items()],
    )
    lines += [f"### Group gaps ({HEADLINE_CODE} = {HEADLINE_VALUE})", ""]
    lines += _table(
        ["Language", "Category", "Group", "Mean gap", "Sets"],
        [
            [g.language, g.category, g.group_id, GAP_FORMAT.format(g.mean_gap), str(g.sets)]
            for g in r.group_gaps
        ],
    )
    lines += ["### Judge failures", ""]
    lines += _table(
        ["Language", "Category", "Group", "Failed", "Total", "Rate"],
        [
            [f.language, f.category, f.group_id, str(f.failed), str(f.total), _rate(f.rate)]
            for f in r.failures
        ],
    )
    return lines


def _optional_tables(r: LogReport) -> list[str]:
    lines: list[str] = []
    if r.leaks:
        lines += ["### Leak check", ""]
        lines += _table(
            ["Language", "Category", "Correct", "Checked", "Rate", "Chance"],
            [
                [
                    k.language,
                    k.category,
                    str(k.correct),
                    str(k.checked),
                    _rate(k.rate),
                    _rate(k.chance),
                ]
                for k in r.leaks
            ],
        )

    if r.consistency is not None:
        c = r.consistency
        lines += ["### Judge consistency", ""]
        lines += [f"{c.compared} answers coded again, {c.judge_failures} judge failures.", ""]
        lines += _table(
            ["Code", "Agreement"], [[code, _rate(share)] for code, share in c.per_code.items()]
        )

    return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hrbias", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="write a Markdown and JSON report from logs")
    report.add_argument("log_dir", help="directory holding Inspect logs")
    report.add_argument("--out-dir", help="where to write the report (default: log_dir)")
    report.add_argument("--data-dir", help="benchmark data directory (default: data/)")
    report.add_argument(
        "--consistency-sample",
        type=int,
        default=0,
        help="code this many answers per log again and report agreement per code",
    )
    report.add_argument(
        "--consistency-grader",
        help="model for the consistency check (default: the pinned default judge)",
    )
    return parser


def _consistency_model(name: str | None) -> Model:
    config = GenerateConfig(temperature=JUDGE_TEMPERATURE)
    if name is None:
        return get_model(DEFAULT_JUDGE_MODEL, config=config, **DEFAULT_JUDGE_MODEL_ARGS)

    return get_model(name, config=config)


def _consistency(log: EvalLog, data: BenchmarkData, size: int, grader: Model) -> CodeAgreement:
    candidates = [
        RejudgeItem(
            redacted_question=a.redacted_question,
            redacted_answer=a.redacted_answer,
            right_description=data.rights[a.set_info.right_id].description[a.set_info.language],
            language=a.set_info.language,
            codes=a.codes,
        )
        for a in coded_answers(log)
        if a.ok and a.redacted_question is not None and a.redacted_answer is not None
    ]
    chosen = random.Random(CONSISTENCY_SEED).sample(candidates, min(size, len(candidates)))
    return asyncio.run(judge_consistency(grader, chosen))


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["None.", ""]

    lines = [f"| {' | '.join(header)} |", f"|{'|'.join('---' for _ in header)}|"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return [*lines, ""]


def _estimate_cells(estimate: Estimate) -> list[str]:
    return [
        _rate(estimate.mean),
        _interval(estimate),
        _optional_rate(estimate.chance_mean),
        _optional_rate(estimate.p_value),
        str(estimate.sets),
    ]


def _interval(estimate: Estimate) -> str:
    return f"{_rate(estimate.low)} to {_rate(estimate.high)}"


def _optional_rate(value: float | None) -> str:
    return MISSING_VALUE if value is None else _rate(value)


def _rate(value: float) -> str:
    return RATE_FORMAT.format(value)
