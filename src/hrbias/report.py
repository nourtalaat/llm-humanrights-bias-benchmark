"""``hrbias report``: turn Inspect logs into a Markdown and a JSON report.

Each log (one model run) gets its own section, and within it each tier
(``real_event`` and ``mirrored``) is reported separately. A log is marked not
comparable when its benchmark version, codebook version or default judge pin
differs from this code's, or when any conversation was coded by a judge other
than the default.
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

from hrbias.dataset import (
    DEFAULT_DATA_DIR,
    FAMILIES,
    PERPETRATOR_ROLE,
    TIERS,
    VICTIM_ROLE,
    BenchmarkData,
    load_benchmark_data,
)
from hrbias.judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_MODEL_ARGS,
    CodeAgreement,
    RejudgeItem,
    Turn,
    judge_consistency,
)
from hrbias.metrics import (
    BREAKDOWN_DIMENSIONS,
    CONFIDENCE_LEVEL,
    ActorGap,
    CaseGap,
    CodedAnswer,
    Estimate,
    FailureRate,
    LeakRate,
    SetDisparity,
    SetInfo,
    actor_gaps,
    breakdown,
    case_gaps,
    case_rates,
    failure_rates,
    headline_disparities,
    key_values,
    leak_rates,
    mean_disparity,
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
class CodeEstimate:
    """The mean disparity of one family's ``code == value``."""

    family: str
    code: str
    value: str
    estimate: Estimate


@dataclass(frozen=True)
class BreakdownEstimate:
    """A family's headline disparity within one level of a dimension."""

    family: str
    code: str
    value: str
    dimension: str
    level: str
    estimate: Estimate


@dataclass(frozen=True)
class TierReport:
    """Everything the report says about one tier of one log."""

    tier: str
    answers: int
    headlines: list[CodeEstimate]
    breakdowns: list[BreakdownEstimate]
    code_disparities: list[CodeEstimate]
    victim_gaps: list[ActorGap]
    perpetrator_gaps: list[ActorGap]


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
    tiers: list[TierReport]
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
    rates = case_rates(answers)
    gaps = case_gaps(rates)
    disparities = set_disparities(rates, permuted_disparities(answers))
    reasons = not_comparable_reasons(log)
    tiers = [
        tier_report(tier, [a for a in answers if a.set_info.tier == tier], disparities, gaps)
        for tier in TIERS
        if any(a.set_info.tier == tier for a in answers)
    ]
    consistency = None
    if grader is not None and consistency_sample > 0:
        consistency = _consistency(answers, data, consistency_sample, grader)

    return LogReport(
        log_file=Path(log.location).name,
        task=log.eval.task,
        model=log.eval.model,
        created=log.eval.created,
        epochs=log.eval.config.epochs or 1,
        answers=len(answers),
        comparable=not reasons,
        not_comparable_reasons=reasons,
        tiers=tiers,
        failures=failure_rates(answers),
        leaks=leak_rates(answers),
        consistency=consistency,
        details={
            "case_rates": [asdict(r) for r in rates],
            "set_disparities": [
                {k: v for k, v in asdict(d).items() if k != PERMUTED_FIELD} for d in disparities
            ],
        },
    )


def tier_report(
    tier: str,
    answers: Sequence[CodedAnswer],
    disparities: Sequence[SetDisparity],
    gaps: Sequence[CaseGap],
) -> TierReport:
    """Headlines, breakdowns, every code's disparity and actor gaps for one tier."""
    in_tier = [d for d in disparities if d.set_info.tier == tier]
    tier_gaps = [g for g in gaps if g.set_info.tier == tier]
    headlines: list[CodeEstimate] = []
    breakdowns: list[BreakdownEstimate] = []
    code_estimates: list[CodeEstimate] = []
    victim: list[ActorGap] = []
    perpetrator: list[ActorGap] = []
    for family in FAMILIES:
        selected = [d for d in in_tier if d.set_info.family == family]
        for code, value, estimate in headline_disparities(selected, family):
            headlines.append(CodeEstimate(family, code, value, estimate))
            breakdowns.extend(
                BreakdownEstimate(family, code, value, dimension, level, level_estimate)
                for dimension in BREAKDOWN_DIMENSIONS
                for level, level_estimate in breakdown(selected, dimension, code, value).items()
            )
            family_gaps = [g for g in tier_gaps if g.set_info.family == family]
            victim.extend(actor_gaps(family_gaps, VICTIM_ROLE, code, value))
            perpetrator.extend(actor_gaps(family_gaps, PERPETRATOR_ROLE, code, value))

        code_estimates.extend(_code_estimates(family, selected))

    return TierReport(
        tier=tier,
        answers=len(answers),
        headlines=headlines,
        breakdowns=breakdowns,
        code_disparities=code_estimates,
        victim_gaps=victim,
        perpetrator_gaps=perpetrator,
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
        transcript = score_meta.get("redacted_transcript")
        answers.append(
            CodedAnswer(
                sample_id=str(sample.id),
                epoch=sample.epoch,
                set_info=SetInfo(
                    set_id=meta["set_id"],
                    template_id=meta["template_id"],
                    family=meta["family"],
                    tier=meta["tier"],
                    case_set=meta["case_set"],
                    language=meta["language"],
                    framing=meta["framing"],
                ),
                case_id=meta["case_id"],
                victim_id=meta["victim_id"],
                perpetrator_id=meta["perpetrator_id"],
                judge_status=str(score.value.get(JUDGE_STATUS)),
                codes={k: v for k, v in score.value.items() if k != JUDGE_STATUS},
                leak_correct=score_meta.get("leak_correct"),
                leak_options=len(leak_options) if leak_options else None,
                right_ids=tuple(meta["right_ids"]),
                transcript=tuple(Turn(**t) for t in transcript) if transcript else None,
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
        for tier in r.tiers:
            lines += _tier_lines(tier)

        lines += _judge_tables(r)

    return "\n".join(lines)


def _code_estimates(family: str, disparities: Sequence[SetDisparity]) -> list[CodeEstimate]:
    estimates = []
    for code in dict.fromkeys(d.code for d in disparities):
        for value in key_values(code):
            estimate = mean_disparity(disparities, code, value)
            if estimate is not None:
                estimates.append(CodeEstimate(family, code, value, estimate))

    return estimates


def _summary_lines(r: LogReport) -> list[str]:
    lines = [f"## {r.model} ({r.log_file})", ""]
    lines += [f"- Task: {r.task}, created {r.created}, {r.epochs} epochs, {r.answers} answers"]
    if r.comparable:
        return [*lines, "- Comparable: yes", ""]

    reasons = [f"  - {reason}" for reason in r.not_comparable_reasons]
    return [*lines, "- Comparable: no", *reasons, ""]


def _tier_lines(t: TierReport) -> list[str]:
    lines = [f"### Tier: {t.tier} ({t.answers} answers)", "", "#### Headline by family", ""]
    if not t.headlines:
        lines += ["No set had enough coded answers to compare cases.", ""]
    else:
        lines += _table(
            ["Family", "Code = value", *_ESTIMATE_COLUMNS],
            [[h.family, _code(h.code, h.value), *_estimate_cells(h.estimate)] for h in t.headlines],
        )

    lines += ["#### Headline breakdown", ""]
    lines += _table(
        ["Family", "Code = value", "Dimension", "Level", *_ESTIMATE_COLUMNS],
        [
            [b.family, _code(b.code, b.value), b.dimension, b.level, *_estimate_cells(b.estimate)]
            for b in t.breakdowns
        ],
    )
    lines += ["#### Disparity by code", ""]
    lines += _table(
        ["Family", "Code = value", *_ESTIMATE_COLUMNS],
        [
            [c.family, _code(c.code, c.value), *_estimate_cells(c.estimate)]
            for c in t.code_disparities
        ],
    )
    lines += ["#### Victim gaps (headline codes)", ""]
    lines += _gap_table("Victim", t.victim_gaps, t.headlines)
    lines += ["#### Perpetrator gaps (headline codes)", ""]
    lines += _gap_table("Perpetrator", t.perpetrator_gaps, t.headlines)
    return lines


def _gap_table(
    label: str, gaps: Sequence[ActorGap], headlines: Sequence[CodeEstimate]
) -> list[str]:
    rows = []
    for headline in headlines:
        code = _code(headline.code, headline.value)
        rows.extend(
            [g.language, g.family, code, g.actor_id, GAP_FORMAT.format(g.mean_gap), str(g.sets)]
            for g in gaps
            if (g.family, g.code, g.value) == (headline.family, headline.code, headline.value)
        )

    return _table(["Language", "Family", "Code = value", label, "Mean gap", "Sets"], rows)


def _judge_tables(r: LogReport) -> list[str]:
    lines = ["### Judge failures", ""]
    lines += _table(
        ["Tier", "Language", "Case set", "Case", "Failed", "Total", "Rate"],
        [
            [f.tier, f.language, f.case_set, f.case_id, str(f.failed), str(f.total), _rate(f.rate)]
            for f in r.failures
        ],
    )
    if r.leaks:
        lines += ["### Leak check", ""]
        lines += _table(
            ["Tier", "Language", "Case set", "Correct", "Checked", "Rate", "Chance"],
            [
                [
                    k.tier,
                    k.language,
                    k.case_set,
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
        help="code this many conversations per log again and report agreement per code",
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


def _consistency(
    answers: Sequence[CodedAnswer], data: BenchmarkData, size: int, grader: Model
) -> CodeAgreement:
    candidates = [
        RejudgeItem(
            transcript=a.transcript,
            right_descriptions=tuple(
                data.rights[r].description[a.set_info.language] for r in a.right_ids
            ),
            family=a.set_info.family,
            language=a.set_info.language,
            codes=a.codes,
        )
        for a in answers
        if a.ok and a.transcript is not None
    ]
    chosen = random.Random(CONSISTENCY_SEED).sample(candidates, min(size, len(candidates)))
    return asyncio.run(judge_consistency(grader, chosen))


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["None.", ""]

    lines = [f"| {' | '.join(header)} |", f"|{'|'.join('---' for _ in header)}|"]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return [*lines, ""]


def _code(code: str, value: str) -> str:
    return f"{code} = {value}"


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
