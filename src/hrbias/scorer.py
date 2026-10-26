"""Inspect scorer: redact question and answer, have the blind judge code it, store codes.

``Score.value`` is a dict with ``judge_status`` (``ok`` or ``failed``) and one
entry per codebook code (``None`` when the judge failed). Bias metrics compare
answers across samples, so they are computed from the logs by ``metrics.py``,
not here.
"""

from pathlib import Path
from typing import Any

from inspect_ai.model import GenerateConfig, Model, get_model, model_roles
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState

from hrbias.codebook import CODES
from hrbias.dataset import (
    DEFAULT_DATA_DIR,
    Actor,
    BenchmarkData,
    Case,
    Language,
    load_benchmark_data,
)
from hrbias.judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_MODEL_ARGS,
    guess_case,
    judge_answer,
)
from hrbias.redact import PERPETRATOR_TOKEN, VICTIM_TOKEN, redact, role_targets

JUDGE_ROLE = "grader"
LEAK_VICTIM_FORM = "people"
LEAK_PERPETRATOR_FORM = "authority"
LEAK_OPTION_SEPARATOR = ", "
JUDGE_TEMPERATURE = 0.0

JUDGE_STATUS = "judge_status"
STATUS_OK = "ok"
STATUS_FAILED = "failed"
GRADER_FAILED_REASON = "grader_failed"


@metric
def judge_success_rate() -> Metric:
    """Share of answers the judge coded successfully."""

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0

        ok = sum(1 for s in scores if _status(s.score) == STATUS_OK)
        return ok / len(scores)

    return compute


@scorer(metrics=[judge_success_rate()])
def hrbias_scorer(data_dir: str | None = None, leak_check: bool = False) -> Scorer:
    """Score each answer by redacting it and its question, then asking the blind judge.

    The judge is the model bound to the ``grader`` role, or the pinned default
    judge when none is bound. With ``leak_check`` the judge is also asked which
    case of the set the redacted conversation is about.
    """
    data = load_benchmark_data(Path(data_dir) if data_dir else DEFAULT_DATA_DIR)

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        language: Language = meta["language"]
        targets = role_targets(
            meta["victim_terms"], meta["perpetrator_terms"], meta["context_terms"]
        )
        question = redact(state.input_text, targets, language)
        redaction = redact(state.output.completion, targets, language)
        judge, judge_is_default = _judge_model()
        right = "\n".join(data.rights[r].description[language] for r in meta["right_ids"])
        result = await judge_answer(judge, question.text, redaction.text, right, language)

        metadata: dict[str, Any] = {
            "redacted_question": question.text,
            "redacted_answer": redaction.text,
            "replacements": redaction.replacements,
            "judge_model": str(judge),
            "judge_is_default": judge_is_default,
            "judge_replies": result.replies,
            "judge_errors": result.errors,
        }
        if leak_check:
            conversation = f"{question.text}\n\n{redaction.text}"
            metadata.update(await _leak_check(judge, conversation, data, meta))

        if result.sheet is None:
            return Score(
                value={JUDGE_STATUS: STATUS_FAILED, **dict.fromkeys(CODES)},
                reason=GRADER_FAILED_REASON,
                explanation="; ".join(result.errors),
                metadata=metadata,
            )

        return Score(value={JUDGE_STATUS: STATUS_OK, **result.sheet.codes()}, metadata=metadata)

    return score


def _judge_model() -> tuple[Model, bool]:
    is_default = JUDGE_ROLE not in model_roles()
    model = get_model(
        role=JUDGE_ROLE,
        default=DEFAULT_JUDGE_MODEL,
        config=GenerateConfig(temperature=JUDGE_TEMPERATURE),
        **DEFAULT_JUDGE_MODEL_ARGS,
    )
    return model, is_default


async def _leak_check(
    judge: Model, redacted: str, data: BenchmarkData, meta: dict[str, Any]
) -> dict[str, Any]:
    language: Language = meta["language"]
    actors = data.actors[language]
    cases = data.case_sets[language][meta["case_set"]].cases
    options = {case.id: _case_label(case, actors) for case in cases}
    leak = await guess_case(judge, redacted, options, language)
    return {
        "leak_options": list(options),
        "leak_guess": leak.guess,
        "leak_correct": leak.guess == meta["case_id"],
        "leak_reply": leak.reply,
    }


def _case_label(case: Case, actors: dict[str, Actor]) -> str:
    parts = []
    if case.perpetrator is not None:
        name = _actor_name(actors[case.perpetrator], LEAK_PERPETRATOR_FORM)
        parts.append(f"{PERPETRATOR_TOKEN} = {name}")

    parts.append(f"{VICTIM_TOKEN} = {_actor_name(actors[case.victim], LEAK_VICTIM_FORM)}")
    return LEAK_OPTION_SEPARATOR.join(parts)


def _actor_name(actor: Actor, preferred_form: str) -> str:
    form = actor.forms.get(preferred_form) or next(iter(actor.forms.values()))
    return form.text


def _status(score: Score | None) -> str | None:
    if score is None or not isinstance(score.value, dict):
        return None

    status = score.value.get(JUDGE_STATUS)
    return status if isinstance(status, str) else None
