import re
from pathlib import Path

import pytest
from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput
from inspect_ai.scorer import Score
from inspect_ai.tool import ToolChoice, ToolInfo
from support import MOCK_MODEL, VALID_CODES, VALID_SHEET_JSON, mock_output, run_scored_eval

from hrbias.codebook import CODES
from hrbias.dataset import hrbias_dataset, load_benchmark_data
from hrbias.redact import GROUP_TOKEN
from hrbias.scorer import STATUS_FAILED, STATUS_OK

SCORER_NAME = "hrbias_scorer"
LEAK_PROMPT_MARKERS = ("Reply with only the number", "أجب برقم اختيارك فقط", "השב במספר")


def test_judge_never_sees_a_group_term(tmp_path: Path) -> None:
    judge_prompts: list[str] = []

    def judge(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        judge_prompts.append(messages[-1].text)
        return mock_output(VALID_SHEET_JSON)

    log = run_scored_eval(tmp_path, judge)

    assert judge_prompts
    terms = _all_group_terms()
    for prompt in judge_prompts:
        question = prompt.split("<question>\n", 1)[1].split("\n</question>", 1)[0]
        assert GROUP_TOKEN in question
        for term in terms:
            assert not _contains(prompt, term), (term, prompt)

    for sample in _samples(log):
        score = _score(sample)
        assert score.value == {"judge_status": STATUS_OK, **VALID_CODES}
        assert score.metadata is not None
        assert score.metadata["replacements"] >= 1
        assert GROUP_TOKEN in score.metadata["redacted_question"]
        assert score.metadata["judge_model"] == MOCK_MODEL
        assert score.metadata["judge_is_default"] is False
        assert score.metadata["judge_errors"] == []


def test_judge_failure_is_recorded_not_dropped(tmp_path: Path) -> None:
    def judge(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        return mock_output("I would rather not.")

    log = run_scored_eval(tmp_path, judge)

    samples = _samples(log)
    assert len(samples) == len(hrbias_dataset())
    for sample in samples:
        score = _score(sample)
        assert score.value == {"judge_status": STATUS_FAILED, **dict.fromkeys(CODES)}
        assert score.reason == "grader_failed"
        assert score.metadata is not None
        assert len(score.metadata["judge_errors"]) == 2


@pytest.mark.parametrize(
    ("guess_reply", "expect_correct_for_first_group"),
    [pytest.param("1", True, id="guesses-first"), pytest.param("none", False, id="no-guess")],
)
def test_leak_check(tmp_path: Path, guess_reply: str, expect_correct_for_first_group: bool) -> None:
    def judge(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        text = messages[-1].text
        is_leak_prompt = any(marker in text for marker in LEAK_PROMPT_MARKERS)
        return mock_output(guess_reply if is_leak_prompt else VALID_SHEET_JSON)

    log = run_scored_eval(tmp_path, judge, leak_check=True)

    for sample in _samples(log):
        meta = _score(sample).metadata
        assert meta is not None
        assert meta["leak_options"][0] in meta["leak_options"]
        is_first = sample.metadata["group_id"] == meta["leak_options"][0]
        expected_correct = expect_correct_for_first_group and is_first
        assert meta["leak_correct"] is expected_correct
        assert meta["leak_reply"] == guess_reply


def _samples(log: EvalLog) -> list[EvalSample]:
    assert log.samples is not None
    return log.samples


def _score(sample: EvalSample) -> Score:
    assert sample.scores is not None
    return sample.scores[SCORER_NAME]


def _all_group_terms() -> set[str]:
    data = load_benchmark_data()
    return {
        term
        for groups in data.groups.values()
        for group in groups
        for term in [*group.group_terms(), *group.related_terms]
    }


def _contains(text: str, term: str) -> bool:
    if term.isascii():
        return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) is not None

    return term in text
