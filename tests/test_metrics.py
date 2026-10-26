import random

import pytest
from support import VALID_CODES

from hrbias.codebook import CODES
from hrbias.metrics import (
    CodedAnswer,
    Estimate,
    SetInfo,
    bootstrap_interval,
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
from hrbias.scorer import STATUS_FAILED, STATUS_OK

SET_A = SetInfo("t1.neutral.p0.en", "t1", "right-a", "cat", "en", "neutral")
SET_B = SetInfo("t2.neutral.p0.ar", "t2", "right-b", "cat", "ar", "neutral")


def _answer(
    set_info: SetInfo,
    group_id: str,
    stance: str | None,
    epoch: int = 1,
    leak_correct: bool | None = None,
) -> CodedAnswer:
    status = STATUS_OK if stance else STATUS_FAILED
    codes = {**VALID_CODES, "stance": stance} if stance else dict.fromkeys(CODES)
    return CodedAnswer(
        sample_id=f"{set_info.set_id}.{group_id}",
        epoch=epoch,
        set_info=set_info,
        group_id=group_id,
        judge_status=status,
        codes=codes,
        leak_correct=leak_correct,
        leak_options=3 if leak_correct is not None else None,
    )


ANSWERS = [
    _answer(SET_A, "g1", "affirms", 1),
    _answer(SET_A, "g1", "affirms", 2),
    _answer(SET_A, "g2", "affirms", 1),
    _answer(SET_A, "g2", "denies", 2),
    _answer(SET_A, "g3", "denies", 1),
    _answer(SET_A, "g3", None, 2),
    _answer(SET_B, "g1", "affirms", 1, leak_correct=True),
    _answer(SET_B, "g2", "affirms", 1, leak_correct=False),
    _answer(SET_B, "g3", "affirms", 1, leak_correct=False),
]


@pytest.mark.parametrize(
    ("set_info", "group_id", "value", "expected_rate", "expected_answers"),
    [
        pytest.param(SET_A, "g1", "affirms", 1.0, 2, id="all-affirm"),
        pytest.param(SET_A, "g2", "affirms", 0.5, 2, id="half-affirm"),
        pytest.param(SET_A, "g2", "denies", 0.5, 2, id="half-deny"),
        pytest.param(SET_A, "g3", "affirms", 0.0, 1, id="failure-not-counted"),
        pytest.param(SET_B, "g3", "affirms", 1.0, 1, id="other-set"),
    ],
)
def test_group_rates(
    set_info: SetInfo,
    group_id: str,
    value: str,
    expected_rate: float,
    expected_answers: int,
) -> None:
    [rate] = [
        r
        for r in group_rates(ANSWERS)
        if r.set_info == set_info and r.group_id == group_id and r.code == "stance"
        if r.value == value
    ]

    assert rate.rate == expected_rate
    assert rate.answers == expected_answers


@pytest.mark.parametrize(
    ("group_id", "expected_gap"),
    [
        pytest.param("g1", 0.5, id="above-mean"),
        pytest.param("g2", 0.0, id="at-mean"),
        pytest.param("g3", -0.5, id="below-mean"),
    ],
)
def test_group_gaps(group_id: str, expected_gap: float) -> None:
    [gap] = [
        g
        for g in group_gaps(group_rates(ANSWERS))
        if g.set_info == SET_A and g.group_id == group_id
        if g.code == "stance" and g.value == "affirms"
    ]

    assert gap.gap == pytest.approx(expected_gap)


@pytest.mark.parametrize(
    ("set_info", "value", "expected"),
    [
        pytest.param(SET_A, "affirms", 1.0, id="set-a-affirms"),
        pytest.param(SET_A, "denies", 1.0, id="set-a-denies"),
        pytest.param(SET_B, "affirms", 0.0, id="set-b-consistent"),
        pytest.param(SET_A, "conditional", 0.0, id="value-never-chosen"),
    ],
)
def test_set_disparities(set_info: SetInfo, value: str, expected: float) -> None:
    [disparity] = [
        d
        for d in set_disparities(group_rates(ANSWERS))
        if d.set_info == set_info and d.code == "stance" and d.value == value
    ]

    assert disparity.disparity == expected
    assert disparity.groups == 3


def test_set_with_one_coded_group_is_left_out() -> None:
    answers = [_answer(SET_A, "g1", "affirms"), _answer(SET_A, "g2", None)]

    assert set_disparities(group_rates(answers)) == []
    assert mean_disparity([]) is None


def test_headline_mean_and_interval_contains_it() -> None:
    estimate = mean_disparity(set_disparities(group_rates(ANSWERS)))

    assert estimate is not None
    assert estimate.mean == pytest.approx(0.5)
    assert estimate.sets == 2
    assert estimate.templates == 2
    assert estimate.low <= estimate.mean <= estimate.high
    assert estimate.low >= 0.0 and estimate.high <= 1.0


def test_breakdown_by_language() -> None:
    by_language = breakdown(set_disparities(group_rates(ANSWERS)), "language")

    assert set(by_language) == {"ar", "en"}
    assert by_language["en"].mean == 1.0
    assert by_language["ar"].mean == 0.0


@pytest.mark.parametrize(
    ("units", "expected"),
    [
        pytest.param([[0.4], [0.4], [0.4]], (0.4, 0.4), id="constant"),
        pytest.param([[0.0], [1.0]], (0.0, 1.0), id="two-units"),
    ],
)
def test_bootstrap_interval(units: list[list[float]], expected: tuple[float, float]) -> None:
    assert bootstrap_interval(units) == pytest.approx(expected)
    assert bootstrap_interval(units) == bootstrap_interval(units)


def test_mean_group_gaps() -> None:
    by_key = {
        (g.language, g.group_id): g.mean_gap
        for g in mean_group_gaps(group_gaps(group_rates(ANSWERS)))
    }

    assert by_key[("en", "g1")] == pytest.approx(0.5)
    assert by_key[("en", "g3")] == pytest.approx(-0.5)
    assert by_key[("ar", "g1")] == pytest.approx(0.0)


def test_failure_rates() -> None:
    rates = {(f.language, f.group_id): f for f in failure_rates(ANSWERS)}

    assert rates[("en", "g3")].failed == 1
    assert rates[("en", "g3")].rate == 0.5
    assert rates[("en", "g1")].rate == 0.0


def test_leak_rates() -> None:
    [leak] = leak_rates(ANSWERS)

    assert (leak.language, leak.correct, leak.checked) == ("ar", 1, 3)
    assert leak.rate == pytest.approx(1 / 3)
    assert leak.chance == pytest.approx(1 / 3)


TEST_PERMUTATIONS = 300
NULL_SETS = 10
GROUPS = ("g1", "g2", "g3")
ANSWERS_PER_GROUP = 10


def _random_answers(seed: int, planted_group: str | None) -> list[CodedAnswer]:
    rng = random.Random(seed)
    answers = []
    for index in range(NULL_SETS):
        set_info = SetInfo(f"t{index}.neutral.p0.en", f"t{index}", "right", "cat", "en", "neutral")
        for group_id in GROUPS:
            for epoch in range(ANSWERS_PER_GROUP):
                affirms = group_id == planted_group or rng.random() < 0.5
                stance = "affirms" if affirms else "denies"
                answers.append(_answer(set_info, group_id, stance, epoch))

    return answers


def _headline_with_chance(answers: list[CodedAnswer]) -> Estimate:
    permuted = permuted_disparities(answers, permutations=TEST_PERMUTATIONS)
    estimate = mean_disparity(set_disparities(group_rates(answers), permuted))
    assert estimate is not None
    assert estimate.chance_mean is not None
    assert estimate.p_value is not None
    return estimate


def test_no_group_effect_matches_chance() -> None:
    estimate = _headline_with_chance(_random_answers(seed=1, planted_group=None))

    assert estimate.mean > 0.1
    assert estimate.mean == pytest.approx(estimate.chance_mean, abs=0.1)
    assert estimate.p_value > 0.05


def test_planted_group_effect_has_small_p_value() -> None:
    estimate = _headline_with_chance(_random_answers(seed=1, planted_group="g1"))

    assert estimate.chance_mean is not None
    assert estimate.mean > estimate.chance_mean + 0.2
    assert estimate.p_value is not None
    assert estimate.p_value < 0.01


def test_permutations_are_reproducible_and_keep_group_sizes() -> None:
    answers = _random_answers(seed=2, planted_group=None)

    first = permuted_disparities(answers, permutations=20)
    second = permuted_disparities(answers, permutations=20)

    assert first == second
    assert all(len(values) == 20 for values in first.values())


def test_without_permutations_chance_is_missing() -> None:
    estimate = mean_disparity(set_disparities(group_rates(ANSWERS)))

    assert estimate is not None
    assert estimate.chance_mean is None
    assert estimate.p_value is None
