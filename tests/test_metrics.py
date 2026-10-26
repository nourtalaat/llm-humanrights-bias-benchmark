import random

import pytest

from hrbias.codebook import CODE_VALUES, sheet_keys, split_key
from hrbias.dataset import FAMILIES
from hrbias.metrics import (
    FAMILY_HEADLINES,
    CodedAnswer,
    Estimate,
    SetInfo,
    actor_gaps,
    bootstrap_interval,
    breakdown,
    case_gaps,
    case_rates,
    failure_rates,
    headline_disparities,
    leak_rates,
    mean_disparity,
    permuted_disparities,
    set_disparities,
)
from hrbias.scorer import STATUS_FAILED, STATUS_OK

STANCE = "t1.stance"
SET_A = SetInfo("t1.skeptical.p0.en", "t1", "detention", "mirrored", "set-a", "en", "skeptical")
SET_B = SetInfo("t2.skeptical.p0.ar", "t2", "detention", "mirrored", "set-a", "ar", "skeptical")
CASES = {"c1": ("v1", "p1"), "c2": ("v2", "p2"), "c3": ("v3", None)}
BASE_CODES = {
    "t1.refusal": "none",
    "t1.hedging": "none",
    "t1.both_sides": "no",
    "t1.justifies_detention": "no",
}


def _answer(
    set_info: SetInfo,
    case_id: str,
    stance: str | None,
    epoch: int = 1,
    leak_correct: bool | None = None,
) -> CodedAnswer:
    victim_id, perpetrator_id = CASES[case_id]
    status = STATUS_OK if stance else STATUS_FAILED
    codes = {**BASE_CODES, STANCE: stance} if stance else dict.fromkeys([*BASE_CODES, STANCE])
    return CodedAnswer(
        sample_id=f"{set_info.set_id}.{case_id}",
        epoch=epoch,
        set_info=set_info,
        case_id=case_id,
        victim_id=victim_id,
        perpetrator_id=perpetrator_id,
        judge_status=status,
        codes=codes,
        leak_correct=leak_correct,
        leak_options=3 if leak_correct is not None else None,
    )


ANSWERS = [
    _answer(SET_A, "c1", "affirms", 1),
    _answer(SET_A, "c1", "affirms", 2),
    _answer(SET_A, "c2", "affirms", 1),
    _answer(SET_A, "c2", "denies", 2),
    _answer(SET_A, "c3", "denies", 1),
    _answer(SET_A, "c3", None, 2),
    _answer(SET_B, "c1", "affirms", 1, leak_correct=True),
    _answer(SET_B, "c2", "affirms", 1, leak_correct=False),
    _answer(SET_B, "c3", "affirms", 1, leak_correct=False),
]


def test_family_headlines_cover_every_family_with_valid_keys() -> None:
    assert set(FAMILY_HEADLINES) == set(FAMILIES)
    for family, pairs in FAMILY_HEADLINES.items():
        for key, value in pairs:
            turn, code = split_key(key)
            assert key in sheet_keys(family, turn)
            assert value in CODE_VALUES[code]


@pytest.mark.parametrize(
    ("set_info", "case_id", "value", "expected_rate", "expected_answers"),
    [
        pytest.param(SET_A, "c1", "affirms", 1.0, 2, id="all-affirm"),
        pytest.param(SET_A, "c2", "affirms", 0.5, 2, id="half-affirm"),
        pytest.param(SET_A, "c2", "denies", 0.5, 2, id="half-deny"),
        pytest.param(SET_A, "c3", "affirms", 0.0, 1, id="failure-not-counted"),
        pytest.param(SET_B, "c3", "affirms", 1.0, 1, id="other-set"),
    ],
)
def test_case_rates(
    set_info: SetInfo,
    case_id: str,
    value: str,
    expected_rate: float,
    expected_answers: int,
) -> None:
    [rate] = [
        r
        for r in case_rates(ANSWERS)
        if r.set_info == set_info and r.case_id == case_id and r.code == STANCE
        if r.value == value
    ]

    assert rate.rate == expected_rate
    assert rate.answers == expected_answers
    assert (rate.victim_id, rate.perpetrator_id) == CASES[case_id]


def test_case_rates_cover_every_coded_key_and_value() -> None:
    rates = [r for r in case_rates(ANSWERS) if r.set_info == SET_A and r.case_id == "c1"]

    assert {(r.code, r.value) for r in rates} == {
        (key, value) for key in [*BASE_CODES, STANCE] for value in CODE_VALUES[split_key(key)[1]]
    }


def test_missing_turn_counts_only_where_present() -> None:
    longer = {**BASE_CODES, STANCE: "affirms", "t2.position_change": "holds"}
    answers = [
        _answer(SET_A, "c1", "affirms"),
        CodedAnswer("x", 2, SET_A, "c1", "v1", "p1", STATUS_OK, longer),
        CodedAnswer(
            "y", 1, SET_A, "c2", "v2", "p2", STATUS_OK, {**longer, "t2.position_change": "reverses"}
        ),
    ]

    rates = {(r.case_id, r.value): r for r in case_rates(answers) if r.code == "t2.position_change"}

    assert rates[("c1", "holds")].rate == 1.0
    assert rates[("c1", "holds")].answers == 1
    [disparity] = [
        d
        for d in set_disparities(case_rates(answers), permuted_disparities(answers, 10))
        if d.code == "t2.position_change" and d.value == "holds"
    ]
    assert disparity.disparity == 1.0
    assert len(disparity.permuted) == 10


@pytest.mark.parametrize(
    ("case_id", "expected_gap"),
    [
        pytest.param("c1", 0.5, id="above-mean"),
        pytest.param("c2", 0.0, id="at-mean"),
        pytest.param("c3", -0.5, id="below-mean"),
    ],
)
def test_case_gaps(case_id: str, expected_gap: float) -> None:
    [gap] = [
        g
        for g in case_gaps(case_rates(ANSWERS))
        if g.set_info == SET_A and g.case_id == case_id
        if g.code == STANCE and g.value == "affirms"
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
        for d in set_disparities(case_rates(ANSWERS))
        if d.set_info == set_info and d.code == STANCE and d.value == value
    ]

    assert disparity.disparity == expected
    assert disparity.cases == 3


def test_set_with_one_coded_case_is_left_out() -> None:
    answers = [_answer(SET_A, "c1", "affirms"), _answer(SET_A, "c2", None)]

    assert set_disparities(case_rates(answers)) == []
    assert mean_disparity([], STANCE, "affirms") is None


def test_headline_mean_and_interval_contains_it() -> None:
    estimate = mean_disparity(set_disparities(case_rates(ANSWERS)), STANCE, "affirms")

    assert estimate is not None
    assert estimate.mean == pytest.approx(0.5)
    assert estimate.sets == 2
    assert estimate.templates == 2
    assert estimate.low <= estimate.mean <= estimate.high
    assert estimate.low >= 0.0 and estimate.high <= 1.0


def test_headline_disparities_use_the_family_headline() -> None:
    disparities = set_disparities(case_rates(ANSWERS))

    [(code, value, estimate)] = headline_disparities(disparities, "detention")

    assert (code, value) == (STANCE, "affirms")
    assert estimate.mean == pytest.approx(0.5)
    assert headline_disparities(disparities, "war_origin") == []


def test_breakdown_by_language() -> None:
    disparities = set_disparities(case_rates(ANSWERS))

    by_language = breakdown(disparities, "language", STANCE, "affirms")

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


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        pytest.param(
            "victim",
            {("en", "v1"): 0.5, ("en", "v3"): -0.5, ("ar", "v1"): 0.0, ("ar", "v3"): 0.0},
            id="victim",
        ),
        pytest.param(
            "perpetrator",
            {("en", "p1"): 0.5, ("en", "p2"): 0.0, ("ar", "p1"): 0.0, ("ar", "p2"): 0.0},
            id="perpetrator-skips-cases-without-one",
        ),
    ],
)
def test_actor_gaps(role: str, expected: dict[tuple[str, str], float]) -> None:
    gaps = actor_gaps(case_gaps(case_rates(ANSWERS)), role, STANCE, "affirms")
    by_key = {(g.language, g.actor_id): g.mean_gap for g in gaps}

    for key, gap in expected.items():
        assert by_key[key] == pytest.approx(gap)
    assert all(g.role == role and g.family == "detention" for g in gaps)
    if role == "perpetrator":
        assert len(by_key) == 4


def test_failure_rates() -> None:
    rates = {(f.language, f.case_id): f for f in failure_rates(ANSWERS)}

    assert rates[("en", "c3")].failed == 1
    assert rates[("en", "c3")].rate == 0.5
    assert rates[("en", "c1")].rate == 0.0
    assert rates[("en", "c1")].tier == "mirrored"


def test_leak_rates() -> None:
    [leak] = leak_rates(ANSWERS)

    assert (leak.tier, leak.language, leak.correct, leak.checked) == ("mirrored", "ar", 1, 3)
    assert leak.rate == pytest.approx(1 / 3)
    assert leak.chance == pytest.approx(1 / 3)


TEST_PERMUTATIONS = 300
NULL_SETS = 10
ANSWERS_PER_CASE = 10


def _random_answers(seed: int, planted_case: str | None) -> list[CodedAnswer]:
    rng = random.Random(seed)
    answers = []
    for index in range(NULL_SETS):
        set_info = SetInfo(
            f"t{index}.skeptical.p0.en",
            f"t{index}",
            "detention",
            "mirrored",
            "s",
            "en",
            "skeptical",
        )
        for case_id in CASES:
            for epoch in range(ANSWERS_PER_CASE):
                affirms = case_id == planted_case or rng.random() < 0.5
                stance = "affirms" if affirms else "denies"
                answers.append(_answer(set_info, case_id, stance, epoch))

    return answers


def _headline_with_chance(answers: list[CodedAnswer]) -> Estimate:
    permuted = permuted_disparities(answers, permutations=TEST_PERMUTATIONS)
    estimate = mean_disparity(set_disparities(case_rates(answers), permuted), STANCE, "affirms")
    assert estimate is not None
    assert estimate.chance_mean is not None
    assert estimate.p_value is not None
    return estimate


def test_no_case_effect_matches_chance() -> None:
    estimate = _headline_with_chance(_random_answers(seed=1, planted_case=None))

    assert estimate.mean > 0.1
    assert estimate.mean == pytest.approx(estimate.chance_mean, abs=0.1)
    assert estimate.p_value is not None
    assert estimate.p_value > 0.05


def test_planted_case_effect_has_small_p_value() -> None:
    estimate = _headline_with_chance(_random_answers(seed=1, planted_case="c1"))

    assert estimate.chance_mean is not None
    assert estimate.mean > estimate.chance_mean + 0.2
    assert estimate.p_value is not None
    assert estimate.p_value < 0.01


def test_permutations_are_reproducible_and_keep_case_sizes() -> None:
    answers = _random_answers(seed=2, planted_case=None)

    first = permuted_disparities(answers, permutations=20)
    second = permuted_disparities(answers, permutations=20)

    assert first == second
    assert all(len(values) == 20 for values in first.values())


def test_without_permutations_chance_is_missing() -> None:
    estimate = mean_disparity(set_disparities(case_rates(ANSWERS)), STANCE, "affirms")

    assert estimate is not None
    assert estimate.chance_mean is None
    assert estimate.p_value is None
