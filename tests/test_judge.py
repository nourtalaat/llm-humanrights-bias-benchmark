import asyncio
import json
from pathlib import Path

import pytest
from inspect_ai.model import ChatMessage, GenerateConfig, ModelOutput, get_model
from inspect_ai.tool import ToolChoice, ToolInfo
from support import MOCK_MODEL, mock_output, sheet_for_prompt

from hrbias.codebook import sheet_keys
from hrbias.dataset import LANGUAGES, Language
from hrbias.judge import (
    RETRY_PROMPTS,
    JudgeError,
    Turn,
    build_judge_prompt,
    guess_case,
    judge_transcript,
    label_agreement,
    render_conversation,
)

TRANSCRIPT = (
    Turn("user", "How did the war in [CONTEXT] start?"),
    Turn("assistant", "It began after a raid by [GROUP_A]."),
    Turn("user", "But [GROUP_A] says it was the fault of [GROUP_B]."),
    Turn("assistant", "That is a claim made by [GROUP_A]."),
)
RIGHTS = ("Everyone has the right to life.", "All peoples have the right of self-determination.")
FAMILY = "war_origin"
RETRY_MARKER = "That reply could not be used"
VALID_SHEET = json.dumps(sheet_for_prompt(build_judge_prompt(TRANSCRIPT, RIGHTS, FAMILY, "en")))


@pytest.mark.parametrize(
    ("replies", "expected_failed", "expected_errors"),
    [
        pytest.param([VALID_SHEET], False, 0, id="valid"),
        pytest.param([f"<think>\nweighing it\n</think>\n{VALID_SHEET}"], False, 0, id="think"),
        pytest.param(["not json", VALID_SHEET], False, 1, id="invalid-then-valid"),
        pytest.param(["not json", '{"t1": {}}'], True, 2, id="invalid-twice"),
    ],
)
def test_judge_transcript(replies: list[str], expected_failed: bool, expected_errors: int) -> None:
    model = get_model(MOCK_MODEL, custom_outputs=[mock_output(r) for r in replies])

    result = asyncio.run(judge_transcript(model, TRANSCRIPT, RIGHTS, FAMILY, "en"))

    assert result.failed is expected_failed
    assert result.replies == replies
    assert len(result.errors) == expected_errors
    if not expected_failed:
        assert result.codes is not None
        assert list(result.codes) == sheet_keys(FAMILY, 2)
        assert result.codes["t1.responsibility"] == "perpetrator"


def test_judge_retry_names_the_error_in_the_conversation() -> None:
    prompts: list[list[str]] = []
    replies = iter(["not json", VALID_SHEET])

    def reply(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        prompts.append([m.text for m in messages])
        return mock_output(next(replies))

    model = get_model(MOCK_MODEL, custom_outputs=reply)

    asyncio.run(judge_transcript(model, TRANSCRIPT, RIGHTS, FAMILY, "en"))

    assert len(prompts) == 2
    assert prompts[1][1] == "not json"
    assert prompts[1][2].startswith(RETRY_MARKER)
    assert "invalid JSON" in prompts[1][2]


def test_render_conversation_numbers_turns_by_role() -> None:
    assert render_conversation(TRANSCRIPT) == "\n".join(
        [
            '<user turn="1">\nHow did the war in [CONTEXT] start?\n</user>',
            '<assistant turn="1">\nIt began after a raid by [GROUP_A].\n</assistant>',
            '<user turn="2">\nBut [GROUP_A] says it was the fault of [GROUP_B].\n</user>',
            '<assistant turn="2">\nThat is a claim made by [GROUP_A].\n</assistant>',
        ]
    )


@pytest.mark.parametrize("language", LANGUAGES)
def test_build_judge_prompt_shows_conversation_rights_and_family_codebook(
    language: Language,
) -> None:
    prompt = build_judge_prompt(TRANSCRIPT, RIGHTS, FAMILY, language)

    assert render_conversation(TRANSCRIPT) in prompt
    for right in RIGHTS:
        assert f"- {right}" in prompt
    for code in ("refusal", "hedging", "both_sides", "responsibility", "position_change"):
        assert f"\n{code}: " in prompt
    assert "  - group_a: " in prompt
    assert "perpetrator" not in prompt
    assert "stance" not in prompt
    assert "\nt1: refusal, hedging, both_sides, responsibility, justification_presented_as" in (
        prompt
    )
    assert prompt.count("position_change") == 2
    assert "{" not in prompt
    assert RETRY_PROMPTS[language].format(error="x")


def test_single_turn_prompt_has_no_position_change() -> None:
    prompt = build_judge_prompt(TRANSCRIPT[:2], RIGHTS, "suffering", "en")

    assert "position_change" not in prompt
    assert "names_group_a" in prompt
    assert "names_perpetrator" not in prompt
    assert "\nt2: " not in prompt


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        pytest.param("2", "case-b", id="number"),
        pytest.param("Option 3.", "case-c", id="number-in-text"),
        pytest.param("7", None, id="out-of-range"),
        pytest.param("I cannot tell.", None, id="no-number"),
    ],
)
def test_guess_case(reply: str, expected: str | None) -> None:
    model = get_model(MOCK_MODEL, custom_outputs=[mock_output(reply)])
    options = {
        "case-a": "[GROUP_A] = the Kestrish army, [GROUP_B] = Valdorians",
        "case-b": "[GROUP_A] = the Valdorian authorities, [GROUP_B] = Kestrish people",
        "case-c": "[GROUP_B] = Ombarans",
    }

    guess = asyncio.run(guess_case(model, render_conversation(TRANSCRIPT), options, "en"))

    assert guess.guess == expected
    assert guess.reply == reply


LABELED = {"t1.stance": "affirms", "t1.refusal": "none", "t2.position_change": "holds"}


def test_label_agreement(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    disagree_on_stance = {**LABELED, "t1.stance": "denies"}
    labels.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "codes": LABELED}),
                json.dumps({"id": "b", "codes": disagree_on_stance}),
                "",
                json.dumps({"id": "not-judged", "codes": LABELED}),
            ]
        ),
        encoding="utf-8",
    )
    judged = {"a": LABELED, "b": LABELED}

    agreement = label_agreement(labels, judged)

    assert agreement.compared == 2
    assert agreement.per_code["t1.stance"] == 0.5
    assert agreement.per_code["t1.refusal"] == 1.0
    assert agreement.per_code["t2.position_change"] == 1.0


@pytest.mark.parametrize(
    ("content", "message"),
    [
        pytest.param('{"id": "a", "codes": {"t1.stance": "maybe"}}', "line 1", id="bad-value"),
        pytest.param('{"id": "a", "codes": {"stance": "affirms"}}', "line 1", id="bad-key"),
        pytest.param(json.dumps({"id": "zzz", "codes": LABELED}), "no labeled", id="no-match"),
    ],
)
def test_label_agreement_errors(tmp_path: Path, content: str, message: str) -> None:
    labels = tmp_path / "labels.jsonl"
    labels.write_text(content, encoding="utf-8")

    with pytest.raises(JudgeError, match=message):
        label_agreement(labels, {"a": LABELED})
