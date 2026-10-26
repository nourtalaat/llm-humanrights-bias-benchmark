import json

import pytest
from support import VALID_CODES, VALID_SHEET_JSON

from hrbias.codebook import (
    CODEBOOK,
    CODES,
    AnswerSheet,
    CodebookError,
    Hedging,
    Refusal,
    Stance,
    YesNo,
    parse_answer_sheet,
)
from hrbias.dataset import LANGUAGES


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(VALID_SHEET_JSON, id="plain"),
        pytest.param(f"  \n{VALID_SHEET_JSON}\n ", id="surrounding-whitespace"),
        pytest.param(f"```json\n{VALID_SHEET_JSON}\n```", id="code-fence"),
    ],
)
def test_parse_valid_answer_sheet(reply: str) -> None:
    sheet = parse_answer_sheet(reply)

    assert sheet == AnswerSheet(
        refusal=Refusal.NONE,
        stance=Stance.AFFIRMS,
        hedging=Hedging.LIGHT,
        both_sides=YesNo.NO,
        blames_group=YesNo.NO,
        rights_conditional_on_security=YesNo.YES,
    )
    assert sheet.codes() == VALID_CODES


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        pytest.param("not json", "invalid JSON", id="not-json"),
        pytest.param(VALID_SHEET_JSON[:-1], "invalid JSON", id="truncated"),
        pytest.param(
            json.dumps({k: v for k, v in VALID_CODES.items() if k != "stance"}),
            "stance",
            id="missing-code",
        ),
        pytest.param(json.dumps({**VALID_CODES, "stance": "maybe"}), "stance", id="unknown-value"),
        pytest.param(json.dumps({**VALID_CODES, "tone": "warm"}), "tone", id="extra-code"),
        pytest.param(json.dumps([VALID_CODES]), "AnswerSheet", id="not-an-object"),
    ],
)
def test_parse_invalid_answer_sheet(reply: str, message: str) -> None:
    with pytest.raises(CodebookError, match=message):
        parse_answer_sheet(reply)


def test_codebook_defines_every_code_and_value_in_every_language() -> None:
    assert set(CODEBOOK) == set(CODES)
    for code, definition in CODEBOOK.items():
        enum_type = AnswerSheet.model_fields[code].annotation
        assert set(definition.values) == {member.value for member in enum_type}
        for language in LANGUAGES:
            assert definition.question[language]
            for meanings in definition.values.values():
                assert meanings[language]
