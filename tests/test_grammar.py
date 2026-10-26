import pytest

from hrbias.grammar import FilledText, GrammarError, fill_slots

TEMPLATE_ID = "example-template"
GROUP_ID = "example-group"
AR_SVO_TEXT = (
    "{group.name} {agree:m_pl=معروفون|f_pl=معروفات|f_sg=معروفة} "
    "في {agree:m_pl=منطقتهم|f_pl=منطقتهن|f_sg=منطقتها}."
)
HE_SVO_TEXT = (
    "{group.name} {agree:m_sg=ידוע|f_pl=ידועות|m_pl=ידועים} "
    "{agree:m_sg=באזורו|f_pl=באזורן|m_pl=באזורם}."
)


@pytest.mark.parametrize(
    ("text", "forms", "agreement", "expected"),
    [
        pytest.param(
            "Do {group.plural} have this right?",
            {"plural": "Valdorians"},
            "pl",
            FilledText("Do Valdorians have this right?", ["Valdorians"]),
            id="en-form",
        ),
        pytest.param(
            "The {group.name} {agree:sg=is|pl=are} known here.",
            {"name": "Selvin Circle"},
            "sg",
            FilledText("The Selvin Circle is known here.", ["Selvin Circle"]),
            id="en-agreement-singular",
        ),
        pytest.param(
            "The {group.name} {agree:sg=is|pl=are} known here.",
            {"name": "Wardens of Esk"},
            "pl",
            FilledText("The Wardens of Esk are known here.", ["Wardens of Esk"]),
            id="en-agreement-plural",
        ),
        pytest.param(
            AR_SVO_TEXT,
            {"name": "حراس إسك"},
            "m_pl",
            FilledText("حراس إسك معروفون في منطقتهم.", ["حراس إسك"]),
            id="ar-masculine-plural",
        ),
        pytest.param(
            AR_SVO_TEXT,
            {"name": "أخوات ميرال"},
            "f_pl",
            FilledText("أخوات ميرال معروفات في منطقتهن.", ["أخوات ميرال"]),
            id="ar-feminine-plural",
        ),
        pytest.param(
            "{agree:m_pl=قضى|f_sg=قضت} {group.name} عقودًا هنا.",
            {"name": "جماعة سيلفين"},
            "f_sg",
            FilledText("قضت جماعة سيلفين عقودًا هنا.", ["جماعة سيلفين"]),
            id="ar-feminine-singular-verb-first",
        ),
        pytest.param(
            HE_SVO_TEXT,
            {"name": "חוג סלווין"},
            "m_sg",
            FilledText("חוג סלווין ידוע באזורו.", ["חוג סלווין"]),
            id="he-masculine-singular",
        ),
        pytest.param(
            HE_SVO_TEXT,
            {"name": "אחיות מירל"},
            "f_pl",
            FilledText("אחיות מירל ידועות באזורן.", ["אחיות מירל"]),
            id="he-feminine-plural",
        ),
        pytest.param(
            "{group.plural} and {group.adjective} and {group.plural}",
            {"plural": "Ombarans", "adjective": "Ombaran"},
            "pl",
            FilledText("Ombarans and Ombaran and Ombarans", ["Ombarans", "Ombaran"]),
            id="inserted-terms-without-repeats",
        ),
        pytest.param(
            "No slots at all.",
            {"plural": "Ombarans"},
            "pl",
            FilledText("No slots at all.", []),
            id="no-slots",
        ),
    ],
)
def test_fill_slots(text: str, forms: dict[str, str], agreement: str, expected: FilledText) -> None:
    filled = fill_slots(text, forms, agreement, template_id=TEMPLATE_ID, group_id=GROUP_ID)

    assert filled == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param("Do {group.singular} agree?", "unknown form 'singular'", id="unknown-form"),
        pytest.param(
            "They {agree:m_pl=go|f_pl=go}.",
            "no option for agreement key 'f_sg'",
            id="missing-agreement-key",
        ),
        pytest.param("Do {grop.plural} agree?", r"unknown slot \{grop.plural\}", id="unknown-slot"),
        pytest.param("They {agree:m_pl}.", "malformed agreement option", id="malformed-option"),
        pytest.param(
            "They {agree:f_sg=a|f_sg=b}.", "duplicate agreement key 'f_sg'", id="duplicate-key"
        ),
        pytest.param("Do {group.plural agree?", "unmatched brace", id="unclosed-brace"),
        pytest.param("Do group.plural} agree?", "unmatched brace", id="stray-closing-brace"),
    ],
)
def test_fill_slots_errors_name_template_and_group(text: str, message: str) -> None:
    with pytest.raises(GrammarError, match=message) as excinfo:
        fill_slots(
            text, {"plural": "Valdorians"}, "f_sg", template_id=TEMPLATE_ID, group_id=GROUP_ID
        )

    assert TEMPLATE_ID in str(excinfo.value)
    assert GROUP_ID in str(excinfo.value)
