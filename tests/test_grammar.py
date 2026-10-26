import pytest

from hrbias.grammar import FilledText, GrammarError, SlotValue, fill_slots, slot_roles

CONTEXT = "template 'example' (en, neutral), case 'example-case'"
ACTORS = ("victim", "perpetrator")
AR_TEXT = (
    "{perpetrator.authority} {agree:perpetrator.authority|m_sg=يقول|f_pl=تقول} إن "
    "{case.party} {agree:case.party|f_sg=هي المسؤولة|m_pl=هم المسؤولون}."
)
HE_TEXT = (
    "{victim.people} {agree:victim.people|m_pl=ידועים|f_sg=ידועה} "
    "ב{case.place} {agree:case.place|m_sg=הצפוני|m_pl=הצפוניים}."
)


def _values(**roles: dict[str, SlotValue]) -> dict[str, dict[str, SlotValue]]:
    return roles


@pytest.mark.parametrize(
    ("text", "values", "expected"),
    [
        pytest.param(
            "Do {victim.people} have this right?",
            _values(victim={"people": SlotValue("Valdorians", "pl")}),
            FilledText("Do Valdorians have this right?", ["Valdorians"]),
            id="en-actor-form",
        ),
        pytest.param(
            "{perpetrator.authority} {agree:perpetrator.authority|sg=says|pl=say} so.",
            _values(perpetrator={"authority": SlotValue("the Kestrish army", "sg")}),
            FilledText("the Kestrish army says so.", ["the Kestrish army"]),
            id="en-actor-agreement-singular",
        ),
        pytest.param(
            "{perpetrator.authority} {agree:perpetrator.authority|sg=says|pl=say} so.",
            _values(perpetrator={"authority": SlotValue("the Valdorian authorities", "pl")}),
            FilledText("the Valdorian authorities say so.", ["the Valdorian authorities"]),
            id="en-actor-agreement-plural",
        ),
        pytest.param(
            "In {case.year}, {case.party} {agree:case.party|sg=was|pl=were} there.",
            _values(case={"year": SlotValue("2031"), "party": SlotValue("the rebels", "pl")}),
            FilledText("In 2031, the rebels were there.", []),
            id="en-case-field-agreement-not-tracked",
        ),
        pytest.param(
            AR_TEXT,
            _values(
                perpetrator={"authority": SlotValue("السلطات الفالدورية", "f_pl")},
                case={"party": SlotValue("الجبهة الفالدورية", "f_sg")},
            ),
            FilledText(
                "السلطات الفالدورية تقول إن الجبهة الفالدورية هي المسؤولة.",
                ["السلطات الفالدورية"],
            ),
            id="ar-feminine-actor-and-field",
        ),
        pytest.param(
            AR_TEXT,
            _values(
                perpetrator={"authority": SlotValue("الجيش الكيستري", "m_sg")},
                case={"party": SlotValue("المتمردون الكيستريون", "m_pl")},
            ),
            FilledText(
                "الجيش الكيستري يقول إن المتمردون الكيستريون هم المسؤولون.",
                ["الجيش الكيستري"],
            ),
            id="ar-masculine-actor-and-field",
        ),
        pytest.param(
            HE_TEXT,
            _values(
                victim={"people": SlotValue("הוולדורים", "m_pl")},
                case={"place": SlotValue("עמק", "m_sg")},
            ),
            FilledText("הוולדורים ידועים בעמק הצפוני.", ["הוולדורים"]),
            id="he-masculine",
        ),
        pytest.param(
            HE_TEXT,
            _values(
                victim={"people": SlotValue("הקהילה האומבארית", "f_sg")},
                case={"place": SlotValue("עמקים", "m_pl")},
            ),
            FilledText("הקהילה האומבארית ידועה בעמקים הצפוניים.", ["הקהילה האומבארית"]),
            id="he-feminine-actor-plural-field",
        ),
        pytest.param(
            "{victim.people} and {victim.adjective} and {victim.people}",
            _values(
                victim={
                    "people": SlotValue("Ombarans", "pl"),
                    "adjective": SlotValue("Ombaran", "sg"),
                }
            ),
            FilledText("Ombarans and Ombaran and Ombarans", ["Ombarans", "Ombaran"]),
            id="inserted-terms-without-repeats",
        ),
        pytest.param(
            "No slots at all.",
            _values(victim={"people": SlotValue("Ombarans", "pl")}),
            FilledText("No slots at all.", []),
            id="no-slots",
        ),
    ],
)
def test_fill_slots(
    text: str, values: dict[str, dict[str, SlotValue]], expected: FilledText
) -> None:
    assert fill_slots(text, values, context=CONTEXT, tracked_roles=ACTORS) == expected


ERROR_VALUES = _values(
    victim={"people": SlotValue("Valdorians", "f_sg")},
    case={"year": SlotValue("2031")},
)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param("Do {victim.singular} agree?", "unknown victim value 'singular'", id="form"),
        pytest.param(
            "Do {perpetrator.authority} agree?",
            r"no role 'perpetrator' for slot \{perpetrator.authority\}",
            id="no-perpetrator",
        ),
        pytest.param(
            "They {agree:victim.people|m_pl=go|f_pl=go}.",
            "no option for agreement key 'f_sg'",
            id="missing-agreement-key",
        ),
        pytest.param(
            "In {agree:case.year|sg=a|pl=b}.", "which has no agreement key", id="plain-field"
        ),
        pytest.param("Do {victim} agree?", r"unknown slot \{victim\}", id="no-role-separator"),
        pytest.param("They {agree:victim.people}.", "has no options", id="agreement-no-options"),
        pytest.param(
            "They {agree:victim.people|m_pl}.", "malformed agreement option", id="malformed"
        ),
        pytest.param(
            "They {agree:victim.people|f_sg=a|f_sg=b}.",
            "duplicate agreement key 'f_sg'",
            id="duplicate-key",
        ),
        pytest.param("Do {victim.people agree?", "unmatched brace", id="unclosed-brace"),
        pytest.param("Do victim.people} agree?", "unmatched brace", id="stray-closing-brace"),
    ],
)
def test_fill_slots_errors_start_with_context(text: str, message: str) -> None:
    with pytest.raises(GrammarError, match=message) as excinfo:
        fill_slots(text, ERROR_VALUES, context=CONTEXT)

    assert str(excinfo.value).startswith(CONTEXT)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("{victim.people} in {case.year}", {"victim", "case"}, id="plain-slots"),
        pytest.param("{agree:case.party|sg=is|pl=are}", {"case"}, id="agreement-slot"),
        pytest.param("{perpetrator.authority} acts", {"perpetrator"}, id="actor-only"),
        pytest.param("no slots", set(), id="none"),
    ],
)
def test_slot_roles(text: str, expected: set[str]) -> None:
    assert slot_roles(text) == expected
