import pytest
from support import FIXTURE_DATA_DIR

from hrbias.dataset import load_benchmark_data
from hrbias.redact import (
    CONTEXT_TOKEN,
    PERPETRATOR_TOKEN,
    VICTIM_TOKEN,
    Redaction,
    redact,
    role_targets,
)

EN_VALDORIAN = ["Valdorians", "Valdorian"]
EN_VALDORIAN_PLACES = ["Valdoria", "Selen"]
EN_MIRAL = ["Sisters of Miral", "Sister of Miral", "Miral"]
EN_MIRAL_PLACES = ["Miral Abbey"]
AR_VALDORIAN = [
    "الفالدوريون",
    "الجالية الفالدورية",
    "الفالدوريين",
    "الفالدوري",
    "الفالدورية",
    "الفالدوريات",
]
AR_VALDORIAN_PLACES = ["فالدوريا", "سيلين"]
AR_OMBARAN = ["الأومباريون", "الأومباري"]
HE_VALDORIAN = [
    "הוולדורים",
    "הקהילה הוולדורית",
    "הוולדורי",
    "הוולדורית",
    "הוולדוריות",
    "וולדורים",
    "וולדורי",
]
HE_VALDORIAN_PLACES = ["וולדוריה", "סלן"]
ARTICLES = {"", "ال", "ה"}


@pytest.mark.parametrize(
    ("language", "group_terms", "related_terms", "text", "expected"),
    [
        pytest.param(
            "en",
            EN_VALDORIAN,
            EN_VALDORIAN_PLACES,
            "Valdorian workers have this right.",
            Redaction("[GROUP_B] workers have this right.", 1),
            id="en-adjective-not-inserted",
        ),
        pytest.param(
            "en",
            EN_VALDORIAN,
            EN_VALDORIAN_PLACES,
            "the valdorians' union and a VALDORIAN's view, from Selen in Valdoria",
            Redaction(
                "the [GROUP_B]' union and a [GROUP_B]'s view, from [CONTEXT] in [CONTEXT]", 4
            ),
            id="en-case-and-possessive",
        ),
        pytest.param(
            "en",
            EN_VALDORIAN,
            EN_VALDORIAN_PLACES,
            "Valdorianism and Selenite are other words.",
            Redaction("Valdorianism and Selenite are other words.", 0),
            id="en-whole-words-only",
        ),
        pytest.param(
            "en",
            EN_MIRAL,
            EN_MIRAL_PLACES,
            "The Sisters  of Miral run Miral Abbey; each Sister of Miral and Miral itself.",
            Redaction("The [GROUP_B] run [CONTEXT]; each [GROUP_B] and [GROUP_B] itself.", 4),
            id="en-longest-first-place-before-group",
        ),
        pytest.param(
            "ar",
            AR_VALDORIAN,
            AR_VALDORIAN_PLACES,
            "يحق للفالدوريين التجمع.",
            Redaction("يحق لل[GROUP_B] التجمع.", 1),
            id="ar-genitive-with-lam-lam",
        ),
        pytest.param(
            "ar",
            AR_VALDORIAN,
            AR_VALDORIAN_PLACES,
            "والفالدوريون فالدوريون، وبالفالدوري وكالفالدورية، ولفالدوريات في فالدوريا.",
            Redaction(
                "وال[GROUP_B] [GROUP_B]، وبال[GROUP_B] وكال[GROUP_B]، ول[GROUP_B] في [CONTEXT].",
                6,
            ),
            id="ar-prefixes-and-article",
        ),
        pytest.param(
            "ar",
            AR_VALDORIAN,
            AR_VALDORIAN_PLACES,
            "أبناء الجالية الفالدورية وللجالية الفالدورية.",
            Redaction("أبناء ال[GROUP_B] ولل[GROUP_B].", 2),
            id="ar-multiword-term",
        ),
        pytest.param(
            "ar",
            AR_OMBARAN,
            [],
            "الاومباريون والإومباري.",
            Redaction("ال[GROUP_B] وال[GROUP_B].", 2),
            id="ar-any-alif-spelling",
        ),
        pytest.param(
            "ar",
            AR_VALDORIAN,
            AR_VALDORIAN_PLACES,
            "الفالدوريونية كلمة أخرى.",
            Redaction("الفالدوريونية كلمة أخرى.", 0),
            id="ar-whole-words-only",
        ),
        pytest.param(
            "he",
            HE_VALDORIAN,
            HE_VALDORIAN_PLACES,
            "יש לוולדורים זכות.",
            Redaction("יש ל[GROUP_B] זכות.", 1),
            id="he-prefix-absorbs-article",
        ),
        pytest.param(
            "he",
            HE_VALDORIAN,
            HE_VALDORIAN_PLACES,
            "אדם ולדורי אחד.",
            Redaction("אדם [GROUP_B] אחד.", 1),
            id="he-singular-single-vav",
        ),
        pytest.param(
            "he",
            HE_VALDORIAN,
            HE_VALDORIAN_PLACES,
            "ולוולדורים, שהוולדורים, מהוולדוריות, ל-וולדורים, ובוולדוריה ובסלן.",
            Redaction(
                "ול[GROUP_B], ש[GROUP_B], מ[GROUP_B], ל-[GROUP_B], וב[CONTEXT] וב[CONTEXT].",
                6,
            ),
            id="he-prefix-combinations",
        ),
        pytest.param(
            "he",
            HE_VALDORIAN,
            HE_VALDORIAN_PLACES,
            "בני הקהילה הוולדורית ולקהילה הוולדורית.",
            Redaction("בני [GROUP_B] ול[GROUP_B].", 2),
            id="he-multiword-term",
        ),
        pytest.param(
            "he",
            ["הודים"],
            [],
            "אנו מודים לך, וגם ההודים והודים.",
            Redaction("אנו מודים לך, וגם ה[GROUP_B] ו[GROUP_B].", 2),
            id="he-leading-he-is-part-of-the-name",
        ),
        pytest.param(
            "he",
            HE_VALDORIAN,
            HE_VALDORIAN_PLACES,
            "והוולדורים ולדורים.",
            Redaction("ו[GROUP_B] [GROUP_B].", 2),
            id="he-article-kept-after-vav-indefinite-listed",
        ),
        pytest.param(
            "en",
            ["Group", "Context"],
            ["Esk Valley"],
            "The [GROUP_B] and [GROUP_A] came from Esk Valley.",
            Redaction("The [GROUP_B] and [GROUP_A] came from [CONTEXT].", 1),
            id="never-replaces-inside-a-token",
        ),
        pytest.param(
            "en",
            ["Miral", "Selvin"],
            ["Miral Selvin Hall"],
            "At Miral Selvin Hall, Miral and Selvin met.",
            Redaction("At [CONTEXT], [GROUP_B] and [GROUP_B] met.", 3),
            id="each-position-replaced-once",
        ),
    ],
)
def test_redact(
    language: str,
    group_terms: list[str],
    related_terms: list[str],
    text: str,
    expected: Redaction,
) -> None:
    assert redact(text, role_targets(group_terms, [], related_terms), language) == expected


def test_redact_removes_every_actor_term_from_the_fixtures() -> None:
    data = load_benchmark_data(FIXTURE_DATA_DIR)
    for language, actors in data.actors.items():
        for actor in actors.values():
            terms = actor.terms()
            redacted = redact(" ".join(terms), role_targets(terms, [], []), language)

            assert redacted.replacements == len(terms)
            for word in redacted.text.split():
                assert word.endswith(VICTIM_TOKEN)
                assert word.removesuffix(VICTIM_TOKEN) in ARTICLES


@pytest.mark.parametrize(
    ("victim", "perpetrator", "context", "text", "expected"),
    [
        pytest.param(
            ["Valdorians"],
            ["the Kestrish army"],
            ["Orrin"],
            "The Kestrish army says Valdorians in Orrin started it.",
            Redaction("[GROUP_A] says [GROUP_B] in [CONTEXT] started it.", 3),
            id="three-roles",
        ),
        pytest.param(
            ["Sudara", "Sudaran"],
            ["the Sudaran army", "Sudaran"],
            ["Sudaran"],
            "The Sudaran army and Sudaran families.",
            Redaction("[GROUP_A] and [GROUP_B] families.", 2),
            id="shared-term-goes-to-victim-longest-first",
        ),
        pytest.param(
            [],
            ["Hale"],
            ["Hale"],
            "Minister Hale spoke.",
            Redaction("Minister [GROUP_A] spoke.", 1),
            id="perpetrator-before-context",
        ),
    ],
)
def test_role_targets(
    victim: list[str], perpetrator: list[str], context: list[str], text: str, expected: Redaction
) -> None:
    assert redact(text, role_targets(victim, perpetrator, context), "en") == expected


def test_role_tokens_are_neutral() -> None:
    assert (VICTIM_TOKEN, PERPETRATOR_TOKEN, CONTEXT_TOKEN) == (
        "[GROUP_B]",
        "[GROUP_A]",
        "[CONTEXT]",
    )


def test_redact_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unsupported language 'fr'"):
        redact("text", {VICTIM_TOKEN: ["term"]}, "fr")


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        pytest.param(
            {"[A]": ["Esk"], "[B]": ["Esk"]}, Redaction("From [A].", 1), id="first-token-wins"
        ),
        pytest.param(
            {"[B]": ["Esk"], "[A]": ["Esk"]}, Redaction("From [B].", 1), id="order-decides"
        ),
        pytest.param({}, Redaction("From Esk.", 0), id="no-targets"),
    ],
)
def test_redact_token_order(targets: dict[str, list[str]], expected: Redaction) -> None:
    assert redact("From Esk.", targets, "en") == expected
