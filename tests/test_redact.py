import pytest
from support import FIXTURE_DATA_DIR

from hrbias.dataset import load_benchmark_data
from hrbias.redact import GROUP_TOKEN, PLACE_TOKEN, Redaction, redact

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
            Redaction("[GROUP] workers have this right.", 1),
            id="en-adjective-not-inserted",
        ),
        pytest.param(
            "en",
            EN_VALDORIAN,
            EN_VALDORIAN_PLACES,
            "the valdorians' union and a VALDORIAN's view, from Selen in Valdoria",
            Redaction("the [GROUP]' union and a [GROUP]'s view, from [PLACE] in [PLACE]", 4),
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
            Redaction("The [GROUP] run [PLACE]; each [GROUP] and [GROUP] itself.", 4),
            id="en-longest-first-place-before-group",
        ),
        pytest.param(
            "ar",
            AR_VALDORIAN,
            AR_VALDORIAN_PLACES,
            "يحق للفالدوريين التجمع.",
            Redaction("يحق لل[GROUP] التجمع.", 1),
            id="ar-genitive-with-lam-lam",
        ),
        pytest.param(
            "ar",
            AR_VALDORIAN,
            AR_VALDORIAN_PLACES,
            "والفالدوريون فالدوريون، وبالفالدوري وكالفالدورية، ولفالدوريات في فالدوريا.",
            Redaction(
                "وال[GROUP] [GROUP]، وبال[GROUP] وكال[GROUP]، ول[GROUP] في [PLACE].",
                6,
            ),
            id="ar-prefixes-and-article",
        ),
        pytest.param(
            "ar",
            AR_VALDORIAN,
            AR_VALDORIAN_PLACES,
            "أبناء الجالية الفالدورية وللجالية الفالدورية.",
            Redaction("أبناء ال[GROUP] ولل[GROUP].", 2),
            id="ar-multiword-term",
        ),
        pytest.param(
            "ar",
            AR_OMBARAN,
            [],
            "الاومباريون والإومباري.",
            Redaction("ال[GROUP] وال[GROUP].", 2),
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
            Redaction("יש ל[GROUP] זכות.", 1),
            id="he-prefix-absorbs-article",
        ),
        pytest.param(
            "he",
            HE_VALDORIAN,
            HE_VALDORIAN_PLACES,
            "אדם ולדורי אחד.",
            Redaction("אדם [GROUP] אחד.", 1),
            id="he-singular-single-vav",
        ),
        pytest.param(
            "he",
            HE_VALDORIAN,
            HE_VALDORIAN_PLACES,
            "ולוולדורים, שהוולדורים, מהוולדוריות, ל-וולדורים, ובוולדוריה ובסלן.",
            Redaction(
                "ול[GROUP], ש[GROUP], מ[GROUP], ל-[GROUP], וב[PLACE] וב[PLACE].",
                6,
            ),
            id="he-prefix-combinations",
        ),
        pytest.param(
            "he",
            HE_VALDORIAN,
            HE_VALDORIAN_PLACES,
            "בני הקהילה הוולדורית ולקהילה הוולדורית.",
            Redaction("בני [GROUP] ול[GROUP].", 2),
            id="he-multiword-term",
        ),
        pytest.param(
            "he",
            ["הודים"],
            [],
            "אנו מודים לך, וגם ההודים והודים.",
            Redaction("אנו מודים לך, וגם ה[GROUP] ו[GROUP].", 2),
            id="he-leading-he-is-part-of-the-name",
        ),
        pytest.param(
            "he",
            HE_VALDORIAN,
            HE_VALDORIAN_PLACES,
            "והוולדורים ולדורים.",
            Redaction("ו[GROUP] [GROUP].", 2),
            id="he-article-kept-after-vav-indefinite-listed",
        ),
        pytest.param(
            "en",
            ["Group", "Place"],
            ["Esk Valley"],
            "The [GROUP] came from Esk Valley.",
            Redaction("The [GROUP] came from [PLACE].", 1),
            id="never-replaces-inside-a-token",
        ),
        pytest.param(
            "en",
            ["Miral", "Selvin"],
            ["Miral Selvin Hall"],
            "At Miral Selvin Hall, Miral and Selvin met.",
            Redaction("At [PLACE], [GROUP] and [GROUP] met.", 3),
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
    assert redact(text, _targets(group_terms, related_terms), language) == expected


def test_redact_removes_every_group_term_from_every_prompt() -> None:
    data = load_benchmark_data(FIXTURE_DATA_DIR)
    for language, groups in data.groups.items():
        for group in groups:
            text = " ".join(group.group_terms())
            redacted = redact(text, _targets(group.group_terms(), group.related_terms), language)

            assert redacted.replacements == len(group.group_terms())
            for word in redacted.text.split():
                assert word.endswith(GROUP_TOKEN)
                assert word.removesuffix(GROUP_TOKEN) in ARTICLES


def test_redact_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unsupported language 'fr'"):
        redact("text", {GROUP_TOKEN: ["term"]}, "fr")


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


def _targets(group_terms: list[str], related_terms: list[str]) -> dict[str, list[str]]:
    return {GROUP_TOKEN: group_terms, PLACE_TOKEN: related_terms}
