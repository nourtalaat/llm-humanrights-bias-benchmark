"""Mask group identity in a model's answer before the judge sees it.

Redaction is deterministic: we know which group each prompt was about, so we
replace that group's names (``group_terms``) with ``[GROUP]`` and its related
places and institutions (``related_terms``) with ``[PLACE]``. All terms are
matched in one pass with the longer terms tried first, so ``Miral Abbey``
becomes ``[PLACE]`` rather than ``[GROUP] Abbey``, and no text is replaced
twice. Tokens already in the text are left as they are.

Matching follows each language's spelling rules:

- English: case-insensitive, whole words, keeping a possessive ``'s`` or ``'``.
- Arabic: optional attached prefixes (conjunction, preposition, article) and
  the ``لل`` form, where the article's alif drops after ``ل``; any alif
  spelling (with or without hamza) matches any other.
- Hebrew: up to three attached one-letter prefixes, then the full term. A
  term's leading ``ה`` may drop only right after ``ב``, ``כ`` or ``ל``, the one
  place Hebrew spelling drops the article (``לוולדורים``); elsewhere it stays
  (``שהוולדורים``), because many names start with a ``ה`` that is part of the
  word. Indefinite forms must therefore be listed as variants. An optional
  hyphen or maqaf may follow the prefixes, and a doubled ``וו`` or ``יי`` also
  matches the single letter.

Attached prefixes are kept in the output (``ל[GROUP]``, ``ال[GROUP]``) so the
sentence still reads naturally for the judge; they say nothing about which
group it is.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

GROUP_TOKEN = "[GROUP]"
PLACE_TOKEN = "[PLACE]"

ARABIC_ARTICLE = "ال"
ARABIC_CONJUNCTION_PREFIXES = "وف"
ARABIC_PREPOSITION_PREFIXES = "بك"
ARABIC_LAM = "ل"
ARABIC_ALIF_FORMS = "اأإآ"

HEBREW_ARTICLE = "ה"
HEBREW_PREFIX_LETTERS = "והבכלמש"
HEBREW_ARTICLE_ABSORBING_PREFIXES = "בכל"
HEBREW_MAX_PREFIX_LETTERS = 3
HEBREW_PREFIX_JOINERS = "-־"
HEBREW_DOUBLED_LETTERS = ("וו", "יי")

ENGLISH_APOSTROPHES = "'’"

_NOT_AFTER_WORD = r"(?<!\w)"
_NOT_BEFORE_WORD = r"(?!\w)"
_WHITESPACE_RUN = r"\s+"
_KEEP_GROUP = "keep"
_TERM_GROUP = "t"

_ARABIC_PREFIX = (
    f"[{ARABIC_CONJUNCTION_PREFIXES}]?"
    f"(?:[{ARABIC_PREPOSITION_PREFIXES}]?(?:{ARABIC_ARTICLE})?|{ARABIC_LAM}{ARABIC_LAM}?)"
)
_HEBREW_JOINER = f"[{re.escape(HEBREW_PREFIX_JOINERS)}]"
_HEBREW_PREFIX = f"[{HEBREW_PREFIX_LETTERS}]{{0,{HEBREW_MAX_PREFIX_LETTERS}}}?{_HEBREW_JOINER}?"
_HEBREW_ABSORBING = f"[{HEBREW_ARTICLE_ABSORBING_PREFIXES}]"
_HEBREW_ABSORBING_JOINED = f"{_HEBREW_ABSORBING}{_HEBREW_JOINER}"
_ENGLISH_APOSTROPHE = f"[{ENGLISH_APOSTROPHES}]"
_ENGLISH_SUFFIX = f"{_ENGLISH_APOSTROPHE}s|(?<=s){_ENGLISH_APOSTROPHE}|"


@dataclass(frozen=True)
class Redaction:
    """Redacted text and how many terms were replaced."""

    text: str
    replacements: int


def redact(
    text: str,
    group_terms: Iterable[str],
    related_terms: Iterable[str],
    language: str,
) -> Redaction:
    """Replace group terms with ``[GROUP]`` and related terms with ``[PLACE]``.

    A term listed in both groups is treated as a group term.
    """
    targets = _targets(group_terms, related_terms)
    pattern = _combined_pattern([term for term, _ in targets], language)
    tokens = [token for _, token in targets]
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        if match[_KEEP_GROUP] is not None:
            return match[_KEEP_GROUP]

        replacements += 1
        token = next(
            tokens[int(name.removeprefix(_TERM_GROUP))]
            for name, value in match.groupdict().items()
            if name.startswith(_TERM_GROUP) and value is not None
        )
        return f"{match['prefix']}{token}{match['suffix']}"

    redacted = pattern.sub(replace, text) if targets else text
    return Redaction(text=redacted, replacements=replacements)


def _targets(group_terms: Iterable[str], related_terms: Iterable[str]) -> list[tuple[str, str]]:
    tokens: dict[str, str] = {}
    for term in related_terms:
        if term.strip():
            tokens[term.strip()] = PLACE_TOKEN

    for term in group_terms:
        if term.strip():
            tokens[term.strip()] = GROUP_TOKEN

    return sorted(tokens.items(), key=lambda item: len(item[0]), reverse=True)


def _combined_pattern(terms: list[str], language: str) -> re.Pattern[str]:
    if language == "ar":
        prefix, suffix, flags, core = _ARABIC_PREFIX, "", 0, _arabic_core
    elif language == "he":
        prefix, suffix, flags, core = _HEBREW_PREFIX, "", 0, _hebrew_core
    elif language == "en":
        prefix, suffix, flags, core = "", _ENGLISH_SUFFIX, re.IGNORECASE, _spaced
    else:
        raise ValueError(f"redact: unsupported language {language!r}")

    alternatives = "|".join(f"(?P<{_TERM_GROUP}{i}>{core(t)})" for i, t in enumerate(terms))
    keep = "|".join(re.escape(token) for token in (GROUP_TOKEN, PLACE_TOKEN))
    return re.compile(
        f"(?P<{_KEEP_GROUP}>{keep})|{_NOT_AFTER_WORD}(?P<prefix>{prefix})"
        f"(?:{alternatives})(?P<suffix>{suffix}){_NOT_BEFORE_WORD}",
        flags,
    )


def _arabic_core(term: str) -> str:
    core = _spaced(term.removeprefix(ARABIC_ARTICLE))
    return re.sub(f"[{ARABIC_ALIF_FORMS}]", f"[{ARABIC_ALIF_FORMS}]", core)


def _hebrew_core(term: str) -> str:
    full = _hebrew_spelling(term)
    if not term.startswith(HEBREW_ARTICLE):
        return full

    after_absorbing_prefix = "|".join(
        f"(?<={absorbing})" for absorbing in (_HEBREW_ABSORBING, _HEBREW_ABSORBING_JOINED)
    )
    rest = _hebrew_spelling(term.removeprefix(HEBREW_ARTICLE))
    return f"{full}|(?:{after_absorbing_prefix}){rest}"


def _hebrew_spelling(term: str) -> str:
    pattern = _spaced(term)
    for doubled in HEBREW_DOUBLED_LETTERS:
        pattern = pattern.replace(doubled, f"{doubled[0]}{{1,2}}")

    return pattern


def _spaced(term: str) -> str:
    return _WHITESPACE_RUN.join(re.escape(word) for word in term.split())
