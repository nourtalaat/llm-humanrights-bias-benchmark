"""Fill template slots with a group's surface forms.

Two slot kinds exist:

- ``{group.<form>}`` inserts the group's surface form named ``<form>``.
- ``{agree:<key>=<text>|<key>=<text>}`` inserts the text whose key matches the
  group's grammatical agreement key (for example ``m_pl`` or ``f_sg``).

Any other slot, or a slot that cannot be filled, raises ``GrammarError``.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

GROUP_SLOT_PREFIX = "group."
AGREE_SLOT_PREFIX = "agree:"
AGREE_OPTION_SEPARATOR = "|"
AGREE_KEY_SEPARATOR = "="

_SLOT_PATTERN = re.compile(r"\{([^{}]*)\}")
_BRACES = "{}"


class GrammarError(ValueError):
    """A template slot could not be filled for a group."""


@dataclass(frozen=True)
class FilledText:
    """Template text with every slot filled.

    ``inserted_terms`` lists, in first-use order and without repeats, the exact
    group surface forms written into the text.
    """

    text: str
    inserted_terms: list[str]


def fill_slots(
    text: str,
    forms: Mapping[str, str],
    agreement: str,
    *,
    template_id: str,
    group_id: str,
) -> FilledText:
    """Fill every slot in ``text`` for one group.

    Raises ``GrammarError`` naming the template and group when a slot names an
    unknown form, has no option for the group's agreement key, is malformed, or
    when a brace is left unmatched.
    """
    context = f"template {template_id!r}, group {group_id!r}"
    inserted: list[str] = []

    def replace(match: re.Match[str]) -> str:
        slot = match.group(1)
        if slot.startswith(GROUP_SLOT_PREFIX):
            value = _fill_form(slot.removeprefix(GROUP_SLOT_PREFIX), forms, context)
            if value not in inserted:
                inserted.append(value)

            return value

        if slot.startswith(AGREE_SLOT_PREFIX):
            return _fill_agreement(slot.removeprefix(AGREE_SLOT_PREFIX), agreement, context)

        raise GrammarError(f"{context}: unknown slot {{{slot}}}")

    filled = _SLOT_PATTERN.sub(replace, text)
    if any(brace in filled for brace in _BRACES):
        raise GrammarError(f"{context}: unmatched brace in {text!r}")

    return FilledText(text=filled, inserted_terms=inserted)


def _fill_form(form: str, forms: Mapping[str, str], context: str) -> str:
    if form not in forms:
        known = ", ".join(sorted(forms))
        raise GrammarError(f"{context}: unknown form {form!r} (known forms: {known})")

    return forms[form]


def _fill_agreement(spec: str, agreement: str, context: str) -> str:
    options = _parse_agreement_options(spec, context)
    if agreement not in options:
        known = ", ".join(options)
        raise GrammarError(
            f"{context}: no option for agreement key {agreement!r} in "
            f"{{{AGREE_SLOT_PREFIX}{spec}}} (options: {known})"
        )

    return options[agreement]


def _parse_agreement_options(spec: str, context: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for option in spec.split(AGREE_OPTION_SEPARATOR):
        key, separator, value = option.partition(AGREE_KEY_SEPARATOR)
        key = key.strip()
        if not separator or not key:
            raise GrammarError(
                f"{context}: malformed agreement option {option!r} in "
                f"{{{AGREE_SLOT_PREFIX}{spec}}}, expected <key>{AGREE_KEY_SEPARATOR}<text>"
            )

        if key in options:
            raise GrammarError(
                f"{context}: duplicate agreement key {key!r} in {{{AGREE_SLOT_PREFIX}{spec}}}"
            )

        options[key] = value

    return options
