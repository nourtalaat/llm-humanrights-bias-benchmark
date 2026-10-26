"""Fill template slots with the values of one case.

A case gives every role (``victim``, ``perpetrator``, ``case``) a set of named
values: actor forms for the two actor roles, case fields for ``case``. Two
slot kinds exist:

- ``{<role>.<name>}`` inserts the text of the role's value ``<name>``.
- ``{agree:<role>.<name>|<key>=<text>|<key>=<text>...}`` inserts the text
  whose key matches that value's grammatical agreement key (for example
  ``pl`` or ``f_sg``).

Any other slot, or a slot that cannot be filled, raises ``GrammarError``.
"""

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass

AGREE_SLOT_PREFIX = "agree:"
AGREE_OPTION_SEPARATOR = "|"
AGREE_KEY_SEPARATOR = "="
ROLE_SEPARATOR = "."

_SLOT_PATTERN = re.compile(r"\{([^{}]*)\}")
_BRACES = "{}"


class GrammarError(ValueError):
    """A template slot could not be filled for a case."""


@dataclass(frozen=True)
class SlotValue:
    """Text a slot inserts, and its agreement key when it has one."""

    text: str
    agreement: str | None = None


@dataclass(frozen=True)
class FilledText:
    """Template text with every slot filled.

    ``inserted_terms`` lists, in first-use order and without repeats, the
    exact texts of the tracked roles written into the text.
    """

    text: str
    inserted_terms: list[str]


def fill_slots(
    text: str,
    values: Mapping[str, Mapping[str, SlotValue]],
    *,
    context: str,
    tracked_roles: Collection[str] = (),
) -> FilledText:
    """Fill every slot in ``text`` with ``values``, which maps role to named values.

    Texts inserted for a role in ``tracked_roles`` are listed in
    ``inserted_terms``. Raises ``GrammarError``, starting with ``context``,
    when a slot names an unknown role or value, has no option for the value's
    agreement key, is malformed, or when a brace is left unmatched.
    """
    inserted: list[str] = []

    def replace(match: re.Match[str]) -> str:
        slot = match.group(1)
        if slot.startswith(AGREE_SLOT_PREFIX):
            return _fill_agreement(slot.removeprefix(AGREE_SLOT_PREFIX), values, context)

        role, value = _lookup(slot, slot, values, context)
        if role in tracked_roles and value.text not in inserted:
            inserted.append(value.text)

        return value.text

    filled = _SLOT_PATTERN.sub(replace, text)
    if any(brace in filled for brace in _BRACES):
        raise GrammarError(f"{context}: unmatched brace in {text!r}")

    return FilledText(text=filled, inserted_terms=inserted)


def slot_roles(text: str) -> set[str]:
    """Return the roles that the slots of ``text`` refer to, agreement slots included."""
    roles = set()
    for slot in _SLOT_PATTERN.findall(text):
        reference = slot.removeprefix(AGREE_SLOT_PREFIX).split(AGREE_OPTION_SEPARATOR, 1)[0]
        role, separator, _ = reference.partition(ROLE_SEPARATOR)
        if separator:
            roles.add(role.strip())

    return roles


def _fill_agreement(spec: str, values: Mapping[str, Mapping[str, SlotValue]], context: str) -> str:
    reference, separator, options_text = spec.partition(AGREE_OPTION_SEPARATOR)
    slot = f"{AGREE_SLOT_PREFIX}{spec}"
    if not separator:
        raise GrammarError(
            f"{context}: {{{slot}}} has no options, expected "
            f"{{{AGREE_SLOT_PREFIX}<role>.<name>|<key>{AGREE_KEY_SEPARATOR}<text>...}}"
        )

    _, value = _lookup(reference.strip(), slot, values, context)
    if value.agreement is None:
        raise GrammarError(
            f"{context}: {{{slot}}} refers to {reference.strip()!r}, which has no agreement key"
        )

    options = _parse_agreement_options(options_text, slot, context)
    if value.agreement not in options:
        known = ", ".join(options)
        raise GrammarError(
            f"{context}: no option for agreement key {value.agreement!r} in "
            f"{{{slot}}} (options: {known})"
        )

    return options[value.agreement]


def _lookup(
    reference: str, slot: str, values: Mapping[str, Mapping[str, SlotValue]], context: str
) -> tuple[str, SlotValue]:
    role, separator, name = reference.partition(ROLE_SEPARATOR)
    if not separator:
        raise GrammarError(f"{context}: unknown slot {{{slot}}}")

    if role not in values:
        known = ", ".join(sorted(values))
        raise GrammarError(f"{context}: no role {role!r} for slot {{{slot}}} (roles here: {known})")

    named = values[role]
    if name not in named:
        known = ", ".join(sorted(named))
        raise GrammarError(f"{context}: unknown {role} value {name!r} (known: {known})")

    return role, named[name]


def _parse_agreement_options(options_text: str, slot: str, context: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for option in options_text.split(AGREE_OPTION_SEPARATOR):
        key, separator, value = option.partition(AGREE_KEY_SEPARATOR)
        key = key.strip()
        if not separator or not key:
            raise GrammarError(
                f"{context}: malformed agreement option {option!r} in "
                f"{{{slot}}}, expected <key>{AGREE_KEY_SEPARATOR}<text>"
            )

        if key in options:
            raise GrammarError(f"{context}: duplicate agreement key {key!r} in {{{slot}}}")

        options[key] = value

    return options
