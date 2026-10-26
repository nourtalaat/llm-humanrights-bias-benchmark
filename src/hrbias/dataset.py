"""Load the benchmark YAML data and expand it into counterfactual Inspect samples.

The format is described in ``docs/data-format.md``. A counterfactual set is
one template paraphrase, in one framing and language, filled with every case
of the template's case set. Answers are only compared within a set, so each
sample records its ``set_id`` in its metadata.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

import yaml
from inspect_ai.dataset import MemoryDataset, Sample
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from hrbias import CANARY_LINE
from hrbias.grammar import SlotValue, fill_slots, slot_roles

Language = Literal["en", "ar", "he"]
Framing = Literal["neutral", "presupposing", "skeptical"]
Tier = Literal["real_event", "mirrored"]
Family = Literal[
    "suffering",
    "self_determination",
    "war_origin",
    "civilian_harm",
    "right_of_return",
    "detention",
    "accountability",
    "genocide_label",
    "collective_blame",
    "humanitarian_access",
]

LANGUAGES: tuple[Language, ...] = get_args(Language)
FRAMINGS: tuple[Framing, ...] = get_args(Framing)
TIERS: tuple[Tier, ...] = get_args(Tier)
FAMILIES: tuple[Family, ...] = get_args(Family)

AGREEMENT_KEYS: dict[Language, frozenset[str]] = {
    "en": frozenset({"sg", "pl"}),
    "ar": frozenset({"m_sg", "f_sg", "m_pl", "f_pl"}),
    "he": frozenset({"m_sg", "f_sg", "m_pl", "f_pl"}),
}

VICTIM_ROLE = "victim"
PERPETRATOR_ROLE = "perpetrator"
CASE_ROLE = "case"
ACTOR_ROLES = (VICTIM_ROLE, PERPETRATOR_ROLE)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RIGHTS_DIR = "rights"
ACTORS_DIR = "actors"
CASES_DIR = "cases"
TEMPLATES_DIR = "templates"
YAML_SUFFIX = ".yaml"
DATASET_NAME = "hrbias"

SAMPLE_ID_SEPARATOR = "."
PARAPHRASE_PREFIX = "p"
MIN_CASES_PER_SET = 2

Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]
Name = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


class DatasetError(ValueError):
    """The benchmark data is missing, malformed or inconsistent."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AgreedText(_Strict):
    """Text with the grammatical agreement key that ``{agree:...}`` slots use."""

    text: NonEmptyText
    agreement: NonEmptyText


class Right(_Strict):
    """A human right from an international instrument."""

    id: Slug
    instrument: NonEmptyText
    article: NonEmptyText
    description: dict[Language, NonEmptyText]


class Actor(_Strict):
    """A people, a state or an armed group, as written in one language.

    ``forms`` holds the texts templates insert with ``{victim.<form>}`` or
    ``{perpetrator.<form>}``. ``variants`` are other forms a model may write,
    and ``related_terms`` places or institutions tied to the actor. Redaction
    masks all of them.
    """

    id: Slug
    forms: dict[Name, AgreedText] = Field(min_length=1)
    variants: list[NonEmptyText] = Field(default_factory=list)
    related_terms: list[NonEmptyText] = Field(default_factory=list)

    def terms(self) -> list[str]:
        """Return every form text, variant and related term, without repeats, in order."""
        forms = [form.text for form in self.forms.values()]
        return list(dict.fromkeys([*forms, *self.variants, *self.related_terms]))


class CaseTerms(_Strict):
    """Terms that identify a case, by the side they belong to."""

    victim: list[NonEmptyText] = Field(default_factory=list)
    perpetrator: list[NonEmptyText] = Field(default_factory=list)
    context: list[NonEmptyText] = Field(default_factory=list)


class Case(_Strict):
    """One filling of a template's roles: a victim, maybe a perpetrator, and fields."""

    id: Slug
    victim: Slug
    perpetrator: Slug | None = None
    fields: dict[Name, NonEmptyText | AgreedText] = Field(default_factory=dict)
    related_terms: CaseTerms = Field(default_factory=CaseTerms)


class CaseSet(_Strict):
    """Every case of one case set, as written in one language."""

    id: Slug
    language: Language
    cases: list[Case]


class Template(_Strict):
    """One question in one language and framing, with its paraphrases.

    A paraphrase is one user turn, or a list of user turns sent in order.
    """

    id: Slug
    language: Language
    family: Family
    tier: Tier
    case_set: Slug
    framing: Framing
    right_ids: list[Slug] = Field(min_length=1)
    paraphrases: list[NonEmptyText | Annotated[list[NonEmptyText], Field(min_length=1)]] = Field(
        min_length=1
    )

    def paraphrase_turns(self) -> list[list[str]]:
        """Return every paraphrase as its list of user turns."""
        return [[p] if isinstance(p, str) else list(p) for p in self.paraphrases]

    def describe(self) -> str:
        """Name the template for error messages."""
        return f"template {self.id!r} ({self.language}, {self.framing})"


class BenchmarkData(_Strict):
    """All validated benchmark data."""

    rights: dict[str, Right]
    actors: dict[Language, dict[str, Actor]]
    case_sets: dict[Language, dict[str, CaseSet]]
    templates: list[Template]


def hrbias_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    languages: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    tiers: Sequence[str] | None = None,
) -> MemoryDataset:
    """Load the data in ``data_dir`` and return it as an Inspect dataset.

    ``languages``, ``families`` and ``tiers`` limit the samples to those
    values; ``None`` keeps all.
    """
    data = load_benchmark_data(data_dir)
    return MemoryDataset(
        samples=expand_samples(data, languages, families, tiers),
        name=DATASET_NAME,
        location=str(data_dir),
    )


def load_benchmark_data(data_dir: Path = DEFAULT_DATA_DIR) -> BenchmarkData:
    """Load and validate every rights, actors, cases and templates file in ``data_dir``.

    Every template is also filled for every case of its case set, so a slot
    that cannot be filled raises ``GrammarError`` here rather than during a
    run. Other problems raise ``DatasetError``.
    """
    rights = _load_rights(data_dir / RIGHTS_DIR)
    actors = _load_all_actors(data_dir / ACTORS_DIR)
    case_sets = _load_all_case_sets(data_dir / CASES_DIR, actors)
    templates = _load_all_templates(data_dir / TEMPLATES_DIR)
    data = BenchmarkData(rights=rights, actors=actors, case_sets=case_sets, templates=templates)
    _check_templates(data)
    expand_samples(data)
    return data


def expand_samples(
    data: BenchmarkData,
    languages: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    tiers: Sequence[str] | None = None,
) -> list[Sample]:
    """Expand every selected template into samples, one per paraphrase and case.

    Sample ids are ``<set_id>.<case_id>`` and ``set_id`` is
    ``<template_id>.<framing>.p<paraphrase_index>.<language>``, so ids are
    stable across runs.
    """
    selected_languages = _select(languages, LANGUAGES, "languages")
    selected_families = _select(families, FAMILIES, "families")
    selected_tiers = _select(tiers, TIERS, "tiers")
    samples: list[Sample] = []
    for template in data.templates:
        if (
            template.language not in selected_languages
            or template.family not in selected_families
            or template.tier not in selected_tiers
        ):
            continue

        cases = data.case_sets[template.language][template.case_set].cases
        actors = data.actors[template.language]
        for index, turns in enumerate(template.paraphrase_turns()):
            set_id = make_set_id(template, index)
            samples.extend(_expand_case(template, set_id, index, turns, c, actors) for c in cases)

    return samples


def make_set_id(template: Template, paraphrase_index: int) -> str:
    """Return the id of the counterfactual set for one template paraphrase."""
    return SAMPLE_ID_SEPARATOR.join(
        [
            template.id,
            template.framing,
            f"{PARAPHRASE_PREFIX}{paraphrase_index}",
            template.language,
        ]
    )


def case_terms(case: Case, actors: dict[str, Actor]) -> CaseTerms:
    """Return every term redaction masks for ``case``, by side.

    The victim side holds the victim actor's forms, variants and related
    terms plus the case's own victim terms; the perpetrator side the same for
    the perpetrator; the context side the case's context terms.
    """
    perpetrator = actors[case.perpetrator].terms() if case.perpetrator else []
    return CaseTerms(
        victim=_unique([*actors[case.victim].terms(), *case.related_terms.victim]),
        perpetrator=_unique([*perpetrator, *case.related_terms.perpetrator]),
        context=_unique(case.related_terms.context),
    )


def _expand_case(
    template: Template,
    set_id: str,
    paraphrase_index: int,
    turns: list[str],
    case: Case,
    actors: dict[str, Actor],
) -> Sample:
    values = _slot_values(case, actors)
    context = f"{template.describe()}, case {case.id!r}"
    filled = [
        fill_slots(turn, values, context=context, tracked_roles=ACTOR_ROLES) for turn in turns
    ]
    terms = case_terms(case, actors)
    return Sample(
        id=f"{set_id}{SAMPLE_ID_SEPARATOR}{case.id}",
        input=filled[0].text,
        metadata={
            "set_id": set_id,
            "template_id": template.id,
            "family": template.family,
            "tier": template.tier,
            "case_set": template.case_set,
            "case_id": case.id,
            "victim_id": case.victim,
            "perpetrator_id": case.perpetrator,
            "language": template.language,
            "framing": template.framing,
            "paraphrase_index": paraphrase_index,
            "turns": [f.text for f in filled],
            "right_ids": list(template.right_ids),
            "inserted_terms": _unique(term for f in filled for term in f.inserted_terms),
            "victim_terms": terms.victim,
            "perpetrator_terms": terms.perpetrator,
            "context_terms": terms.context,
        },
    )


def _slot_values(case: Case, actors: dict[str, Actor]) -> dict[str, dict[str, SlotValue]]:
    values = {
        VICTIM_ROLE: _actor_values(actors[case.victim]),
        CASE_ROLE: {
            name: SlotValue(field)
            if isinstance(field, str)
            else SlotValue(field.text, field.agreement)
            for name, field in case.fields.items()
        },
    }
    if case.perpetrator is not None:
        values[PERPETRATOR_ROLE] = _actor_values(actors[case.perpetrator])

    return values


def _actor_values(actor: Actor) -> dict[str, SlotValue]:
    return {name: SlotValue(form.text, form.agreement) for name, form in actor.forms.items()}


def _select(values: Sequence[str] | None, allowed: Sequence[str], what: str) -> set[str]:
    if values is None:
        return set(allowed)

    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise DatasetError(f"unknown {what} {unknown}, expected some of {list(allowed)}")

    return set(values)


def _load_rights(rights_dir: Path) -> dict[str, Right]:
    rights: list[Right] = []
    for path in _yaml_files(rights_dir):
        try:
            rights.extend(Right.model_validate(item) for item in _require_list(path, "rights"))
        except ValidationError as err:
            raise DatasetError(f"load rights {path}: {err}") from err

    _check_unique((r.id for r in rights), f"right id in {rights_dir}")
    for right in rights:
        missing = [lang for lang in LANGUAGES if lang not in right.description]
        if missing:
            raise DatasetError(f"right {right.id!r} has no description for {missing}")

    return {r.id: r for r in rights}


def _load_all_actors(actors_dir: Path) -> dict[Language, dict[str, Actor]]:
    _check_language_dirs(actors_dir)
    actors = {lang: _load_actors(actors_dir / lang, lang) for lang in LANGUAGES}
    _check_same_ids({lang: list(by_id) for lang, by_id in actors.items()}, "actor")
    return actors


def _load_actors(directory: Path, language: Language) -> dict[str, Actor]:
    actors: list[Actor] = []
    for path in _yaml_files(directory):
        try:
            loaded = [Actor.model_validate(item) for item in _require_list(path, "actors")]
        except ValidationError as err:
            raise DatasetError(f"load actors {path}: {err}") from err

        for actor in loaded:
            for name, form in actor.forms.items():
                _check_agreement(form, language, f"{path}: actor {actor.id!r} form {name!r}")

        actors.extend(loaded)

    _check_unique((a.id for a in actors), f"actor id in {directory}")
    return {a.id: a for a in actors}


def _load_all_case_sets(
    cases_dir: Path, actors: dict[Language, dict[str, Actor]]
) -> dict[Language, dict[str, CaseSet]]:
    _check_language_dirs(cases_dir)
    case_sets: dict[Language, dict[str, CaseSet]] = {}
    for language in LANGUAGES:
        loaded = [
            _load_case_set(path, language, actors[language])
            for path in _yaml_files(cases_dir / language)
        ]
        case_sets[language] = {s.id: s for s in loaded}

    _check_same_ids({lang: list(sets) for lang, sets in case_sets.items()}, "case set")
    for set_id in case_sets[LANGUAGES[0]]:
        _check_cases_match_across_languages({lang: case_sets[lang][set_id] for lang in LANGUAGES})

    return case_sets


def _load_case_set(path: Path, language: Language, actors: dict[str, Actor]) -> CaseSet:
    try:
        case_set = CaseSet(
            id=path.stem,
            language=language,
            cases=[Case.model_validate(item) for item in _require_list(path, "cases")],
        )
    except ValidationError as err:
        raise DatasetError(f"load cases {path}: {err}") from err

    if len(case_set.cases) < MIN_CASES_PER_SET:
        raise DatasetError(
            f"load cases {path}: case set has {len(case_set.cases)} case(s), "
            f"a counterfactual set needs at least {MIN_CASES_PER_SET}"
        )

    _check_unique((c.id for c in case_set.cases), f"case id in {path}")
    field_names = {c.id: sorted(c.fields) for c in case_set.cases}
    if len({tuple(names) for names in field_names.values()}) > 1:
        raise DatasetError(
            f"load cases {path}: all cases of a set need the same field names, got {field_names}"
        )

    for case in case_set.cases:
        for role, actor_id in ((VICTIM_ROLE, case.victim), (PERPETRATOR_ROLE, case.perpetrator)):
            if actor_id is not None and actor_id not in actors:
                raise DatasetError(
                    f"load cases {path}: case {case.id!r} {role} {actor_id!r} is unknown"
                )

        for name, field in case.fields.items():
            if isinstance(field, AgreedText):
                _check_agreement(field, language, f"{path}: case {case.id!r} field {name!r}")

    return case_set


def _load_all_templates(templates_dir: Path) -> list[Template]:
    _check_language_dirs(templates_dir)
    templates = [
        _load_template(path, language)
        for language in LANGUAGES
        for path in _yaml_files(templates_dir / language)
    ]
    _check_unique((t.describe() for t in templates), f"template in {templates_dir}")
    return templates


def _load_template(path: Path, language: Language) -> Template:
    raw = _read_yaml(path)
    if not isinstance(raw, dict):
        raise DatasetError(f"load template {path}: expected a mapping at the top level")

    if "language" in raw:
        raise DatasetError(
            f"load template {path}: remove the 'language' field, it comes from the directory"
        )

    try:
        return Template.model_validate({**raw, "language": language})
    except ValidationError as err:
        raise DatasetError(f"load template {path}: {err}") from err


def _check_templates(data: BenchmarkData) -> None:
    for template in data.templates:
        where = template.describe()
        unknown_rights = [r for r in template.right_ids if r not in data.rights]
        if unknown_rights:
            raise DatasetError(f"{where}: unknown rights {unknown_rights}")

        if template.case_set not in data.case_sets[template.language]:
            raise DatasetError(f"{where}: unknown case set {template.case_set!r}")

        turn_counts = sorted({len(turns) for turns in template.paraphrase_turns()})
        if len(turn_counts) > 1:
            raise DatasetError(
                f"{where}: all paraphrases need the same number of turns, got {turn_counts}"
            )

        if template.tier == "mirrored":
            for turns in template.paraphrase_turns():
                if any(CASE_ROLE in slot_roles(turn) for turn in turns):
                    raise DatasetError(
                        f"{where}: a mirrored template may not use {{{CASE_ROLE}.*}} slots"
                    )

    _check_templates_match_across_languages(data.templates)


def _check_templates_match_across_languages(templates: Sequence[Template]) -> None:
    by_key: dict[tuple[str, str], dict[str, Template]] = defaultdict(dict)
    for template in templates:
        by_key[(template.id, template.framing)][template.language] = template

    for (template_id, framing), by_language in by_key.items():
        where = f"template {template_id!r} ({framing})"
        missing = [lang for lang in LANGUAGES if lang not in by_language]
        if missing:
            raise DatasetError(f"{where} is missing for languages {missing}")

        shapes = {lang: _template_shape(t) for lang, t in by_language.items()}
        if len(set(shapes.values())) > 1:
            raise DatasetError(
                f"{where} differs across languages in family, tier, case set, rights, "
                f"paraphrase count or turn count: {shapes}"
            )


def _template_shape(template: Template) -> tuple[Any, ...]:
    return (
        template.family,
        template.tier,
        template.case_set,
        tuple(sorted(template.right_ids)),
        len(template.paraphrases),
        tuple(len(turns) for turns in template.paraphrase_turns()),
    )


def _check_cases_match_across_languages(case_sets: dict[Language, CaseSet]) -> None:
    roles = {
        lang: {c.id: (c.victim, c.perpetrator) for c in case_set.cases}
        for lang, case_set in case_sets.items()
    }
    set_id = case_sets[LANGUAGES[0]].id
    _check_same_ids(
        {lang: list(by_id) for lang, by_id in roles.items()}, "case", f" in set {set_id!r}"
    )
    for case_id, expected in roles[LANGUAGES[0]].items():
        differing = {
            lang: by_id[case_id] for lang, by_id in roles.items() if by_id[case_id] != expected
        }
        if differing:
            raise DatasetError(
                f"case {case_id!r} in set {set_id!r} has a different victim or perpetrator "
                f"across languages: {LANGUAGES[0]} {expected}, {differing}"
            )


def _check_agreement(value: AgreedText, language: Language, where: str) -> None:
    allowed = AGREEMENT_KEYS[language]
    if value.agreement not in allowed:
        raise DatasetError(
            f"{where} has agreement {value.agreement!r}, expected one of {sorted(allowed)}"
        )


def _check_same_ids(ids_by_language: dict[Language, list[str]], what: str, where: str = "") -> None:
    all_ids = sorted({i for ids in ids_by_language.values() for i in ids})
    for item_id in all_ids:
        missing = [lang for lang in LANGUAGES if item_id not in ids_by_language[lang]]
        if missing:
            raise DatasetError(f"{what} {item_id!r}{where} is missing for languages {missing}")


def _check_language_dirs(directory: Path) -> None:
    if not directory.is_dir():
        raise DatasetError(f"{directory}: directory not found")

    found = {p.name for p in directory.iterdir() if p.is_dir()}
    unknown = sorted(found - set(LANGUAGES))
    if unknown:
        raise DatasetError(f"{directory}: unknown languages {unknown}")

    missing = [lang for lang in LANGUAGES if lang not in found]
    if missing:
        raise DatasetError(f"{directory}: missing languages {missing}")


def _check_unique(values: Iterable[str], what: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise DatasetError(f"duplicate {what}: {value}")

        seen.add(value)


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise DatasetError(f"{directory}: directory not found")

    paths = sorted(directory.glob(f"*{YAML_SUFFIX}"))
    if not paths:
        raise DatasetError(f"{directory}: no {YAML_SUFFIX} files")

    return paths


def _require_list(path: Path, key: str) -> list[object]:
    raw = _read_yaml(path)
    if not isinstance(raw, dict) or not isinstance(raw.get(key), list):
        raise DatasetError(f"load {path}: expected a top-level {key!r} list")

    return raw[key]


def _read_yaml(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise DatasetError(f"read {path}: {err}") from err

    if text.partition("\n")[0].rstrip() != CANARY_LINE:
        raise DatasetError(f"read {path}: first line must be the canary line {CANARY_LINE!r}")

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise DatasetError(f"parse {path}: {err}") from err


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
