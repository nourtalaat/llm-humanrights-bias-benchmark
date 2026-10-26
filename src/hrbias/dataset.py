"""Load the benchmark YAML data and expand it into counterfactual Inspect samples.

A counterfactual set is one template paraphrase, in one framing and language,
filled with every group of the template's category. Answers are only compared
within a set, so each sample records its ``set_id`` in its metadata.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Literal, get_args

import yaml
from inspect_ai.dataset import MemoryDataset, Sample
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from hrbias import CANARY_LINE
from hrbias.grammar import fill_slots

Language = Literal["en", "ar", "he"]
Framing = Literal["neutral", "sympathetic", "hostile"]

LANGUAGES: tuple[Language, ...] = get_args(Language)
FRAMINGS: tuple[Framing, ...] = get_args(Framing)

AGREEMENT_KEYS: dict[Language, frozenset[str]] = {
    "en": frozenset({"sg", "pl"}),
    "ar": frozenset({"m_sg", "f_sg", "m_pl", "f_pl"}),
    "he": frozenset({"m_sg", "f_sg", "m_pl", "f_pl"}),
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RIGHTS_FILE = "rights.yaml"
GROUPS_DIR = "groups"
TEMPLATES_DIR = "templates"
YAML_SUFFIX = ".yaml"
DATASET_NAME = "hrbias"

UDHR_FIRST_ARTICLE = 1
UDHR_LAST_ARTICLE = 30

SAMPLE_ID_SEPARATOR = "."
MIN_GROUPS_PER_SET = 2

Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


class DatasetError(ValueError):
    """The benchmark data is missing, malformed or inconsistent."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Right(_Strict):
    """A human right the benchmark asks about."""

    id: Slug
    udhr_article: int = Field(ge=UDHR_FIRST_ARTICLE, le=UDHR_LAST_ARTICLE)
    description: dict[Language, NonEmptyText]


class Group(_Strict):
    """One group as written in one language.

    ``forms`` holds the surface forms templates insert with ``{group.<form>}``.
    ``agreement`` is the grammatical gender and number key that picks the text
    of ``{agree:...}`` slots. ``variants`` are inflected or other forms of the
    group's name that a model is likely to write but templates do not insert
    (other cases, singular, feminine). ``related_terms`` are words tied to the
    group (places, institutions). Redaction masks all of them.
    """

    id: Slug
    category: Slug
    agreement: NonEmptyText
    forms: dict[NonEmptyText, NonEmptyText] = Field(min_length=1)
    variants: list[NonEmptyText] = Field(default_factory=list)
    related_terms: list[NonEmptyText] = Field(default_factory=list)

    def group_terms(self) -> list[str]:
        """Return every form value and variant, without repeats, in order."""
        return list(dict.fromkeys([*self.forms.values(), *self.variants]))


class Template(_Strict):
    """One scenario in one language and framing, with its paraphrases."""

    id: Slug
    language: Language
    right_id: Slug
    category: Slug
    framing: Framing
    paraphrases: list[NonEmptyText] = Field(min_length=1)


class BenchmarkData(_Strict):
    """All validated benchmark data."""

    rights: dict[str, Right]
    groups: dict[Language, list[Group]]
    templates: list[Template]


def hrbias_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    languages: Sequence[str] | None = None,
) -> MemoryDataset:
    """Load the data in ``data_dir`` and return it as an Inspect dataset.

    ``languages`` limits the samples to those languages; ``None`` keeps all.
    """
    data = load_benchmark_data(data_dir)
    return MemoryDataset(
        samples=expand_samples(data, languages),
        name=DATASET_NAME,
        location=str(data_dir),
    )


def load_benchmark_data(data_dir: Path = DEFAULT_DATA_DIR) -> BenchmarkData:
    """Load and validate every rights, groups and templates file in ``data_dir``.

    Every template is also filled for every group of its category, so a slot
    that cannot be filled raises ``GrammarError`` here rather than during a run.
    Other problems raise ``DatasetError``.
    """
    rights = _load_rights(data_dir / RIGHTS_FILE)
    groups = _load_all_groups(data_dir / GROUPS_DIR)
    templates = _load_all_templates(data_dir / TEMPLATES_DIR)
    data = BenchmarkData(rights=rights, groups=groups, templates=templates)
    _check_templates(data)
    expand_samples(data)
    return data


def expand_samples(data: BenchmarkData, languages: Sequence[str] | None = None) -> list[Sample]:
    """Expand every template into samples, one per paraphrase and group.

    Sample ids are ``<set_id>.<group_id>`` and ``set_id`` is
    ``<template_id>.<framing>.p<paraphrase_index>.<language>``, so ids are
    stable across runs.
    """
    selected = _select_languages(languages)
    samples: list[Sample] = []
    for template in data.templates:
        if template.language not in selected:
            continue

        groups = _category_groups(data, template.language, template.category)
        for index, paraphrase in enumerate(template.paraphrases):
            set_id = make_set_id(template, index)
            samples.extend(_expand_set(template, set_id, index, paraphrase, groups))

    return samples


def make_set_id(template: Template, paraphrase_index: int) -> str:
    """Return the id of the counterfactual set for one template paraphrase."""
    return SAMPLE_ID_SEPARATOR.join(
        [template.id, template.framing, f"p{paraphrase_index}", template.language]
    )


def _expand_set(
    template: Template,
    set_id: str,
    paraphrase_index: int,
    paraphrase: str,
    groups: Iterable[Group],
) -> list[Sample]:
    samples = []
    for group in groups:
        filled = fill_slots(
            paraphrase,
            group.forms,
            group.agreement,
            template_id=template.id,
            group_id=group.id,
        )
        samples.append(
            Sample(
                id=f"{set_id}{SAMPLE_ID_SEPARATOR}{group.id}",
                input=filled.text,
                metadata={
                    "set_id": set_id,
                    "template_id": template.id,
                    "right_id": template.right_id,
                    "category": template.category,
                    "group_id": group.id,
                    "language": template.language,
                    "framing": template.framing,
                    "paraphrase_index": paraphrase_index,
                    "inserted_terms": filled.inserted_terms,
                    "group_terms": group.group_terms(),
                    "related_terms": list(group.related_terms),
                },
            )
        )

    return samples


def _category_groups(data: BenchmarkData, language: Language, category: str) -> list[Group]:
    return [g for g in data.groups[language] if g.category == category]


def _select_languages(languages: Sequence[str] | None) -> set[str]:
    if languages is None:
        return set(LANGUAGES)

    unknown = sorted(set(languages) - set(LANGUAGES))
    if unknown:
        raise DatasetError(f"unknown languages {unknown}, expected some of {list(LANGUAGES)}")

    return set(languages)


def _load_rights(path: Path) -> dict[str, Right]:
    raw = _read_yaml(path)
    try:
        rights = [Right.model_validate(item) for item in _require_list(raw, "rights", path)]
    except ValidationError as err:
        raise DatasetError(f"load rights {path}: {err}") from err

    _check_unique((r.id for r in rights), f"right id in {path}")
    for right in rights:
        missing = [lang for lang in LANGUAGES if lang not in right.description]
        if missing:
            raise DatasetError(
                f"load rights {path}: right {right.id!r} has no description for {missing}"
            )

    return {r.id: r for r in rights}


def _load_all_groups(groups_dir: Path) -> dict[Language, list[Group]]:
    _check_language_entries(groups_dir, (p.stem for p in groups_dir.glob(f"*{YAML_SUFFIX}")))
    groups = {lang: _load_groups(groups_dir / f"{lang}{YAML_SUFFIX}", lang) for lang in LANGUAGES}
    _check_groups_match_across_languages(groups)
    return groups


def _load_groups(path: Path, language: Language) -> list[Group]:
    raw = _read_yaml(path)
    try:
        groups = [Group.model_validate(item) for item in _require_list(raw, "groups", path)]
    except ValidationError as err:
        raise DatasetError(f"load groups {path}: {err}") from err

    _check_unique((g.id for g in groups), f"group id in {path}")
    allowed = AGREEMENT_KEYS[language]
    for group in groups:
        if group.agreement not in allowed:
            raise DatasetError(
                f"load groups {path}: group {group.id!r} has agreement {group.agreement!r}, "
                f"expected one of {sorted(allowed)}"
            )

    return groups


def _load_all_templates(templates_dir: Path) -> list[Template]:
    if not templates_dir.is_dir():
        raise DatasetError(f"{templates_dir}: templates directory not found")

    _check_language_entries(templates_dir, (p.name for p in templates_dir.iterdir() if p.is_dir()))
    templates = []
    for language in LANGUAGES:
        for path in sorted((templates_dir / language).glob(f"*{YAML_SUFFIX}")):
            templates.append(_load_template(path, language))

    _check_unique(
        (f"{t.id} ({t.framing}, {t.language})" for t in templates),
        f"template in {templates_dir}",
    )
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
    categories = {g.category for groups in data.groups.values() for g in groups}
    for template in data.templates:
        where = f"template {template.id!r} ({template.language}, {template.framing})"
        if template.right_id not in data.rights:
            raise DatasetError(f"{where}: unknown right {template.right_id!r}")

        if template.category not in categories:
            raise DatasetError(f"{where}: unknown category {template.category!r}")

        group_count = len(_category_groups(data, template.language, template.category))
        if group_count < MIN_GROUPS_PER_SET:
            raise DatasetError(
                f"{where}: category {template.category!r} has {group_count} group(s) in "
                f"{template.language}, a counterfactual set needs at least {MIN_GROUPS_PER_SET}"
            )


def _check_groups_match_across_languages(groups: dict[Language, list[Group]]) -> None:
    categories_by_language = {
        lang: {g.id: g.category for g in lang_groups} for lang, lang_groups in groups.items()
    }
    all_ids = sorted({gid for by_id in categories_by_language.values() for gid in by_id})
    for group_id in all_ids:
        missing = [lang for lang in LANGUAGES if group_id not in categories_by_language[lang]]
        if missing:
            raise DatasetError(f"group {group_id!r} has no forms for languages {missing}")

        categories = {categories_by_language[lang][group_id] for lang in LANGUAGES}
        if len(categories) > 1:
            raise DatasetError(
                f"group {group_id!r} has different categories across languages: "
                f"{sorted(categories)}"
            )


def _check_language_entries(directory: Path, names: Iterable[str]) -> None:
    found = set(names)
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


def _require_list(raw: object, key: str, path: Path) -> list[object]:
    if not isinstance(raw, dict) or not isinstance(raw.get(key), list):
        raise DatasetError(f"load {path}: expected a top-level {key!r} list")

    return raw[key]
