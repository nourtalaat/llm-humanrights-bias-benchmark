import shutil
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from inspect_ai.dataset import Sample

from hrbias import CANARY_LINE
from hrbias.dataset import (
    DEFAULT_DATA_DIR,
    LANGUAGES,
    BenchmarkData,
    DatasetError,
    expand_samples,
    hrbias_dataset,
    load_benchmark_data,
)
from hrbias.grammar import GrammarError

METADATA_KEYS = {
    "set_id",
    "template_id",
    "right_id",
    "category",
    "group_id",
    "language",
    "framing",
    "paraphrase_index",
    "inserted_terms",
    "related_terms",
}


@pytest.fixture(scope="module")
def data() -> BenchmarkData:
    return load_benchmark_data()


@pytest.fixture(scope="module")
def samples(data: BenchmarkData) -> list[Sample]:
    return expand_samples(data)


def test_set_size_equals_groups_in_category(data: BenchmarkData, samples: list[Sample]) -> None:
    groups_by_set: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        groups_by_set[_meta(sample)["set_id"]].append(_meta(sample)["group_id"])

    for sample in samples:
        meta = _meta(sample)
        expected = [g.id for g in data.groups[meta["language"]] if g.category == meta["category"]]
        assert groups_by_set[meta["set_id"]] == expected


def test_sample_count_covers_every_paraphrase_and_group(
    data: BenchmarkData, samples: list[Sample]
) -> None:
    expected = sum(
        len(t.paraphrases) * sum(1 for g in data.groups[t.language] if g.category == t.category)
        for t in data.templates
    )

    assert len(samples) == expected


def test_metadata_has_exactly_the_shared_keys(samples: list[Sample]) -> None:
    for sample in samples:
        assert set(_meta(sample)) == METADATA_KEYS


@pytest.mark.parametrize(
    ("sample_id", "expected_metadata", "expected_input"),
    [
        pytest.param(
            "assembly-permit.neutral.p0.en.kestrish",
            {
                "set_id": "assembly-permit.neutral.p0.en",
                "template_id": "assembly-permit",
                "right_id": "peaceful-assembly",
                "category": "nationality",
                "group_id": "kestrish",
                "language": "en",
                "framing": "neutral",
                "paraphrase_index": 0,
                "inserted_terms": ["Kestrish"],
                "related_terms": ["Kestria", "Brannock"],
            },
            "A group of Kestrish residents has applied for a permit",
            id="en",
        ),
        pytest.param(
            "travel-restrictions.sympathetic.p1.ar.sisters-of-miral",
            {
                "set_id": "travel-restrictions.sympathetic.p1.ar",
                "template_id": "travel-restrictions",
                "right_id": "freedom-of-movement",
                "category": "fictional-order",
                "group_id": "sisters-of-miral",
                "language": "ar",
                "framing": "sympathetic",
                "paraphrase_index": 1,
                "inserted_terms": ["أخوات ميرال"],
                "related_terms": ["دير ميرال"],
            },
            "قضت أخوات ميرال عقودًا في مساعدة الجيران",
            id="ar",
        ),
        pytest.param(
            "travel-restrictions.sympathetic.p0.he.selvin-circle",
            {
                "set_id": "travel-restrictions.sympathetic.p0.he",
                "template_id": "travel-restrictions",
                "right_id": "freedom-of-movement",
                "category": "fictional-order",
                "group_id": "selvin-circle",
                "language": "he",
                "framing": "sympathetic",
                "paraphrase_index": 0,
                "inserted_terms": ["חוג סלווין"],
                "related_terms": ["אולם סלווין"],
            },
            "חוג סלווין ידוע בהפעלת בנקי מזון ובטיפול בקשישים באזורו",
            id="he",
        ),
    ],
)
def test_sample_metadata_and_input(
    samples: list[Sample],
    sample_id: str,
    expected_metadata: dict[str, object],
    expected_input: str,
) -> None:
    sample = next(s for s in samples if s.id == sample_id)

    assert sample.metadata == expected_metadata
    assert isinstance(sample.input, str)
    assert sample.input.startswith(expected_input)


def test_sample_ids_are_unique_and_stable(samples: list[Sample]) -> None:
    ids = [s.id for s in samples]
    reloaded_ids = [s.id for s in expand_samples(load_benchmark_data())]

    assert len(set(ids)) == len(ids)
    assert ids == reloaded_ids
    for sample in samples:
        meta = _meta(sample)
        assert sample.id == f"{meta['set_id']}.{meta['group_id']}"


def test_inserted_terms_appear_in_input(samples: list[Sample]) -> None:
    for sample in samples:
        terms = _meta(sample)["inserted_terms"]
        assert terms
        for term in terms:
            assert term in sample.input


@pytest.mark.parametrize(
    "languages",
    [pytest.param(["ar"], id="ar"), pytest.param(["en", "he"], id="en-he")],
)
def test_language_filter(data: BenchmarkData, languages: list[str]) -> None:
    filtered = expand_samples(data, languages)

    assert filtered
    assert {_meta(s)["language"] for s in filtered} == set(languages)


def test_unknown_language_filter_raises(data: BenchmarkData) -> None:
    with pytest.raises(DatasetError, match="unknown languages"):
        expand_samples(data, ["fr"])


def test_hrbias_dataset_returns_inspect_dataset(samples: list[Sample]) -> None:
    dataset = hrbias_dataset(languages=["he"])

    assert len(dataset) == sum(1 for s in samples if _meta(s)["language"] == "he")


def test_every_data_file_starts_with_canary() -> None:
    paths = sorted(DEFAULT_DATA_DIR.rglob("*.yaml"))

    assert paths
    for path in paths:
        assert path.read_text(encoding="utf-8").splitlines()[0] == CANARY_LINE, path


def test_every_language_has_group_and_template_files() -> None:
    for language in LANGUAGES:
        assert (DEFAULT_DATA_DIR / "groups" / f"{language}.yaml").is_file()
        assert list((DEFAULT_DATA_DIR / "templates" / language).glob("*.yaml"))


def _set_template_right(data_dir: Path) -> None:
    _edit(data_dir / "templates/en/assembly-permit.yaml", lambda d: d.update(right_id="no-such"))


def _set_template_category(data_dir: Path) -> None:
    _edit(data_dir / "templates/he/assembly-permit.yaml", lambda d: d.update(category="no-such"))


def _drop_hebrew_group(data_dir: Path) -> None:
    _edit(
        data_dir / "groups/he.yaml",
        lambda d: d.update(groups=[g for g in d["groups"] if g["id"] != "ombaran"]),
    )


def _set_english_agreement_to_gendered(data_dir: Path) -> None:
    _edit(data_dir / "groups/en.yaml", lambda d: d["groups"][0].update(agreement="m_pl"))


def _drop_rights_description_language(data_dir: Path) -> None:
    _edit(data_dir / "rights.yaml", lambda d: d["rights"][0]["description"].pop("ar"))


def _add_language_field_to_template(data_dir: Path) -> None:
    _edit(data_dir / "templates/ar/assembly-permit.yaml", lambda d: d.update(language="ar"))


def _duplicate_template(data_dir: Path) -> None:
    source = data_dir / "templates/en/assembly-permit.yaml"
    shutil.copy(source, source.with_name("assembly-permit-copy.yaml"))


def _leave_one_group_in_category(data_dir: Path) -> None:
    removed = {"wardens-of-esk", "selvin-circle"}
    for language in LANGUAGES:
        _edit(
            data_dir / f"groups/{language}.yaml",
            lambda d: d.update(groups=[g for g in d["groups"] if g["id"] not in removed]),
        )


def _remove_canary(data_dir: Path) -> None:
    path = data_dir / "groups/ar.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace(CANARY_LINE, ""), encoding="utf-8")


def _set_unknown_form_slot(data_dir: Path) -> None:
    _edit(
        data_dir / "templates/en/assembly-permit.yaml",
        lambda d: d.update(paraphrases=["Do {group.singular} have this right?"]),
    )


def _set_gender_only_agreement_slot(data_dir: Path) -> None:
    _edit(
        data_dir / "templates/ar/travel-restrictions.yaml",
        lambda d: d.update(paraphrases=["{group.name} {agree:m_pl=معروفون|f_pl=معروفات}"]),
    )


@pytest.mark.parametrize(
    ("break_data", "error", "message"),
    [
        pytest.param(_set_template_right, DatasetError, "unknown right 'no-such'", id="right"),
        pytest.param(
            _set_template_category, DatasetError, "unknown category 'no-such'", id="category"
        ),
        pytest.param(
            _drop_hebrew_group,
            DatasetError,
            r"group 'ombaran' has no forms for languages \['he'\]",
            id="missing-language-forms",
        ),
        pytest.param(
            _set_english_agreement_to_gendered,
            DatasetError,
            "has agreement 'm_pl'",
            id="agreement-key-for-language",
        ),
        pytest.param(
            _drop_rights_description_language,
            DatasetError,
            r"has no description for \['ar'\]",
            id="right-description-language",
        ),
        pytest.param(
            _add_language_field_to_template,
            DatasetError,
            "remove the 'language' field",
            id="template-language-field",
        ),
        pytest.param(
            _duplicate_template, DatasetError, "duplicate template", id="duplicate-template"
        ),
        pytest.param(
            _leave_one_group_in_category,
            DatasetError,
            "category 'fictional-order' has 1 group",
            id="single-group-category",
        ),
        pytest.param(_remove_canary, DatasetError, "canary", id="missing-canary"),
        pytest.param(
            _set_unknown_form_slot,
            GrammarError,
            "template 'assembly-permit', group 'valdorian': unknown form 'singular'",
            id="unknown-form-slot",
        ),
        pytest.param(
            _set_gender_only_agreement_slot,
            GrammarError,
            "template 'travel-restrictions', group 'selvin-circle': "
            "no option for agreement key 'f_sg'",
            id="missing-agreement-key",
        ),
    ],
)
def test_invalid_data_raises(
    tmp_path: Path,
    break_data: Callable[[Path], None],
    error: type[Exception],
    message: str,
) -> None:
    data_dir = tmp_path / "data"
    shutil.copytree(DEFAULT_DATA_DIR, data_dir)
    break_data(data_dir)

    with pytest.raises(error, match=message):
        load_benchmark_data(data_dir)


def _meta(sample: Sample) -> dict[str, Any]:
    assert sample.metadata is not None
    return sample.metadata


def _edit(path: Path, change: Callable[[dict[str, Any]], object]) -> None:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(content)
    dumped = yaml.safe_dump(content, allow_unicode=True, sort_keys=False)
    path.write_text(f"{CANARY_LINE}\n{dumped}", encoding="utf-8")
