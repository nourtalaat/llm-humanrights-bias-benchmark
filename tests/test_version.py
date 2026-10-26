import re

from hrbias import BENCHMARK_VERSION

SEMVER_PATTERN = r"\d+\.\d+\.\d+"


def test_benchmark_version_is_semver() -> None:
    assert re.fullmatch(SEMVER_PATTERN, BENCHMARK_VERSION)
