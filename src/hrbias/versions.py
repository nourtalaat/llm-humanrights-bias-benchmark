"""The versions a run is pinned to, stored in every Inspect log.

Two runs are comparable only when all of these match.
"""

from hrbias import BENCHMARK_VERSION
from hrbias.codebook import CODEBOOK_VERSION
from hrbias.judge import DEFAULT_JUDGE_HF_REPO, DEFAULT_JUDGE_HF_REVISION


def run_metadata() -> dict[str, str]:
    """Return the version metadata the task stores and the report checks."""
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "codebook_version": CODEBOOK_VERSION,
        "judge_hf_repo": DEFAULT_JUDGE_HF_REPO,
        "judge_hf_revision": DEFAULT_JUDGE_HF_REVISION,
    }
