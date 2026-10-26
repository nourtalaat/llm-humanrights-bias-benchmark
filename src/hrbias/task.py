"""Inspect entry point for the LLM human rights bias benchmark.

Run it with any model, for example::

    uv run inspect eval src/hrbias/task.py --model openai/gpt-4o-mini

The judge is the model bound to the ``grader`` role
(``--model-role grader=<model>``), or the pinned default judge otherwise.
"""

from pathlib import Path

from inspect_ai import Task, task

from hrbias import BENCHMARK_VERSION
from hrbias.dataset import DEFAULT_DATA_DIR, hrbias_dataset
from hrbias.scorer import hrbias_scorer
from hrbias.solver import scripted_conversation
from hrbias.versions import run_metadata

DEFAULT_EPOCHS = 3
LANGUAGE_SEPARATOR = ","


@task
def hrbias(
    languages: str | list[str] | None = None,
    leak_check: bool = False,
    data_dir: str | None = None,
) -> Task:
    """Ask every counterfactual prompt, then code each answer with the blind judge.

    ``languages`` limits the run to some of ``en``, ``ar`` and ``he``, as a
    list or a comma-separated string (``-T languages=ar,he``). ``leak_check``
    also asks the judge to guess the group from each redacted answer.
    ``data_dir`` reads the benchmark data from another directory than
    ``data/``. The
    model's own default temperature is used, and every prompt is asked
    ``DEFAULT_EPOCHS`` times.
    """
    if isinstance(languages, str):
        languages = [lang.strip() for lang in languages.split(LANGUAGE_SEPARATOR) if lang.strip()]

    return Task(
        dataset=hrbias_dataset(Path(data_dir) if data_dir else DEFAULT_DATA_DIR, languages),
        solver=scripted_conversation(),
        scorer=hrbias_scorer(data_dir=data_dir, leak_check=leak_check),
        epochs=DEFAULT_EPOCHS,
        version=BENCHMARK_VERSION,
        metadata=run_metadata(),
    )
