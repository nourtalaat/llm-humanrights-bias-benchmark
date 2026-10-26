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
LIST_SEPARATOR = ","


@task
def hrbias(
    languages: str | list[str] | None = None,
    families: str | list[str] | None = None,
    tiers: str | list[str] | None = None,
    leak_check: bool = False,
    data_dir: str | None = None,
) -> Task:
    """Play every counterfactual conversation, then code it with the blind judge.

    ``languages`` (``en``, ``ar``, ``he``), ``families`` and ``tiers``
    (``real_event``, ``mirrored``) limit the run, each as a list or a
    comma-separated string (``-T languages=ar,he``). ``leak_check`` also asks
    the judge to guess the case from each redacted conversation. ``data_dir``
    reads the benchmark data from another directory than ``data/``. The
    model's own default temperature is used, and every conversation is played
    ``DEFAULT_EPOCHS`` times.
    """
    return Task(
        dataset=hrbias_dataset(
            Path(data_dir) if data_dir else DEFAULT_DATA_DIR,
            languages=_as_list(languages),
            families=_as_list(families),
            tiers=_as_list(tiers),
        ),
        solver=scripted_conversation(),
        scorer=hrbias_scorer(data_dir=data_dir, leak_check=leak_check),
        epochs=DEFAULT_EPOCHS,
        version=BENCHMARK_VERSION,
        metadata=run_metadata(),
    )


def _as_list(values: str | list[str] | None) -> list[str] | None:
    if isinstance(values, str):
        return [value.strip() for value in values.split(LIST_SEPARATOR) if value.strip()]

    return values
