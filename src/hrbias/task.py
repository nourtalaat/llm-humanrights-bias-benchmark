"""Inspect entry point for the LLM human rights bias benchmark.

Run it with any model, for example::

    uv run inspect eval src/hrbias/task.py --model openai/gpt-4o-mini

The judge is the model bound to the ``grader`` role
(``--model-role grader=<model>``), or the pinned default judge otherwise.
"""

from inspect_ai import Task, task
from inspect_ai.solver import generate

from hrbias import BENCHMARK_VERSION
from hrbias.dataset import hrbias_dataset
from hrbias.scorer import hrbias_scorer
from hrbias.versions import run_metadata

DEFAULT_EPOCHS = 3
LANGUAGE_SEPARATOR = ","


@task
def hrbias(languages: str | list[str] | None = None, leak_check: bool = False) -> Task:
    """Ask every counterfactual prompt, then code each answer with the blind judge.

    ``languages`` limits the run to some of ``en``, ``ar`` and ``he``, as a
    list or a comma-separated string (``-T languages=ar,he``). ``leak_check``
    also asks the judge to guess the group from each redacted answer. The
    model's own default temperature is used, and every prompt is asked
    ``DEFAULT_EPOCHS`` times.
    """
    if isinstance(languages, str):
        languages = [lang.strip() for lang in languages.split(LANGUAGE_SEPARATOR) if lang.strip()]

    return Task(
        dataset=hrbias_dataset(languages=languages),
        solver=generate(),
        scorer=hrbias_scorer(leak_check=leak_check),
        epochs=DEFAULT_EPOCHS,
        version=BENCHMARK_VERSION,
        metadata=run_metadata(),
    )
