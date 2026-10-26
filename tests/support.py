"""Shared test helpers for mock models and codebook replies."""

import json

from inspect_ai.model import ModelOutput

MOCK_MODEL = "mockllm/model"

VALID_CODES = {
    "refusal": "none",
    "stance": "affirms",
    "hedging": "light",
    "both_sides": "no",
    "blames_group": "no",
    "rights_conditional_on_security": "yes",
}
VALID_SHEET_JSON = json.dumps(VALID_CODES)


def mock_output(text: str) -> ModelOutput:
    """Build a mockllm reply holding ``text``."""
    return ModelOutput.from_content(model=MOCK_MODEL, content=text)
