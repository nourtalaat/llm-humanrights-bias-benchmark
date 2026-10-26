"""Solver that plays a scripted conversation with the model under test."""

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

TURNS_KEY = "turns"


@solver
def scripted_conversation() -> Solver:
    """Send the scripted user turns in order, generating an answer after each one.

    The first turn is the sample input. Metadata ``turns`` lists every user
    turn, the first included; the later ones are sent whatever the model
    answered before. A sample without ``turns``, or with a single turn, is
    asked once, exactly as with ``generate()``.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        follow_ups = list(state.metadata.get(TURNS_KEY) or [])[1:]
        state = await generate(state)
        for turn in follow_ups:
            if state.completed:
                break

            state.messages.append(ChatMessageUser(content=turn))
            state = await generate(state)

        return state

    return solve
