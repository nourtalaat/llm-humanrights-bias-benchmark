# LLM human rights bias benchmark

This benchmark checks whether a language model treats groups of people
differently when asked about the same human right. Each question is asked
several times, changing only the group it is about (or the framing of the
question), and the answers are compared with each other. A model counts as
biased on a question when its answer changes with the group: for example it
affirms the right to peaceful assembly for one group but hedges or refuses for
another. There is no single "correct" answer; only consistency across groups
is measured. Questions are written natively in English, Arabic and Hebrew.

## Status

Skeleton with placeholder data. The groups and scenarios in `data/` are
fictional examples that exist only to exercise the code. Results produced with
them say nothing about any real group.

## Running the tests

Requires [uv](https://docs.astral.sh/uv/).

```sh
make ci       # lint, format check and tests
make test     # tests only
make format   # apply ruff formatting and safe lint fixes
```
