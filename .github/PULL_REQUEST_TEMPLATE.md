## What and why

<!-- The change, and the problem it solves. Link the issue if there is one. -->

## Checks

- [ ] `uv run ruff format . && uv run ruff check . && uv run mypy src` pass
- [ ] `uv run pytest --cov` passes, with new or changed behaviour tested
- [ ] Statistical code has a known-value test with the arithmetic in its comment
- [ ] A change to the gate's behaviour has an ADR in `docs/adr/`
- [ ] `CHANGELOG.md` has an entry under the next version
