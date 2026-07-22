# Contributing

Contributions are welcome when they keep the project reproducible and within its offline, permitted-environment boundary.

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run pytest
uv run cortex-arena doctor
```

Please add synthetic unit tests for perception and planning changes. Gameplay changes should also include a seeded benchmark report and the opponent revision used for comparison.

Do not submit code for anti-cheat bypass, process-memory reading, input injection into online games, arbitrary model-generated Python execution, or account/reward automation.
