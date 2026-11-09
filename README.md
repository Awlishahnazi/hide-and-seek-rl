# Hide-and-Seek RL

Lightweight 2D grid-world hide-and-seek, trained with multi-agent PPO (MAPPO). The only reward is whether hiders are seen. Strategies are not hard-coded; the project studies which ones emerge, and in what order, as boxes, vision radius, and team size change.

Full specification: [PROJECT_SPEC.md](PROJECT_SPEC.md). Implementation follows Section 12 one step at a time.

## Status

Step 1 of 17: repository skeleton only. Environment, training, and the demo are not implemented yet.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -e .` installs the `hideseek` package and its runtime dependencies. The `dev` extra adds pytest.
