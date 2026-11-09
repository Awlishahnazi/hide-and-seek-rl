# Hide-and-Seek RL: Emergent Strategies in a Lightweight 2D Multi-Agent Environment

> Full project specification. Paste this file into the root of the repo (or into Cursor as context) and implement it step by step following Section 12 (commit plan).

---

## 1. Overview

Build, from scratch, a lightweight 2D grid-world **hide-and-seek** environment and train two teams of agents (hiders vs. seekers) with **multi-agent reinforcement learning (MAPPO)**. The only reward signal is "were the hiders seen or not". No strategy is hard-coded. The goal is to study **which strategies emerge, in what order, and how environment parameters change that**.

This is inspired by OpenAI's hide-and-seek work (2019). Our contribution is NOT the idea itself. It is:

1. A **small, fast, fully reproducible** 2D version that trains on a laptop / free GPU.
2. A **concrete research question** with controlled ablations and multi-seed statistics (Section 2).
3. A **graphical, interactive demo** that makes the results easy to show (Section 11).

Target audiences: graduate admissions committees and professors (research rigor), and employers (clean engineering + visual demo).

## 2. Research Question

> How do the number of boxes, the vision radius, and the team size determine **which strategies emerge and in what order** in a 2D hide-and-seek game trained only with a team-based visibility reward?

Hypotheses (to be tested, not assumed):

- H1: With 0 boxes, hiders can only learn running/hiding behind static walls. With >= 1 box, door-blocking can appear.
- H2: Larger vision radius for seekers shifts the equilibrium toward more complex hider strategies, or toward hider collapse.
- H3: Larger teams enable coordinated strategies (e.g. one agent blocks while another hides), but are harder to train.

Negative or partial results are acceptable as long as they are reported honestly with enough seeds.

## 3. Environment Specification

### 3.1 Map
- Grid size: default `15 x 15` (configurable).
- Border is wall. Inside: a central wall that splits the map into two areas, with **1 or 2 door gaps** (width 1 cell). Optionally one enclosed room with a single door.
- Map layout is configurable via a small config / preset function. Add a procedural option later (Section 12, optional step).

### 3.2 Entities
- **Hiders**: default 2. **Seekers**: default 2. (Configurable `n_hiders`, `n_seekers`.)
- **Boxes**: default 2, each occupies one cell. A box can be pushed.
- Boxes can be **locked** by hiders only. A locked box cannot be pushed. Only hiders can unlock.

### 3.3 Actions (discrete, per agent)
| id | action | notes |
|----|--------|-------|
| 0 | no-op | |
| 1 | move up | |
| 2 | move down | |
| 3 | move left | |
| 4 | move right | |
| 5 | toggle lock | hiders only: lock/unlock an adjacent box. For seekers this is a no-op. |

Movement rules:
- Moving into a wall or another agent: blocked (stay).
- Moving into an unlocked box: push it one cell in the same direction if the cell behind it is free (no wall, agent, or other box). Otherwise blocked.
- Moving into a locked box: blocked.
- Conflicts (two agents moving into the same cell in the same step): resolve deterministically (e.g. fixed agent order with seeded random tie-break). Document this in code.

### 3.4 Visibility (line of sight)
- An agent `A` sees entity at cell `c` iff Euclidean distance `<= vision_radius` AND the grid line (Bresenham) from `A` to `c` is not blocked by a wall or a box.
- Default `vision_radius = 5` (configurable).
- Agents never block line of sight (only walls and boxes do).
- (Optional later) Replace omnidirectional vision with a facing direction + cone.
- A hider is **"seen"** at a step if at least one seeker sees it.

### 3.5 Episode structure
- Episode length: `T = 200` steps (configurable).
- **Preparation phase**: first `T_prep = 40` steps, seekers are frozen (forced no-op) and rewards are 0. Hiders can move boxes / hide.
- After the preparation phase, rewards are active and seekers move.

### 3.6 Reward (team-based, zero-sum, no strategy shaping)
After the preparation phase, at each step:
- If **no** hider is seen by any seeker: each hider gets `+1`, each seeker gets `-1`.
- If **at least one** hider is seen: each hider gets `-1`, each seeker gets `+1`.
During the preparation phase: reward is `0` for everyone.

Do NOT add shaping rewards for boxes, doors, or distance. If training is unstable, first try curriculum and hyperparameters (Section 8.3) before touching rewards, and document any change.

### 3.7 Observations
Each agent receives:
- **Local egocentric grid** (default `11 x 11` window centered on the agent), channels masked by line of sight:
  1. walls
  2. unlocked boxes
  3. locked boxes
  4. teammates
  5. opponents
- **Self vector**: normalized (x, y), `in_prep_phase` flag, normalized time remaining, team id.
- Output as a dict: `{"grid": float32 [C, H, W], "vec": float32 [D]}`.

**Global state** (for the centralized critic only): full map channels + all positions + lock flags + time features.

### 3.8 API
Implement as a **PettingZoo Parallel API** environment (`HideSeekEnv`) with `reset(seed=...)`, `step(actions)`, `agents`, `observation_space(agent)`, `action_space(agent)`, plus `state()` returning the global state. Rendering with **Pygame** (`render_mode="rgb_array"` for GIFs/videos, `"human"` for live view).

## 4. Algorithm: MAPPO

- **Two teams**, each with its own policy network, parameter-shared across the agents inside the team: `pi_hider`, `pi_seeker`.
- **Centralized critic** per team using the global state (centralized training, decentralized execution).
- PPO with clipping, GAE (`gamma=0.99`, `lambda=0.95`), entropy bonus, value loss clipping, gradient clipping, advantage normalization.
- Network: small CNN (or flatten + MLP at first) for `grid` + MLP for `vec`, merged, then (optional) GRU for partial observability. Start without GRU, add as an ablation.
- Both teams are trained **simultaneously** (self-play across teams).
- Vectorized environments (default 16 parallel envs) for throughput.
- Use PyTorch. Keep the implementation small and readable (single `mappo/` package). Do not hide the algorithm behind a large framework; clarity matters for the report.

## 5. Metrics (logged every N updates)

Core:
- mean episode return per team, hider "seen" fraction (after prep phase)
- win rate per team (fraction of post-prep steps in the team's favor)

Strategy detectors (computed automatically from the env state, no human labeling):
- `door_blocked_rate`: fraction of episodes where all door cells are blocked by a locked box at the end of prep phase
- `boxes_locked_mean`: mean number of locked boxes at end of prep
- `box_moves_mean`: mean number of box pushes per episode
- `hider_wall_hugging`: fraction of steps a hider is adjacent to a wall while unseen
- `seeker_unlock_attempts` / `seeker_box_pushes`: seeker interaction with boxes
- `mean_hider_seeker_distance`

Emergence timeline: for each detector, record the training step when it first crosses a threshold (e.g. `door_blocked_rate > 0.5` over a rolling window).

## 6. Experiments

All experiments use **>= 3 seeds** (5 if compute allows). Report mean +/- std (or 95% CI) and plot learning curves.

| ID | Experiment | Variation |
|----|-----------|-----------|
| E0 | Baselines | random policy vs random; scripted heuristic seeker vs learned hider |
| E1 | Boxes | `n_boxes` in {0, 1, 2, 3} |
| E2 | Vision | `vision_radius` in {3, 5, 7} |
| E3 | Team size | 1v1, 2v2, 3v3 |
| E4 | Prep phase | `T_prep` in {0, 20, 40, 80} |
| E5 | Memory (optional) | MLP vs GRU policy |

Always keep other parameters at defaults when varying one. Save each run's config, seed, git commit hash, and metrics as JSON/CSV.

## 7. Repository Structure

```
hide-and-seek-rl/
├── README.md
├── PROJECT_SPEC.md
├── AGENTS.md                # rules for Cursor (see Section 13)
├── pyproject.toml           # or requirements.txt
├── .gitignore
├── configs/
│   ├── default.yaml
│   └── experiments/         # e1_boxes.yaml, e2_vision.yaml, ...
├── src/hideseek/
│   ├── __init__.py
│   ├── env/
│   │   ├── core.py          # game logic (no rendering, no gym deps)
│   │   ├── pz_env.py        # PettingZoo wrapper
│   │   ├── visibility.py    # Bresenham LOS
│   │   ├── maps.py          # layout presets
│   │   └── render.py        # Pygame rendering
│   ├── mappo/
│   │   ├── networks.py
│   │   ├── buffer.py        # rollout buffer + GAE
│   │   ├── trainer.py
│   │   └── utils.py
│   ├── metrics.py           # strategy detectors
│   ├── train.py             # CLI entry point
│   ├── evaluate.py          # run policy, save GIF/replay JSON
│   └── export.py            # replay JSON (and ONNX) export
├── scripts/
│   ├── run_experiments.sh
│   └── plot_results.py
├── tests/
│   ├── test_visibility.py
│   ├── test_env_rules.py
│   └── test_reward.py
├── demo/                    # web viewer (Section 11)
├── results/                 # small CSV/JSON only (no big files)
└── assets/                  # GIFs, figures used in README
```

Rules for files in git:
- **Do not commit** model checkpoints, `wandb/`, large videos, or `__pycache__`. Put checkpoints in GitHub Releases or Hugging Face.
- Commit small GIFs (< 5 MB each) and result CSVs.

## 8. Implementation Notes

### 8.1 Determinism and reproducibility
- Every random source takes a `seed`. Provide `set_seed(seed)` for numpy/torch/random.
- Log the config + seed + git commit hash at the start of each run.
- Pin dependency versions.

### 8.2 Performance
- Keep `core.py` pure NumPy with minimal Python loops. Cache LOS computations where possible.
- Profile steps/sec early. Target >= 5k env steps/sec total on CPU with vectorization. If lower, optimize before training.

### 8.3 If training does not show emergent behavior
Try in this order, and record each attempt in the README "lessons" section:
1. Check reward/logic bugs with scripted agents and unit tests.
2. Increase total steps; check entropy does not collapse early.
3. Curriculum: start with a longer prep phase and fewer opponents, then move to defaults.
4. Tune: learning rate, number of envs, batch size, `clip`, entropy coefficient.
5. Only then consider environment changes (smaller map, larger vision radius), documenting why.

An honest negative result with solid diagnostics is still valuable.

### 8.4 Stack
Python 3.10+, NumPy, PyTorch, PettingZoo (parallel API), Gymnasium, Pygame, PyYAML, matplotlib/pandas, Weights & Biases or TensorBoard, pytest, imageio (GIFs).

## 9. Testing Requirements

- `test_visibility.py`: LOS blocked by walls and boxes, symmetric cases, radius cutoff.
- `test_env_rules.py`: push mechanics, locked boxes, collisions, prep-phase freezing, determinism with the same seed.
- `test_reward.py`: reward values in prep phase and after, zero-sum property.
- A smoke test that runs a short MAPPO training (a few updates) without crashing.

## 10. Deliverables (Definition of Done)

1. Clean repo with README (problem, env description, how to install/train/evaluate, results, GIFs).
2. Reproducible configs for E0-E4 and result tables/plots with seed statistics.
3. 1-2 minute video or GIF set showing the emergent behaviors with timeline.
4. Interactive web demo (Section 11).
5. A 4-8 page report in paper style: introduction, environment, method, experiments, results, limitations, future work. Optionally post to arXiv or submit to a workshop.
6. Pretrained checkpoints published outside git (Releases or Hugging Face).

## 11. Web Demo (graphical presentation)

Do it in two stages so there is always something showable:

**Stage A (recommended first): replay viewer.**
- `export.py` saves episodes as JSON (map, positions per step, box/lock states, visibility per step).
- `demo/` contains a static page (HTML + JavaScript + canvas) that loads those JSON files, plays them back with play/pause/speed controls, shows the vision of each agent, and lets the user choose between checkpoints (e.g. "step 1M", "step 5M", "final") to see strategies emerging over training.
- Host it with GitHub Pages and link it from the personal website.

**Stage B (optional, harder): live interactive.**
- Port the env to JavaScript, export policies to ONNX, run with `onnxruntime-web`, and let the user edit map parameters (boxes, vision radius). Only do this if Stage A is done and polished, since it requires keeping two env implementations consistent (add a cross-check test).

## 12. Step-by-Step Commit Plan

Follow Conventional Commits. Each step must leave the repo in a working state and pass tests. After each step: run tests, run the quick sanity script, then commit and push.

| # | Commit message | What to do | Acceptance criteria |
|---|----------------|-----------|---------------------|
| 1 | `chore: initialize repo structure` | Folders, `pyproject.toml`/requirements, `.gitignore`, empty package, README stub, this spec | `pip install -e .` works |
| 2 | `feat(env): add grid core with walls, agents, boxes` | `core.py` state, map presets, movement and push logic | Unit tests for movement/push pass |
| 3 | `feat(env): add line-of-sight visibility` | `visibility.py` (Bresenham, radius), tests | LOS tests pass |
| 4 | `feat(env): add reward and prep phase logic` | Episode phases, team zero-sum reward | Reward tests pass |
| 5 | `feat(env): add PettingZoo parallel wrapper and observations` | `pz_env.py`, observation dict, `state()` | Passes PettingZoo parallel API test |
| 6 | `feat(render): add Pygame renderer and GIF export` | `render.py`, `evaluate.py` with random policy | A random-agent GIF is generated in `assets/` |
| 7 | `feat(mappo): add networks and rollout buffer with GAE` | `networks.py`, `buffer.py` | Shape tests pass |
| 8 | `feat(mappo): add trainer with two-team self-play` | PPO update, vectorized envs, logging | Smoke training runs, loss finite |
| 9 | `feat(metrics): add strategy detectors and logging` | `metrics.py`, wandb/TensorBoard | Metrics appear in logs |
| 10 | `exp: baseline runs and first learning curves` | E0 + first default training | Plot in `results/` + README section |
| 11 | `exp: E1 boxes ablation` | Multi-seed runs | Table + plot in README |
| 12 | `exp: E2 vision and E3 team size` | Multi-seed runs | Tables + plots |
| 13 | `exp: E4 prep phase (and optional GRU)` | Runs | Tables + plots |
| 14 | `feat(export): replay JSON export` | Save episodes per checkpoint | JSON files load correctly |
| 15 | `feat(demo): static replay viewer` | Canvas viewer, checkpoint selector | Opens locally and on GitHub Pages |
| 16 | `docs: write README results and report draft` | GIFs, tables, lessons learned | README complete |
| 17 | `release: publish checkpoints and v1.0` | Tag, GitHub Release / Hugging Face upload | Links in README |

Commit often within each step (small commits are fine). Never commit a broken state to `main`: use a branch per step if you like and merge when tests pass.

## 13. Rules for Cursor (`AGENTS.md` / `.cursorrules`)

Put the following in `AGENTS.md` at the repo root:

```
- This repo implements the project described in PROJECT_SPEC.md. Follow it strictly.
- Work on ONE step of the commit plan at a time. Do not implement future steps.
- Write small, readable, typed Python (3.10+), with docstrings on public functions.
- Keep game logic in src/hideseek/env/core.py free of rendering and RL dependencies.
- Every new function with logic needs a pytest test. Run `pytest -q` before saying a step is done.
- Everything random must take a seed. Never use global unseeded randomness.
- Do not add shaping rewards or change the reward without asking.
- Do not commit checkpoints, wandb folders, or large files.
- After each step, summarize what changed and propose the commit message.
```

### Suggested Cursor prompt template (one per step)

```
Read PROJECT_SPEC.md and AGENTS.md. Implement ONLY step <N> of Section 12
("<commit message>"). Follow the spec for details. Add the tests listed in
Section 9 that apply to this step. Run pytest and fix failures. Then give me
a short summary and the exact git commands to commit this step.
```

## 14. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| No emergent behavior after reasonable training | Follow Section 8.3; report honestly with diagnostics; reduce map or simplify |
| Environment bugs masquerading as "strategies" | Unit tests, scripted-agent sanity checks, watch GIFs frequently |
| Training instability | Multiple seeds, smaller LR, entropy schedule, curriculum |
| Compute limits | Small grid, CPU-friendly networks, prioritize E1 and E2, treat others as optional |
| Scope creep | Stage A demo first; GRU, procedural maps, JS port are optional |

## 15. README Outline (to fill at the end)

1. Title + one-paragraph summary + hero GIF
2. Motivation and research question
3. Environment (rules, observations, reward) with a figure
4. Method (MAPPO details)
5. Results: emergence timeline, ablation tables/plots (mean +/- std)
6. Demo link
7. How to install, train, evaluate, reproduce
8. Limitations and lessons learned
9. References (OpenAI hide-and-seek 2019; MAPPO paper "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"; PPO; GAE; PettingZoo)
10. Citation / license
