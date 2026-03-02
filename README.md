# MSOR-KLU2026 - 6x6 Checkers (MDP, Heuristic, RL)

This project models a 6x6 checkers environment as an episodic MDP and compares a rule-based heuristic policy against tabular Q-learning. It includes reproducible training/evaluation pipelines, statistical analysis outputs, and play modes for human-vs-human and human-vs-computer.

## Instructor-Focused Summary

- Formal MDP formulation (`S, A, T, R, gamma`) with deterministic transitions.
- Dynamic legal-action handling with forced-capture constraints.
- Two required solution approaches implemented:
  - Heuristic rule-based strategy.
  - Tabular Q-learning with epsilon-greedy exploration.
- Reproducible evaluation with:
  - alternating starting side,
  - multi-seed aggregation,
  - draw handling,
  - confidence intervals for win rate,
  - additional game-level metrics (captures, promotions, terminal material difference).

## Installation

```bash
git clone <repo-url>
cd MSOR-KLU2026
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Quick check:

```bash
python3 run.py test
```

## One-Command Reproducible Pipeline (Recommended)

Run full training + evaluation and write final artifacts to `experiments/results_final`:

```bash
python3 experiments/reproduce_pipeline.py \
  --episodes 8000 \
  --seed 42 \
  --opponent heuristic \
  --games 200 \
  --num-seeds 5 \
  --alternate-start \
  --out experiments/results_final
```

## Expected Final Artifacts (`experiments/results_final/`)

- `q_table.npy`
- `training_metrics.npz`
- `reward_curve.png`
- `episode_length_curve.png`
- `winrate_over_training.png`
- `head_to_head_winrates.png`
- `evaluation_summary.json`

`evaluation_summary.json` includes per-seed and aggregate metrics, win-rate confidence intervals, draw rates, captures, promotions, and terminal material difference.

## Manual Commands (If Needed)

Training:

```bash
python3 run.py train --episodes 8000 --seed 42 --opponent heuristic --out experiments/results
```

Evaluation:

```bash
python3 experiments/evaluate_agents.py \
  --q-table experiments/results/q_table.npy \
  --games 200 \
  --seed 42 \
  --num-seeds 5 \
  --alternate-start \
  --out experiments/results
```

## Play Modes

Interactive menu:

```bash
python3 run.py
```

Direct commands:

```bash
python3 run.py human-cli
python3 run.py human-gui
python3 run.py ai-cli --opponent heuristic --human-color b --seed 42
python3 run.py ai-gui --opponent random --human-color r --seed 42
python3 run.py ai-cli --opponent rl --human-color b --q-table experiments/results/q_table.npy --seed 42
```

## Project Structure

```text
src/checkers/      core rules + Gymnasium environment
agents/            random, heuristic, and Q-table agents
experiments/       train/evaluate/reproduce scripts
play/              human-vs-AI gameplay (CLI/GUI)
tests/             rule-focused unit tests
run.py             unified launcher
```

## Main Files

- `src/checkers/core.py` - legal moves, captures, promotion, transitions
- `src/checkers/env.py` - Gymnasium wrapper + action mask
- `agents/heuristic_agent.py` - interpretable baseline policy
- `experiments/train_q_learning.py` - tabular RL training
- `experiments/evaluate_agents.py` - robust multi-seed evaluation
- `experiments/reproduce_pipeline.py` - one-command reproducibility pipeline
- `docs/MSOR_thesis_report.md` - thesis-style report draft

## Notes and Limits

- Tabular Q-learning is intentionally simple and may underperform against strong heuristics under short training horizons.
- For stronger RL results, increase episodes and compare multiple seeds.
- A concise usage guide is available in [USAGE.md](USAGE.md).
