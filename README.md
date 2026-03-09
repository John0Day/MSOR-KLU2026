# MSOR-KLU2026 - 6x6 Checkers (MDP, Heuristic, RL)

This project models 6x6 checkers as an episodic Markov Decision Process (MDP) and compares a rule-based heuristic policy against tabular reinforcement learning (RL).

The repository provides:

- a Gymnasium environment,
- heuristic and RL agents,
- reproducible training and evaluation workflows,
- a unified runner (`run.py`),
- plots and report artifacts.

## Key Features

- Formal MDP framing (`S, A, T, R, gamma`)
- Forced-capture and multi-jump rule handling
- Two solution approaches:
  - Rule-based heuristic
  - Tabular Q-learning (adaptive extended workflow)
- Reproducible experiments with fixed seeds
- Multi-seed head-to-head evaluation with confidence intervals
- Reward shaping in the environment (capture/promotion/step feedback)

## Installation

```bash
git clone <repo-url>
cd MSOR-KLU2026
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Quick sanity check:

```bash
python3 run.py test
```

## Primary Workflow (Recommended)

Use the interactive menu:

```bash
python3 run.py
```

Or run directly:

```bash
python3 run.py train --episodes 20000 --seed 42 --out experiments/results
python3 run.py eval --q-table experiments/results/q_table.npy --games 300 --seed 42 --num-seeds 5 --alternate-start --out experiments/results
python3 run.py plots-extended --metrics experiments/results/training_metrics.npz --out experiments/results --window 500
```

## Additional Commands

Play modes:

```bash
python3 run.py human-cli
python3 run.py human-gui
python3 run.py ai-cli --opponent heuristic --human-color b --seed 42
python3 run.py ai-gui --opponent random --human-color r --seed 42
python3 run.py play-extended --q-table experiments/results/q_table.npy --episodes 50 --opponent heuristic --agent-color b
```

Legacy baseline pipeline (kept for comparison/backward compatibility):

```bash
python3 run.py train-legacy --episodes 8000 --seed 42 --opponent heuristic --out experiments/results
python3 run.py eval-legacy --q-table experiments/results/q_table.npy --games 300 --seed 42 --num-seeds 5 --alternate-start --out experiments/results
```

## Output Artifacts

Typical output directory: `experiments/results/`

- `q_table.npy`
- `training_metrics.npz`
- `evaluation_summary.json`
- `head_to_head_winrates.png`
- `winrate_over_training.png`
- `learning_curve_win_rate.png`
- `state_space_growth.png`
- `game_length.png`
- `eval_black_vs_red_win_rates.png`

## Project Structure

```text
src/checkers/      core rules + Gymnasium environment
agents/            heuristic, random, and Q-table agents
experiments/       training, evaluation, plots, and batch scripts
play/              gameplay and evaluation scripts
tests/             unit tests
run.py             unified launcher
```

## Main Files

- `src/checkers/core.py` - legal moves, capture logic, promotion, transitions
- `src/checkers/env.py` - Gymnasium environment and reward signal
- `agents/heuristic_agent.py` - heuristic agents
- `agents/q_agent.py` - baseline and adaptive Q-table agents
- `experiments/train_extended.py` - default RL training workflow
- `experiments/evaluate_extended_agents.py` - default RL evaluation workflow
- `experiments/plots_extended.py` - plotting pipeline
- `docs/MSOR_thesis_report.md` - thesis-style write-up

## Notes

- The default workflow is the extended RL setup (`train` / `eval`).
- The legacy workflow remains available for baseline comparison.
- A concise command reference is available in `USAGE.md`.
