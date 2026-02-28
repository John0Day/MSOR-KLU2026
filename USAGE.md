# Usage Guide

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start the Unified Runner

Interactive menu:

```bash
python3 run.py
```

Menu flow for computer games:
1. `Play against computer`
2. choose `CLI` or `GUI`
3. choose opponent: `Random`, `Heuristic`, or `RL`
4. choose your color (`Black` or `Red`)

Direct commands:

```bash
python3 run.py human-cli
python3 run.py human-gui
python3 run.py ai-cli --opponent heuristic --human-color b --seed 42
python3 run.py ai-gui --opponent random --human-color b --seed 42
python3 run.py ai-cli --opponent rl --human-color b --q-table experiments/results/q_table.npy --seed 42
python3 run.py train --episodes 8000 --seed 42 --opponent heuristic --out experiments/results
python3 run.py eval --q-table experiments/results/q_table.npy --games 300 --seed 42 --out experiments/results
python3 run.py test
```

## Recommended Reproducible Workflow

1. Run tests
```bash
python3 run.py test
```

2. Train RL model
```bash
python3 run.py train --episodes 8000 --seed 42 --opponent heuristic --out experiments/results
```

3. Evaluate trained model
```bash
python3 run.py eval --q-table experiments/results/q_table.npy --games 300 --seed 42 --out experiments/results
```

## Output Files

After training/evaluation, check:

- `experiments/results/q_table.npy`
- `experiments/results/training_metrics.npz`
- `experiments/results/reward_curve.png`
- `experiments/results/episode_length_curve.png`
- `experiments/results/winrate_over_training.png`
- `experiments/results/head_to_head_winrates.png`

## Troubleshooting

- If you see macOS Python GUI crash popups, keep using `python3 run.py ...` commands (scripts are configured for headless plotting in experiment mode).
- If a long run is interrupted, you can restart training with the same seed for consistent comparison behavior.
