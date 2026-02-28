# MSOR-KLU2026 - 6x6 Checkers (MDP, Heuristic, Reinforcement Learning)

This repository implements a 6x6 checkers project for Management Science / Operations Research, including three policy types (Random, Heuristic, RL), reproducible experiments, and play modes for Human-vs-Human and Human-vs-Computer.

## 1. For the Instructor: Theoretical Insights

### 1.1 Problem Formulation as an MDP
We model checkers as an episodic Markov Decision Process `M = (S, A, T, R, gamma)`:

- `S` (State): full board configuration + player to move + forced multi-jump context.
- `A(s)` (Action): legal moves only, including forced-capture constraints.
- `T(s,a)` (Transition): deterministic (apply move, remove captured piece if any, promote if needed, update turn).
- `R(s,a,s')`: `+1` (win), `-1` (loss), `0` (otherwise).
- Episode end: no legal moves (`terminated=True`) or turn cap (`truncated=True`) for stable evaluation.

### 1.2 Key Theoretical Findings
- **State-space size vs. tractability:** 6x6 reduces complexity compared to 8x8 while preserving core strategic structure (forced captures, kings, multi-jumps).
- **Deterministic dynamics:** because `T` is deterministic, learning difficulty is driven mainly by state-space size, exploration, and sparse rewards rather than transition stochasticity.
- **Dynamic action space:** with forced captures, `A(s)` is strongly state-dependent. Action masking is therefore both conceptually and practically important.
- **Heuristic as interpretable baseline:** material + mobility + advancement provide a transparent benchmark against Random and RL.
- **RL learning signal:** sparse terminal rewards make the task harder; performance gains require enough episodes and appropriate training opponents (Random vs. Heuristic).

### 1.3 Methodological Value
The project supports a clean comparison of:

- rule-based decision logic (Heuristic),
- random policy (Random) as a lower baseline,
- data-driven learning (tabular Q-learning).

This enables transparent discussion of typical OR/RL questions:

- When does a hand-crafted policy outperform a learned policy?
- How does training distribution (opponent choice) affect generalization?
- What are the practical limits of tabular learning as state-space grows?

### 1.4 Reproducibility and Auditability
Reproducibility is intentionally enforced via:

- fixed seeds in training and evaluation,
- explicit output artifacts (metrics, plots, Q-table),
- unit tests for rule-critical behavior (forced capture, multi-jump, promotion, terminal detection).

## 2. GitHub-Style Installation

### 2.1 Prerequisites
- Python `3.10+` (this project consistently uses `python3`)
- `pip`

### 2.2 Setup
```bash
git clone <repo-url>
cd MSOR-KLU2026
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.3 Quick Verification
```bash
python3 run.py test
```

## 3. Reproducible Experiments

### 3.1 Training
```bash
python3 run.py train --episodes 8000 --seed 42 --opponent heuristic --out experiments/results
```

### 3.2 Evaluation
```bash
python3 run.py eval --q-table experiments/results/q_table.npy --games 300 --seed 42 --out experiments/results
```

### 3.3 Expected Output Artifacts
After training/evaluation, `experiments/results/` should include:

- `q_table.npy`
- `training_metrics.npz`
- `reward_curve.png`
- `episode_length_curve.png`
- `winrate_over_training.png`
- `head_to_head_winrates.png`

## 4. Play Modes (Runner)

### 4.1 Interactive Menu
```bash
python3 run.py
```

Supported multi-step flow:
- `Play locally` (Human vs Human)
- `Play against computer` -> `CLI/GUI` -> opponent (`Random`, `Heuristic`, `RL`) -> color selection

### 4.2 Direct Commands
```bash
python3 run.py human-cli
python3 run.py human-gui
python3 run.py ai-cli --opponent heuristic --human-color b --seed 42
python3 run.py ai-gui --opponent random --human-color r --seed 42
python3 run.py ai-cli --opponent rl --human-color b --q-table experiments/results/q_table.npy --seed 42
```

## 5. Project Description and Structure

```text
src/checkers/      Core logic + Gymnasium environment
agents/            Random, Heuristic, and Q-table agents
experiments/       Training, evaluation, plot generation
play/              Human-vs-AI modes for CLI and GUI
tests/             Rule-focused tests
run.py             Unified launcher
```

Key files:
- `src/checkers/core.py`: rule engine (legal moves, captures, promotion, transitions)
- `src/checkers/env.py`: Gymnasium wrapper with action masking
- `agents/heuristic_agent.py`: interpretable baseline policy
- `experiments/train_q_learning.py`: tabular RL training
- `experiments/evaluate_agents.py`: head-to-head comparison
- `play/human_vs_ai_cli.py`, `play/human_vs_ai_gui.py`: human-vs-computer gameplay

## 6. Limitations and Extensions

- Tabular Q-learning scales poorly as state-space grows.
- Against stronger heuristics, additional training or function approximation (for example DQN) may be required.
- Useful extensions:
  - self-play curriculum,
  - feature engineering / state aggregation,
  - systematic hyperparameter sweeps,
  - separate evaluation by starting color.

## 7. Additional Documentation

For concise step-by-step usage instructions:
- [USAGE.md](USAGE.md)
