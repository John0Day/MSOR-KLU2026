# 6x6 Checkers as a Markov Decision Process
## Baseline and Extended Comparison of Rule-Based Heuristics and Tabular Q-Learning

**Course:** MSOR-KLU2026  
**Team:** _Satya, Isaac, John, Chandan, Bun Kijera_  
**Date:** _09.03.2026_

---

## Abstract
This project models 6x6 checkers as an episodic Markov Decision Process (MDP) and compares rule-based and reinforcement-learning solutions. We implement a baseline tabular Q-learning workflow and an extended workflow with canonical state encoding, adaptive exploration/learning rates, and curriculum/self-play training. All components are implemented in Python using Gymnasium and exposed through a unified runner (`run.py`). Reproducible experiments were executed with fixed seeds. Baseline multi-seed head-to-head results show a strong heuristic baseline and mixed RL behavior. Extended training improves evaluation performance against both random and heuristic opponents under its own benchmark protocol.

---

## AI Statement
Generative AI tools were used for drafting support, structure suggestions, and language refinement. All implementation choices, integration decisions, debugging, experiment execution, and metric verification were performed by the project team. All reported results below were generated from repository runs on 09 March 2026.

---

## 1. Problem Description

### 1.1 Game Setup
We model a deterministic two-player zero-sum checkers game on a 6x6 board.

- Players: Black (`b`) and Red (`r`)
- Black moves first
- Only dark squares are playable
- Initial setup: two rows of men per side

### 1.2 Rules

- Men move diagonally forward by one square
- Kings move diagonally in both directions
- Captures jump over an adjacent opponent piece
- Forced capture is enforced
- Multi-jump continuation is enforced for the same piece
- Promotion to king occurs at the opposite edge row
- The game ends when the side to move has no legal moves

### 1.3 Interface Requirements
The project includes:

- Console play (human vs human)
- Console/GUI play against AI
- Programmatic training and evaluation scripts
- Unified command interface through `run.py`

---

## 2. MDP Formulation
We define the MDP as:

`M = (S, A, T, R, gamma)`

### 2.1 States `S`
A state includes:

- board configuration,
- player to move,
- forced-piece context during capture chains.

Observation encoding uses integer piece IDs (`0..4`) and a player indicator.

### 2.2 Actions `A(s)`
Actions are legal moves in the current state.

- If captures exist, non-captures are illegal
- In multi-jump, only moves of the active piece are legal

Baseline environment actions are represented as indices in the legal move list. Extended modules additionally use explicit move tuples.

### 2.3 Transition Function `T`
Transitions are deterministic:

1. apply move,
2. remove captured piece,
3. promote if needed,
4. continue capture chain if required,
5. otherwise switch player.

### 2.4 Reward Function `R`
Environment reward design (current default):

- `-0.005` step penalty for each valid move,
- `+0.1` capture bonus,
- `+0.15` promotion bonus,
- `+1.0` terminal win bonus,
- `-1.0` terminal loss / invalid terminal action.

This adds dense intermediate feedback while preserving terminal outcome incentives.

### 2.5 Discount `gamma`
`gamma = 0.99`.

---

## 3. Solution Approaches

## 3.1 Heuristic Policies

### 3.1.1 Baseline Heuristic (`HeuristicAgent`)
Scores candidate moves by material, mobility, advancement, and immediate counter-capture risk.

### 3.1.2 Priority Heuristic (`PriorityHeuristicAgent`)
Decision priority:

1. capture,
2. promotion,
3. edge safety,
4. center advance,
5. random fallback.

## 3.2 Reinforcement Learning Policies

### 3.2.1 Baseline Q-Table (`QTableAgent`)
Tabular Q-learning with legal-action masking and epsilon-greedy exploration.

### 3.2.2 Extended Adaptive Q-Table (`AdaptiveQTableAgent`)
Adds:

- canonical state normalization,
- state-dependent epsilon,
- dynamic alpha(s,a) using visit counts,
- curriculum opponent scheduling,
- self-play with snapshot pools.

### 3.2.3 Update Equation
`Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`

---

## 4. Pseudocode

### 4.1 Baseline Heuristic
```text
function HEURISTIC_POLICY(board, player, legal_moves):
    candidates <- captures if captures exist else legal_moves
    return argmax(candidates) by ( -immediate_counter_capture_risk,
                                   weighted_board_score )
```

### 4.2 Baseline Q-Learning
```text
initialize Q-table
for each episode:
    reset env
    while not terminal:
        choose legal action by epsilon-greedy
        step env
        update Q on legal next actions
    decay epsilon
```

### 4.3 Extended Curriculum Q-Learning
```text
initialize adaptive Q-agent
initialize snapshot pools (recent, historical)
for each episode:
    sample opponent type by curriculum phase
    sample self-play snapshot if required
    choose training side (balance or weaker-side bias)
    run episode with legal-action masking
    update Q with adaptive alpha and state-dependent epsilon

    every eval_interval:
        evaluate vs random and heuristic from both colors
        advance curriculum if thresholds are reached

    every snapshot_interval:
        store Q-table snapshots for self-play
```

---

## 5. Implementation

- `src/checkers/core.py`: core game rules
- `src/checkers/env.py`: Gymnasium environment
- `agents/heuristic_agent.py`: baseline and priority heuristics
- `agents/q_agent.py`: baseline and adaptive Q-table agents
- `experiments/train_q_learning.py`: baseline training
- `experiments/evaluate_agents.py`: baseline evaluation
- `experiments/train_extended.py`: extended curriculum/self-play training
- `play/evaluate_extended.py`: extended play/evaluation script
- `experiments/plots_extended.py`: extended plotting pipeline
- `run.py`: unified launcher

All experiments below were run via `run.py` on 09 March 2026.

---

## 6. Experimental Setup

## 6.1 Primary Runs (Current Default)

```bash
python3 run.py train --episodes 20000 --seed 42 --out experiments/results
python3 run.py eval --q-table experiments/results/q_table.npy --games 300 --seed 42 --num-seeds 5 --alternate-start --out experiments/results
python3 run.py plots-extended --metrics experiments/results/training_metrics.npz --out experiments/results --window 500
```

## 6.2 Legacy Baseline Runs (Reference Only)

```bash
python3 run.py train-legacy --episodes 8000 --seed 42 --opponent heuristic --out experiments/results
python3 run.py eval-legacy --q-table experiments/results/q_table.npy --games 200 --seed 42 --num-seeds 5 --alternate-start --out experiments/results
```

---

## 7. Results

## 7.1 Baseline Multi-Seed Head-to-Head (5 seeds x 200 games)
Source: `experiments/results/evaluation_summary.json`.

### 7.1.1 RL vs Random
- win/loss/draw: `0.488 / 0.454 / 0.058`
- avg game length: `44.85`
- pooled 95% win-rate CI: `[0.457, 0.519]`

### 7.1.2 RL vs Heuristic
- win/loss/draw: `0.500 / 0.000 / 0.500`
- avg game length: `115.00`
- pooled 95% win-rate CI: `[0.469, 0.531]`

### 7.1.3 Heuristic vs Random
- win/loss/draw: `0.966 / 0.030 / 0.004`
- avg game length: `31.56`
- pooled 95% win-rate CI: `[0.953, 0.976]`

## 7.2 Baseline Training Summary (8000 episodes)
Source: `experiments/train_q_learning.py` log + `training_metrics.npz`.

- final checkpoint win rate vs random: `0.412`
- final checkpoint win rate vs heuristic: `1.000`

(Checkpoint values are computed on internal periodic evaluation windows.)

## 7.3 Extended Training Summary (20000 episodes)
Source: `experiments/results/training_metrics.npz`.

- episodes: `20000`
- mean reward: `0.137`
- mean reward (last 500): `0.350`
- mean episode length: `28.33`
- mean episode length (last 500): `24.72`

Evaluation series (20 checkpoints):

- eval win vs random: first `0.950`, last `0.975`
- eval win vs heuristic (overall): first `0.675`, last `0.750`
- eval win vs heuristic as black: first `0.900`, last `0.500`
- eval win vs heuristic as red: first `0.450`, last `1.000`
- Q-table size: first `12938`, last `89726`

## 7.4 Interpretation

- Baseline evaluation confirms a very strong heuristic baseline against random play.
- Baseline RL remains behaviorally asymmetric against the heuristic benchmark.
- Extended training improves aggregate evaluation performance in its own protocol and grows a substantially larger Q-table.
- Color asymmetry remains important and should always be reported explicitly.

## 7.5 Episode Budget Study (Why 20000 Episodes)
To justify the default training budget, we ran a controlled comparison:

- Config A: `20000` episodes
- Config B: `40000` episodes
- For each config: `3` training seeds (`42, 142, 242`)
- Evaluation protocol per run: `games=300`, `num-seeds=10`, `alternate-start=True`

Aggregated over the three runs per config:

- **Config A (20000 episodes)**
  - RL vs Random: `win=0.523 +- 0.005`
  - RL vs Heuristic: `win=0.790 +- 0.027`
  - Heuristic vs Random: `win=0.675 +- 0.004`

- **Config B (40000 episodes)**
  - RL vs Random: `win=0.537 +- 0.009`
  - RL vs Heuristic: `win=0.678 +- 0.155`
  - Heuristic vs Random: `win=0.671 +- 0.007`

Conclusion from this ablation:

- Increasing to `40000` episodes marginally improved RL vs Random.
- Against the main benchmark (RL vs Heuristic), `40000` episodes were less stable and weaker on average due to high variance across seeds.
- Therefore, `20000` episodes is selected as the default because it provides the best trade-off between performance, stability, and compute time in our current setup.

---

## 8. Visualization

Generated figures in `experiments/results/`:

- `reward_curve.png`
- `episode_length_curve.png`
- `winrate_over_training.png`
- `head_to_head_winrates.png`
- `learning_curve_win_rate.png`
- `reward_curve_extended.png`
- `state_space_growth.png`
- `game_length.png`
- `eval_black_vs_red_win_rates.png`

Example inclusions:

![Baseline winrate over training](../experiments/results/winrate_over_training.png)
![Baseline head-to-head](../experiments/results/head_to_head_winrates.png)
![Extended learning curve](../experiments/results/learning_curve_win_rate.png)
![Extended color split](../experiments/results/eval_black_vs_red_win_rates.png)

---

## 9. Assignment Coverage

### 9.1 Written Requirements
- Problem description: fulfilled
- MDP formulation (`S, A, T, R, gamma`): fulfilled
- Heuristic + RL approach with pseudocode: fulfilled
- Comparative analysis: fulfilled
- Visual result presentation: fulfilled
- AI statement: fulfilled

### 9.2 Implementation Requirements
- Python implementation: fulfilled
- Gymnasium environment: fulfilled
- GitHub-sharable codebase: fulfilled
- Reproducibility with fixed seeds: fulfilled

### 9.3 Remaining Formal Delivery Checks
Before final submission PDF:

1. ensure final exported document is within max page limit (15 pages all-inclusive + AI statement),
2. keep citations/references consistent,
3. include final selected plots/tables only (for concise defense slides and report).

---

## 10. Conclusion
The project fulfills the technical assignment requirements and now includes both a baseline and an extended RL workflow under a unified runner. Baseline evaluation highlights a strong heuristic policy and nontrivial RL asymmetry. Extended training provides improved benchmark performance and richer diagnostics (state-space growth and color-split evaluation), giving a stronger basis for oral defense and critical discussion.

---

## Appendix A: Runner Commands

```bash
python3 run.py
python3 run.py human-cli
python3 run.py human-gui
python3 run.py ai-cli --opponent heuristic --human-color b --seed 42
python3 run.py ai-gui --opponent random --human-color r --seed 42
python3 run.py train --episodes 8000 --seed 42 --opponent heuristic --out experiments/results
python3 run.py eval --q-table experiments/results/q_table.npy --games 300 --seed 42 --out experiments/results
python3 run.py train-extended --episodes 20000 --seed 42 --out experiments/results
python3 run.py play-extended --q-table experiments/results/q_table.npy --episodes 200 --opponent heuristic --agent-color b
python3 run.py plots-extended --metrics experiments/results/training_metrics.npz --out experiments/results --window 500
python3 run.py test
```
