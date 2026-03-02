# 6x6 Checkers as a Markov Decision Process
## Comparison of a Rule-Based Heuristic and Tabular Q-Learning

**Course:** MSOR-KLU2026  
**Team:** _(add names)_  
**Date:** _(add date)_

---

## Abstract
This report models a simplified 6x6 checkers game as an episodic Markov Decision Process (MDP) and compares two solution approaches: a rule-based heuristic policy and tabular Q-learning. The environment enforces forced captures, multi-jump continuation, and king promotion, which creates a dynamic legal action set. We evaluate policies in controlled head-to-head matchups using fixed seeds, alternating starting side, and multi-seed aggregation. Current results show that the heuristic policy strongly outperforms random play, while the current Q-learning policy remains weak against the heuristic baseline. Learning curves indicate that the RL pipeline functions correctly but remains undertrained under sparse terminal rewards and limited training horizon. The report provides a reproducible methodological baseline and identifies concrete next steps for improving RL performance.

---

## AI Statement
Generative AI tools were used for drafting support, report structuring, and language refinement. All implementation decisions, integration, debugging, experiment execution, metric extraction, and final interpretation were performed and verified by the project team. All reported values and figures were generated from repository code and stored experiment artifacts; no results were fabricated.

---

## 1. Problem Description (Game Rules)

## 1.1 Game Overview
We study a deterministic two-player zero-sum game on a 6x6 board. Players are Black and Red. Black moves first. Only dark squares are playable.

## 1.2 Initial Setup
Black occupies playable squares in the top two rows; Red occupies playable squares in the bottom two rows. All pieces start as men.

## 1.3 Movement Rules
- Men move diagonally forward by one square.
- Kings move diagonally by one square in both forward and backward directions.

## 1.4 Capture Rules
A capture is performed by jumping diagonally over an adjacent opponent piece onto an empty landing square directly behind it. The captured piece is removed.

## 1.5 Forced Capture and Multi-Jump
- If at least one capture is available, non-capturing moves are illegal.
- After a capture, if the same piece can capture again, it must continue capturing within the same turn.

## 1.6 Promotion
A man reaching the opposite edge row is promoted to a king immediately after the move.

## 1.7 Terminal Condition
An episode terminates when the current player has no legal actions. The opponent wins.

---

## 2. Mathematical Formulation
We model the environment as an episodic MDP
`M = (S, A, T, R, gamma)`.

## 2.1 State Space `S`
A state `s` contains:
1. Board configuration,
2. Player to move,
3. Multi-jump continuation context (forced piece, if any).

Implementation observation encoding:
- board matrix `6 x 6` with values in `{0,1,2,3,4}` mapped to `{empty, b, B, r, R}`,
- player bit in `{0,1}` for `{black, red}`.

## 2.2 Action Space `A(s)`
Actions are state-dependent legal moves.
- If captures exist, only capture actions are legal.
- During multi-jump continuation, only actions of the forced piece are legal.

For agent compatibility, actions are represented as indices into the legal move list with an action mask over a fixed discrete space.

## 2.3 Transition Function `T`
Transitions are deterministic:
1. apply selected move,
2. remove captured piece (if any),
3. promote on edge row,
4. continue same-turn capture if required,
5. otherwise switch player.

## 2.4 Reward Function `R`
Sparse terminal rewards:
- `+1` if the acting side wins,
- `-1` if the acting side loses,
- `0` otherwise.

## 2.5 Discount Factor `gamma`
We use `gamma = 0.99`.

---

## 3. Solution Approaches

## 3.1 Heuristic Method (Rule-Based)
The heuristic is explicit and interpretable.

### 3.1.1 Evaluation Components
For candidate successor states:
- material: men = 1, kings = 2,
- mobility: own legal moves minus opponent legal moves,
- advancement: forward progress of men,
- risk: immediate opponent capture opportunities after the move.

### 3.1.2 Decision Rule
The policy applies a lexicographic key:
1. minimize immediate counter-capture risk,
2. maximize weighted static evaluation.

If captures are available, candidate set is restricted to captures.

### 3.1.3 Pseudocode
```text
function HEURISTIC_POLICY(board, player, legal_moves):
    candidates <- capture_moves(legal_moves) if any else legal_moves
    best_move <- None
    best_key <- (-infinity, -infinity)

    for m in candidates:
        board_next <- simulate(board, m)
        risk <- - immediate_opponent_captures(board_next, player)
        score <- w_material * material(board_next, player)
               + w_mobility * mobility(board_next, player)
               + w_advancement * advancement(board_next, player)
        key <- (risk, score)  # lexicographic
        if key > best_key:
            best_key <- key
            best_move <- m

    return best_move
```

## 3.2 Reinforcement Learning Method (Tabular Q-Learning)
The RL setup learns Q-values for the controlled side (Black) against a fixed opponent policy (Random or Heuristic during training).

### 3.2.1 State Representation for Q-Table
State key is serialized as:
`(tuple(board.flatten()), player_to_move)`.

### 3.2.2 Update Rule
For legal actions only:
`Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_{a' in A(s')} Q(s',a') - Q(s,a)]`.

### 3.2.3 Exploration
Epsilon-greedy policy with exponential decay:
`epsilon <- max(epsilon_min, epsilon * epsilon_decay)`.

### 3.2.4 Pseudocode
```text
Initialize Q-table (default 0)
Set alpha, gamma, epsilon_start, epsilon_end, epsilon_decay

for each episode:
    s <- env.reset(seed)
    done <- false

    while not done:
        if current player is opponent side:
            a_opp <- opponent_policy(s)
            s, done <- env.step(a_opp)
            continue

        choose a with epsilon-greedy over legal actions
        s_next, r, done <- env.step(a)

        if done:
            target <- r
        else:
            target <- r + gamma * max_legal_action_value(s_next)

        Q[s,a] <- Q[s,a] + alpha * (target - Q[s,a])
        s <- s_next

    decay epsilon
```

---

## 4. Experimental Setup

## 4.1 Configuration from Repository Scripts
From `experiments/train_q_learning.py`:
- episodes: `300` (for current stored test run in `results_testlauf`)
- alpha: `0.15`
- gamma: `0.99`
- epsilon: start `1.0`, end `0.05`, decay `0.9993`
- periodic train-time eval: every `250` episodes over `80` games
- seed base: `42`

From `src/checkers/env.py`:
- max turn cap per episode: `200` (otherwise game truncated as draw)

From `experiments/evaluate_agents.py`:
- default evaluation games: `300` (we also ran `200` for analysis section)
- fixed seed offsets by matchup

## 4.2 Matchups
- RL vs Random
- RL vs Heuristic
- Heuristic vs Random

## 4.3 Protocol Notes
- Starting side is alternated in the current evaluation protocol (`--alternate-start`).
- Draws are possible via turn cap (`truncated=True`) and are reported explicitly.
- Multi-seed evaluation is enabled via `--num-seeds`; per-seed and aggregate statistics are exported.

## 4.4 Metrics
- win/loss/draw counts and rates,
- mean episode length,
- mean return,
- mean captures per game,
- mean promotions per game,
- terminal material difference,
- reward trend,
- win rate over training checkpoints.

---

## 5. Results and Analysis

All results in this section come from reruns using `experiments/results_testlauf/q_table.npy` with:
- `games = 200` per seed and matchup,
- `num_seeds = 3`,
- `alternate_start = True`.

## 5.1 Aggregated Head-to-Head Results (3 Seeds x 200 Games)

### 5.1.1 RL vs Random
- mean win/loss/draw rates: `0.478 / 0.453 / 0.068`
- win-rate std across seeds: `0.006`
- mean return: `0.025` (std `0.036`)
- mean episode length: `46.25` (std `5.01`)
- pooled wins/losses/draws over 600 games: `287 / 272 / 41`
- pooled 95% Wilson CI for win rate: `[0.439, 0.518]`

### 5.1.2 RL vs Heuristic
- mean win/loss/draw rates: `0.000 / 0.500 / 0.500`
- win-rate std across seeds: `0.000`
- mean return: `-0.500` (std `0.000`)
- mean episode length: `116.00` (std `0.00`)
- pooled wins/losses/draws over 600 games: `0 / 300 / 300`
- pooled 95% Wilson CI for win rate: `[0.000, 0.006]`

### 5.1.3 Heuristic vs Random
- mean win/loss/draw rates: `0.973 / 0.023 / 0.003`
- win-rate std across seeds: `0.005`
- mean return: `0.950` (std `0.014`)
- mean episode length: `31.32` (std `1.30`)
- pooled wins/losses/draws over 600 games: `584 / 14 / 2`
- pooled 95% Wilson CI for win rate: `[0.957, 0.984]`

## 5.2 Learning Dynamics (Current Stored Run)
From `training_metrics.npz` in `results_testlauf`:
- episodes trained: `300`
- mean reward: `-0.98`
- mean reward (last 20 episodes): `-1.00`
- mean episode length: `28.86`
- mean episode length (last 20): `28.50`
- checkpoint at episode 250: win rate vs Random `0.25`, vs Heuristic `0.00`

## 5.3 Interpretation
- The heuristic baseline is currently the strongest policy.
- RL under current setup is functional but substantially underperforming.
- Likely causes:
  1. sparse terminal reward signal,
  2. short training horizon in current result set,
  3. large effective state-action sparsity due to dynamic legal action sets,
  4. fixed-opponent training dynamics.

## 5.4 Validity and Caveats
- Training remains asymmetric (controlled-side RL against a fixed opponent policy).
- Draws exist and should be analyzed separately from wins/losses.
- Current training horizon (300 episodes in stored run) is likely below convergence for tabular learning.
- Additional game-level indicators (captures/promotions/material) improve interpretability but do not replace policy-level robustness checks across larger seed sets.

---

## 6. Visualization

## 6.1 Reward Curve (300 episodes, moving average window = 100)
![Reward curve](../experiments/results_testlauf/reward_curve.png)

## 6.2 Episode Length Curve (300 episodes, moving average window = 100)
![Episode length](../experiments/results_testlauf/episode_length_curve.png)

## 6.3 Win Rate over Training (evaluated every 250 episodes, 80 games per checkpoint)
![Winrate over training](../experiments/results_testlauf/winrate_over_training.png)

## 6.4 Final Head-to-Head Bar Chart (evaluation set)
![Head-to-head comparison](../experiments/results_testlauf/head_to_head_winrates.png)

Full evaluation metadata and per-seed metrics:
- `../experiments/results_testlauf/evaluation_summary.json`

---

## 7. Conclusion
This project satisfies the assignment requirements: formal MDP definition, heuristic and RL solution methods with pseudocode, reproducible experiments, comparative analysis, and graphical result presentation. In the current state, the heuristic policy dominates RL. The RL pipeline is valid but not yet optimized for competitive performance.

Recommended next steps:
1. increase training horizon,
2. tune exploration and learning-rate schedule,
3. evaluate on more seeds and larger match counts,
4. consider richer value approximation beyond tabular Q-learning.

---

## Appendix A: Repository Components Used
- `src/checkers/core.py`
- `src/checkers/env.py`
- `agents/heuristic_agent.py`
- `agents/q_agent.py`
- `experiments/train_q_learning.py`
- `experiments/evaluate_agents.py`
- `tests/test_rules.py`
- `experiments/results_testlauf/*`
