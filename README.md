### Tabular Q-Learning for 6x6 Checkers

A custom 6x6 Checkers reinforcement learning project built on top of a Gymnasium-compatible environment and a tabular Q-learning agent. The environment models full Checkers dynamics (including forced captures and multi-jumps), while the agent uses a **canonical state representation**, **backward-pass Q-learning updates**, and an **auto-curriculum** of opponents (Self-play, Heuristic, and Random). Baseline agents (Random and a rule-based Priority Heuristic) are included for evaluation and comparison, along with plotting utilities to visualize training dynamics and final performance.

---

### Features

- **Custom 6x6 Gymnasium Environment**  
  `Checkers6x6Env` implements a compact 6x6 Checkers variant with a `MultiDiscrete([6, 6, 6, 6])` action space and a structured observation (`board`, `current_player`).

- **Canonical State Representation**  
  `observation_to_state` converts raw observations into a **player-0-centric** canonical tuple by flipping the board and remapping piece IDs when it is player 1’s turn, and by keeping only the 18 dark squares. This allows a single tabular Q-function to be shared across both roles.

- **Tabular Q-Learning with Backward-Pass Updates**  
  `QLearningAgent` maintains a dictionary-based Q-table over `(state, action)` pairs, and performs **episode-level backward-pass updates** with dynamic, visit-count-based learning rates.

- **Dynamic Role Assignment (P1/P2)**  
  During training, the agent is dynamically assigned as Player 0 or Player 1 based on recent evaluation performance, biasing training toward whichever side is currently weaker.

- **Auto-Curriculum of Opponents**  
  Training progresses through phases that mix **Random**, **Heuristic**, and **Self-Play** opponents. A pool of **recent and historical Q-table snapshots** is used as self-play opponents to reduce forgetting.

- **Multi-Jump Handling**  
  The environment and training loop correctly handle **multi-jump capture sequences**, enforcing continued moves from the same piece when more captures are available while keeping rewards and transitions aligned with the agent’s perspective.

- **Decoupled Evaluation**  
  Clean evaluation runs (`evaluate_agent`, `plots.py`) are **fully separated from training**, using fixed opponents, no Q-table updates, and pure greedy policies to produce unbiased win-rate measurements from both Player 1 and Player 2 perspectives.

- **Priority Heuristic Baseline Agent**  
  `PriorityHeuristicAgent` implements a strong hand-crafted baseline that prioritizes captures, promotions, edge safety, and center control, enabling meaningful comparisons to the learned policy.

- **Comprehensive Training Statistics & Plots**  
  Training logs (`training_stats.npz`) and plotting utilities (`plots.py`) generate learning curves, Q-table growth, episode length trends, role-specific win rates, and performance distributions across agents.

- **Terminal-Based Gameplay**  
  `play.py` loads the trained Q-table and lets you watch the learned agent play against Random or Heuristic opponents in the terminal using a text-based board renderer.

---

### Prerequisites

- **Python**: **Python 3.8+** (3.8, 3.9, 3.10 or newer should work)
- **Core Libraries** (exactly as listed in `requirements.txt`):
  - **NumPy** (`numpy`)
  - **Gymnasium** (`gymnasium`)
  - **Matplotlib** (`matplotlib`)

---

### Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/John0Day/MSOR-KLU2026.git
```

2. **(Optional but recommended) Create a virtual environment**

On **Windows (PowerShell)**:

```bash
python -m venv .venv
.\.venv\Scripts\Activate
```

On **macOS / Linux**:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

All required runtime dependencies are captured in `requirements.txt` and match the list above:

```bash
pip install -r requirements.txt
```

---

### How to Run (Replicating the Project)

The main workflow is:

1. **Train** the Q-learning agent.  
2. **Evaluate / Play** games against baseline agents in the terminal.  
3. **Visualize** learning curves and performance plots.

To fully replicate the project, run all the Python files (`train.py`, `play.py`, `plots.py`, etc.) from inside the `MSOR_Checkers_6x6` folder where they are located.


#### 1. Training

Run the training script to learn a tabular Q-policy and record statistics:

```bash
python train.py
```

This will:

- Train the `QLearningAgent` in the custom `Checkers6x6Env` using:
  - Canonical state representation (shared across roles).
  - Backward-pass Q-learning updates.
  - Auto-curriculum opponents: Random, Heuristic, and Self-Play (with historical snapshots).
  - Dynamic role balancing (P1 vs P2) based on evaluation win rates.
- Saves the **learned Q-table** as:

  - `q_table.pkl`

- Saves **training statistics** as a compressed NumPy archive:

  - `training_stats.npz`

Both files are written to the same directory as `train.py`.

#### 2. Evaluation & Gameplay

Once training has finished and `q_table.pkl` exists, you can watch the trained agent play against baseline opponents.

Run:

```bash
python play.py
```

This will:

- Load `q_table.pkl` into a `QLearningAgent`.
- Create a fresh `Checkers6x6Env` instance.
- Run multiple episodes where:
  - The Q-learning agent plays as Player 0 by default (configurable).
  - The opponent is either:
    - **Random** (uniform random legal moves), or
    - **Heuristic** (`PriorityHeuristicAgent`, rule-based).
- Prints each board state and the current player to the terminal after every move using `env.render()`.
- Prints per-episode results and an aggregate summary:

  - Total wins, losses, and draws across episodes.

You can adjust parameters like number of episodes, opponent type, and rendering in `play.py`’s `evaluate` function (e.g., `opponent_type="heuristic"`, `render=False` for faster evaluation).

#### 3. Visualizing Results

After training has produced `training_stats.npz` and `q_table.pkl`, you can generate all plots used for analysis:

```bash
python plots.py
```

This script will:

- Load `training_stats.npz` and `q_table.pkl`.
- Generates the following PNG files in the same directory:

  - **`learning_curve_win_rate.png`**  
    Moving-average training win rate (from agent’s perspective) plus **decoupled** evaluation win rates vs Random and Heuristic opponents at 1000-episode checkpoints.

  - **`state_space_growth.png`**  
    Growth of the Q-table (number of `(state, action)` entries) over training, as a proxy for explored state space.

  - **`game_length.png`**  
    Moving-average episode lengths (number of environment steps) vs episode index.

  - **`eval_p1_vs_p2_win_rates.png`**  
    Separate evaluation win rates when the agent plays as **Player 1** vs **Player 2** against the fixed heuristic opponent, showing role asymmetry.

  - **`performance_distribution.png`**  
    Stacked bar chart comparing **Random**, **Heuristic**, and **Q-Learning** agents’ win/loss/draw proportions vs a common opponent, using many evaluation games.

These plots give a comprehensive picture of learning progress, exploration, role balance, and final performance relative to baselines.

---

### Project Structure

(This is focused on the core RL and environment components.)

- **`checkers_env.py`**  
  - Defines the **`Checkers6x6Env`** Gymnasium environment.  
  - Implements:
    - 6x6 board encoding, piece types, and turn tracking.
    - Legal move generation with forced captures.
    - Single and multi-jump capture sequences via `active_piece`.
    - Promotion to kings, draw rules (max steps and no-progress), and terminal rewards.
    - Text-based rendering (`render`) and a convenience factory `make_env()`.

- **`q_agent.py`**  
  - Defines:
    - **`observation_to_state`**: canonical state mapping to a player-0 perspective tuple over the 18 dark squares.
    - **`QLearningAgent`**: tabular Q-learning agent with:
      - A dictionary Q-table over `(state, action)` pairs.
      - State-dependent epsilon-greedy exploration (`epsilon_greedy_policy`).
      - Backward-pass episode updates (`backward_pass_update`) with visit-based learning rates.
      - A purely greedy action selector (`greedy_action`) for evaluation.

- **`heuristic_agent.py`**  
  - Defines **`PriorityHeuristicAgent`**, a rule-based agent that:
    - Prioritizes captures, promotions, edge safety, and centralization.
    - Serves as a strong baseline opponent and evaluation benchmark.

- **`train.py`**  
  - Orchestrates training:
    - Creates the environment and the `QLearningAgent`.
    - Runs episodes via `run_episode`, which:
      - Correctly handles dynamic turns, multi-jumps, and reward aggregation from the agent’s perspective.
      - Supports Random, Heuristic, and Self-Play opponents (with optional opponent Q-tables).
    - Implements:
      - **Auto-curriculum** logic over training phases (Random → Heuristic → Self-Play mix).
      - **Dynamic role assignment** based on decoupled evaluation metrics.
      - **Opponent pool** of recent and historical Q-table snapshots for self-play.
      - **Decoupled evaluation** via `evaluate_agent` every 1000 episodes vs fixed Random and Heuristic opponents.
    - Saves:
      - `q_table.pkl` (learned Q-table).
      - `training_stats.npz` (rewards, winners, episode lengths, evaluation win rates, and Q-table sizes).

- **`play.py`**  
  - Provides interactive evaluation and terminal gameplay:
    - Loads `q_table.pkl` into a `QLearningAgent`.
    - Runs episodes where the agent plays vs Random or Heuristic opponents.
    - Renders the board and logs winners and aggregate win/loss/draw statistics.

- **`plots.py`**  
  - Loads `training_stats.npz` (and `q_table.pkl` where needed).
  - Generates all analysis plots:
    - Learning curve and evaluation win rates.
    - Q-table size growth.
    - Episode length trends.
    - Role-specific win rates (P1 vs P2) vs Heuristic.
    - Performance distribution across Random, Heuristic, and Q-learning agents.

---

### License

This project is released under a permissive open-source license.

 
> This project is licensed under the **MIT License**.  
