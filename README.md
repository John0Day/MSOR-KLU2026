# MSOR-KLU2026
Let’s build a Python Game 
# Checkers Game Project — MDP Modeling & Implementation

## Project Overview

This project implements a **simplified 6×6 Checkers game** and models it as a **Markov Decision Process (MDP)** for analysis in Management Science / Operations Research.
The objective is to study sequential decision-making, rule-based heuristics, and reinforcement learning approaches using a manageable but strategically meaningful board game.

The game implementation focuses on clarity of state representation, deterministic transitions, and well-defined rewards to support future reinforcement learning experiments.

---

## Game Description

Checkers is a two-player, turn-based strategy board game where players move pieces diagonally and capture opponent pieces by jumping over them.
Our project uses a **6×6 board** rather than the traditional 8×8 board to reduce computational complexity while preserving strategic structure.

### Basic Rules

* Two players: **Black (b)** and **Red (r)**
* Pieces move diagonally on dark squares only
* Regular pieces move forward diagonally:

  * Black moves downward
  * Red moves upward
* Captures occur by jumping over an opponent piece
* Multiple captures in one turn are allowed if available
* Promotion to **king** occurs when a piece reaches the opposite edge:

  * Kings move diagonally both forward and backward
* If a capture is available, it must be taken

### Winning Conditions

A player wins if:

* The opponent has no legal moves, or
* The opponent has no remaining pieces.

---

## Markov Decision Process (MDP) Formulation

### State Space (S)

Each state contains all information necessary to determine future actions:

* Board configuration (piece positions and types)
* Player whose turn it is
* Forced capture continuation context (if applicable)

Terminal states occur when:

* A player has no legal actions, or
* All pieces of one player are captured.

---

### Action Space (A)

Actions correspond to legal moves available in a state:

* Diagonal movement to an empty square
* Capture sequences (possibly multiple jumps)
* Promotion actions if a piece reaches the last row

If any capture is possible, only capture actions are allowed.

---

### Transition Model (P)

Transitions are **deterministic**:

Applying an action results in:

* Piece movement
* Removal of captured pieces
* Possible promotion to king
* Turn switching between players

This deterministic transition supports classical reinforcement learning modeling.

---

### Reward Function (R)

Standard competitive reward structure:

* **+1** for winning
* **−1** for losing
* **0** otherwise

Optional shaping (future work):

* Small positive reward for captures
* Slight penalty per move to encourage faster wins.

---

### Discount Factor (γ)

Since the game is episodic:

* γ ≈ 1.0 preserves long-term win/loss objectives
* Slightly smaller values (e.g., 0.99) can encourage faster victories.

---

## Implementation Details

### Key Features

* Fully playable console-based 6×6 Checkers game
* Deterministic game engine
* Forced capture rule enforcement
* Multi-jump capture handling
* Automatic king promotion
* Board visualization in terminal

### Technical Stack

* Python 3
* Standard library only (no external dependencies)

---

## Running the Game

1. Ensure Python 3 is installed.
2. Run:

```bash
python checkers.py
```

3. Enter moves in coordinate format:

```
b6 a5
```

(Type `q` to quit.)

---

## Educational Objectives

This project supports:

* Sequential decision modeling
* Reinforcement learning experimentation
* Heuristic strategy comparison
* Operations Research formulation skills
* Game environment simulation design

---

## Future Extensions

Potential improvements include:

* Reinforcement learning agent implementation
* Policy optimization experiments
* Graphical interface
* Larger board variants
* Statistical performance evaluation

---

## Contributors

Isaac Ebu-Danso
John
Alhajie
Chandar
Satya


---

## Course Context

This project was developed as part of coursework in **Management Science / Operations Research**, focusing on optimization, decision modeling, and computational solution methods.
