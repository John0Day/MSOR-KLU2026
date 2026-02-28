# MSOR-KLU2026 - 6x6 Checkers (MDP, Heuristic, Reinforcement Learning)

Dieses Repository implementiert ein 6x6-Checkers-Projekt für Management Science / Operations Research mit drei Policy-Typen (Random, Heuristik, RL), reproduzierbaren Experimenten und Spielmodi für Mensch-vs-Mensch sowie Mensch-vs-Computer.

## 1. Für die Lehrperson: Theoretische Erkenntnisse

### 1.1 Problem als MDP
Wir modellieren Checkers als episodisches Markov Decision Process `M = (S, A, T, R, gamma)`:

- `S` (State): komplette Brettbelegung + Spieler am Zug + Multi-Jump-Zwangskontext.
- `A(s)` (Action): nur legale Züge, inklusive Forced-Capture-Regel.
- `T(s,a)` (Transition): deterministisch (Zug anwenden, evtl. Schlag entfernen, Promotion, Zugrecht-Update).
- `R(s,a,s')`: `+1` (Sieg), `-1` (Niederlage), `0` (sonst).
- Episode-Ende: keine legalen Züge (`terminated=True`) oder Turn-Cap (`truncated=True`) für stabile Evaluation.

### 1.2 Zentrale theoretische Einsichten
- **Zustandsraum vs. Lösbarkeit:** 6x6 reduziert die Komplexität gegenüber 8x8, erhält aber wesentliche strategische Struktur (Forced Captures, Kings, Multi-Jumps).
- **Deterministische Dynamik:** Da `T` deterministisch ist, hängen Lernschwierigkeiten primär an Zustandsgröße, Exploration und Reward-Sparseness, nicht an stochastischer Transition.
- **Dynamischer Aktionsraum:** Durch Forced Captures ist `A(s)` stark zustandsabhängig. Action-Masking ist deshalb konzeptionell und praktisch wichtig.
- **Heuristik als erklärbare Baseline:** Material + Mobility + Advancement liefern eine nachvollziehbare, interpretierbare Referenz gegen Random und RL.
- **RL-Lernsignal:** Sparse Terminal-Reward macht das Problem schwerer; Leistungsgewinn benötigt ausreichend Episoden und sinnvolle Gegnerwahl (Random vs. Heuristik).

### 1.3 Methodische Aussagekraft
Das Projekt erlaubt eine saubere Gegenüberstellung von:

- regelbasierter Entscheidungslogik (Heuristik),
- zufälliger Policy (Random) als Untergrenze,
- datengetriebenem Lernen (tabular Q-Learning).

Dadurch können typische OR/RL-Fragen transparent diskutiert werden:

- Wann schlägt eine handgebaute Policy eine gelernte Policy?
- Wie wirkt sich die Trainingsverteilung (Gegnerwahl) auf Generalisierung aus?
- Welche Limitationen hat tabulares Lernen bei wachsendem Zustandsraum?

### 1.4 Verifizierbarkeit der Ergebnisse
Die Nachvollziehbarkeit ist absichtlich technisch abgesichert:

- fixe Seeds in Training und Evaluation,
- explizite Output-Artefakte (Metriken, Plots, Q-Table),
- Unit-Tests für kritische Regeln (Forced Capture, Multi-Jump, Promotion, Terminalzustand).

## 2. GitHub-Style Installation

### 2.1 Voraussetzungen
- Python `3.10+` (im Projekt wird durchgehend `python3` verwendet)
- `pip`

### 2.2 Setup
```bash
git clone <repo-url>
cd MSOR-KLU2026
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.3 Schnelltest
```bash
python3 run.py test
```

## 3. Reproduzierbare Experimente

### 3.1 Training
```bash
python3 run.py train --episodes 8000 --seed 42 --opponent heuristic --out experiments/results
```

### 3.2 Evaluation
```bash
python3 run.py eval --q-table experiments/results/q_table.npy --games 300 --seed 42 --out experiments/results
```

### 3.3 Erwartete Artefakte
Nach Training/Evaluation in `experiments/results/`:

- `q_table.npy`
- `training_metrics.npz`
- `reward_curve.png`
- `episode_length_curve.png`
- `winrate_over_training.png`
- `head_to_head_winrates.png`

## 4. Spielmodi (Runner)

### 4.1 Interaktives Menü
```bash
python3 run.py
```

Mehrstufige Auswahl möglich:
- `Play locally` (Human vs Human)
- `Play against computer` -> `CLI/GUI` -> Gegnerwahl (`Random`, `Heuristic`, `RL`) -> Farbwahl

### 4.2 Direkte Commands
```bash
python3 run.py human-cli
python3 run.py human-gui
python3 run.py ai-cli --opponent heuristic --human-color b --seed 42
python3 run.py ai-gui --opponent random --human-color r --seed 42
python3 run.py ai-cli --opponent rl --human-color b --q-table experiments/results/q_table.npy --seed 42
```

## 5. Projektbeschreibung und Struktur

```text
src/checkers/      Kernlogik + Gymnasium-Environment
agents/            Random-, Heuristic- und Q-Table-Agent
experiments/       Training, Evaluation, Plot-Erzeugung
play/              Human-vs-AI in CLI und GUI
tests/             Regeltests
run.py             Zentraler Launcher
```

Wichtige Dateien:
- `src/checkers/core.py`: Regelwerk (legal moves, captures, promotion, transitions)
- `src/checkers/env.py`: Gymnasium-Wrapper mit Action-Mask
- `agents/heuristic_agent.py`: erklärbare Baseline-Policy
- `experiments/train_q_learning.py`: tabulares RL-Training
- `experiments/evaluate_agents.py`: Head-to-Head-Vergleich
- `play/human_vs_ai_cli.py`, `play/human_vs_ai_gui.py`: Spielen gegen Computer

## 6. Grenzen und sinnvolle Erweiterungen

- Tabular Q-Learning skaliert nur begrenzt bei größerem Zustandsraum.
- Gegen starke Heuristiken kann mehr Training oder alternative Approximation (z. B. DQN) nötig sein.
- Potenzielle Erweiterungen:
  - Self-Play-Curriculum,
  - Feature-Engineering/State-Aggregation,
  - systematische Hyperparameter-Sweeps,
  - getrennte Evaluation nach Startfarbe.

## 7. Zusatzdokument

Für eine kompakte Schritt-für-Schritt-Bedienung:
- [USAGE.md](USAGE.md)
