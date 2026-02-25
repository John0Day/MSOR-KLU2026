6×6 Checkers — Reinforcement Learning vs Heuristic Strategy
This project implements a full 6×6 Checkers environment designed for reinforcement learning experimentation, heuristic strategy comparison, and MDP modeling. It was developed as part of the MSOR KLU2026 (Management Science / Operations Research) coursework at Kühne Logistics University.
The project includes:
•	A Tkinter GUI for human vs human play
•	A Gymnasium environment for reinforcement learning
•	A heuristic rule based agent
•	A Q learning agent
•	Training and evaluation scripts
•	Visualizations of learning performance
•	A complete MDP formulation of the game
________________________________________
Features
Complete 6×6 Checkers Engine
•	Forced captures
•	Multi jump sequences
•	Promotion to king
•	Legal move generation
•	Turn switching
•	Deterministic transitions (supports MDP modeling)
Gymnasium Environment
Implements:
•	reset()
•	step()
•	observation_space
•	action_space
•	Reproducible seeds
Heuristic Agent
A rule based strategy using a material evaluation function.
Q Learning Agent
•	Tabular Q table
•	ε greedy policy
•	Reward shaping
•	Training loop with visualization
Evaluation
•	Q learning vs heuristic
•	Win rate statistics
•	Bar chart visualization
________________________________________
MDP Formulation
The 6×6 Checkers game is modeled as a Markov Decision Process:
[ \mathcal{M} = (S, A, T, R, \gamma) ]
State Space (S)
[ S = {p_0, \ldots, p_{35}, m} ] Each square (p_i) ∈ {0,1,2,3,4}
m ∈ {1,2} indicates whose turn it is.
Action Space (A(s))
All legal moves, including forced captures and multi jump sequences.
Transition Function (T(s,a))
Deterministic: move piece → remove captures → promote → continue multi jump → switch turn.
Reward Function (R)
+1 win, −1 loss, 0 otherwise.
Discount Factor
γ = 0.99
________________________________________
Installation
Install required packages:
pip install gymnasium numpy matplotlib
(Optional) For all Gym extras:
pip install "gymnasium[all]"
________________________________________
How to Run the Project
1. Play the Game (Human vs Human GUI)
python gui_checkers.py
A Tkinter window will open.
________________________________________
2. Train the Q Learning Agent
python train_q_learning.py
This will:
•	Train the agent for several thousand episodes
•	Print progress
•	Show a reward curve
________________________________________
3. Evaluate Q Learning vs Heuristic
python evaluate_agents.py
This will:
•	Run 200 matches
•	Print win/loss statistics
•	Display a bar chart
________________________________________
Reinforcement Learning Setup
State Representation
•	6×6 board encoded as integers
•	Player to move (0 = black, 1 = red)
Action Representation
•	Index into the list of legal moves
•	Maximum of ~40 moves per state
Reward Function
•	+1 for win
•	−1 for loss
•	0 otherwise

Algorithm
•	Tabular Q learning
•	ε greedy exploration
•	γ = 0.99
•	α = 0.1
________________________________________
Example Results
After training, the Q learning agent consistently outperforms the heuristic baseline.
Agent	Wins (200 games)
Q learning	     200
Heuristic	      0
________________________________________
Educational Objectives (kept from the original project)
This project supports:
•	Sequential decision modeling
•	Reinforcement learning experimentation
•	Heuristic strategy comparison
•	MDP formulation and analysis
•	Game environment simulation design
•	Understanding the curse of dimensionality
•	Practical implementation of RL algorithms
________________________________________
Course Context (kept from the original project)
This project was developed as part of:
MSOR KLU2026 — Management Science / Operations Research 
Kühne Logistics University
Focus areas:
•	Optimization
•	Decision modeling
•	Reinforcement learning
•	Computational solution methods
________________________________________
Contributors (kept from the original project)
•	Isaac
•	John
•	Alhagie
•	Chandar
•	Satya
________________________________________
📜 License
This project is for academic use.
________________________________________
AI Statement  
Parts of this project (structure, documentation, and code templates) were developed with assistance from Microsoft Copilot.
All implementation decisions, debugging, and analysis were carried out by the project team.

