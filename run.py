from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANSI_OPTION = "\033[96m"
ANSI_RESET = "\033[0m"


def _run(cmd: list[str], headless: bool = False) -> int:
    env = os.environ.copy()
    if headless:
        env.setdefault("MPLBACKEND", "Agg")
        env.setdefault("MPLCONFIGDIR", "/tmp")
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


def run_human_cli() -> int:
    return _run([sys.executable, str(ROOT / "checkers6x6.py")])


def run_human_gui() -> int:
    return _run([sys.executable, str(ROOT / "Checkers" / "gui_checkers.py")])


def run_ai_cli(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "play" / "human_vs_ai_cli.py"),
        "--opponent",
        args.opponent,
        "--human-color",
        args.human_color,
        "--seed",
        str(args.seed),
        "--q-table",
        args.q_table,
    ]
    return _run(cmd)


def run_ai_gui(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "play" / "human_vs_ai_gui.py"),
        "--opponent",
        args.opponent,
        "--human-color",
        args.human_color,
        "--seed",
        str(args.seed),
        "--q-table",
        args.q_table,
    ]
    return _run(cmd)


def run_train(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "train_q_learning.py"),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--opponent",
        args.opponent,
        "--out",
        args.out,
    ]
    return _run(cmd, headless=True)


def run_eval(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "evaluate_agents.py"),
        "--q-table",
        args.q_table,
        "--games",
        str(args.games),
        "--seed",
        str(args.seed),
        "--out",
        args.out,
    ]
    return _run(cmd, headless=True)


def run_tests() -> int:
    return _run([sys.executable, "-m", "pytest", "-q"], headless=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified launcher for MSOR-KLU2026 checkers")
    sub = parser.add_subparsers(dest="mode", required=True)

    sub.add_parser("human-cli", help="Play in terminal (manual moves)")
    sub.add_parser("human-gui", help="Play in Tkinter GUI")
    p_ai_cli = sub.add_parser("ai-cli", help="Play in terminal against computer")
    p_ai_cli.add_argument("--opponent", choices=["random", "heuristic", "rl"], default="heuristic")
    p_ai_cli.add_argument("--human-color", choices=["b", "r"], default="b")
    p_ai_cli.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    p_ai_cli.add_argument("--seed", type=int, default=42)

    p_ai_gui = sub.add_parser("ai-gui", help="Play in GUI against computer")
    p_ai_gui.add_argument("--opponent", choices=["random", "heuristic", "rl"], default="heuristic")
    p_ai_gui.add_argument("--human-color", choices=["b", "r"], default="b")
    p_ai_gui.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    p_ai_gui.add_argument("--seed", type=int, default=42)

    p_train = sub.add_parser("train", help="Train tabular Q-learning")
    p_train.add_argument("--episodes", type=int, default=8000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--opponent", choices=["random", "heuristic"], default="heuristic")
    p_train.add_argument("--out", type=str, default="experiments/results")

    p_eval = sub.add_parser("eval", help="Evaluate agents using saved Q-table")
    p_eval.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    p_eval.add_argument("--games", type=int, default=300)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--out", type=str, default="experiments/results")

    sub.add_parser("test", help="Run unit tests")
    return parser


def _print_line(num: str, title: str, desc: str) -> None:
    print(f"  {num} {ANSI_OPTION}{title}{ANSI_RESET} - {desc}")


def _prompt_choice(prompt: str, options: dict[str, str], invalid_hint: str) -> str:
    while True:
        choice = input(prompt).strip()
        if choice in options:
            return options[choice]
        print(invalid_hint)


def _interactive_mode_selection() -> list[str]:
    options = {
        "1": "play-human",
        "2": "play-ai",
        "3": "train",
        "4": "eval",
        "5": "test",
        "0": "exit",
    }
    print("Choose mode:")
    _print_line("0)", "Exit", "Quit the launcher")
    _print_line("1)", "Play locally", "Human vs human")
    _print_line("2)", "Play against computer", "Choose UI and AI type")
    _print_line("3)", "Train Q-learning", "Run RL training and save outputs")
    _print_line("4)", "Evaluate agents", "Compare RL, heuristic, and random agents")
    _print_line("5)", "Run tests", "Execute unit tests")

    mode = _prompt_choice(
        "Enter number (0-5): ",
        options,
        "Invalid choice. Please enter 0, 1, 2, 3, 4, or 5.",
    )
    if mode in {"exit", "train", "eval", "test"}:
        return [mode]

    if mode == "play-human":
        ui_map = {"1": "human-cli", "2": "human-gui", "0": "exit"}
        print("\nPlay locally:")
        _print_line("0)", "Back/Exit", "Return without starting a game")
        _print_line("1)", "CLI", "Play in terminal")
        _print_line("2)", "GUI", "Play in Tkinter window")
        ui_mode = _prompt_choice(
            "Enter number (0-2): ",
            ui_map,
            "Invalid choice. Please enter 0, 1, or 2.",
        )
        return [ui_mode]

    # mode == "play-ai"
    ui_map = {"1": "ai-cli", "2": "ai-gui", "0": "exit"}
    print("\nPlay against computer - choose interface:")
    _print_line("0)", "Back/Exit", "Return without starting a game")
    _print_line("1)", "CLI", "Play in terminal")
    _print_line("2)", "GUI", "Play in Tkinter window")
    ui_mode = _prompt_choice(
        "Enter number (0-2): ",
        ui_map,
        "Invalid choice. Please enter 0, 1, or 2.",
    )
    if ui_mode == "exit":
        return ["exit"]

    opp_map = {"1": "random", "2": "heuristic", "3": "rl", "0": "exit"}
    print("\nPlay against computer - choose opponent:")
    _print_line("0)", "Back/Exit", "Return without starting a game")
    _print_line("1)", "Random AI", "Uninformed random moves")
    _print_line("2)", "Heuristic AI", "Rule-based baseline")
    _print_line("3)", "RL AI", "Uses trained Q-table")
    opponent = _prompt_choice(
        "Enter number (0-3): ",
        opp_map,
        "Invalid choice. Please enter 0, 1, 2, or 3.",
    )
    if opponent == "exit":
        return ["exit"]

    color_map = {"1": "b", "2": "r", "0": "exit"}
    print("\nChoose your color:")
    _print_line("0)", "Back/Exit", "Return without starting a game")
    _print_line("1)", "Black", "You move first")
    _print_line("2)", "Red", "Computer moves first")
    human_color = _prompt_choice(
        "Enter number (0-2): ",
        color_map,
        "Invalid choice. Please enter 0, 1, or 2.",
    )
    if human_color == "exit":
        return ["exit"]

    args = [ui_mode, "--opponent", opponent, "--human-color", human_color]
    if opponent == "rl":
        q_default = "experiments/results/q_table.npy"
        q_path = input(f"Q-table path [{q_default}]: ").strip()
        args.extend(["--q-table", q_path or q_default])
    return args


def main() -> int:
    parser = build_parser()
    if len(sys.argv) == 1:
        selected_args = _interactive_mode_selection()
        if selected_args[0] == "exit":
            print("Exiting.")
            return 0
        args = parser.parse_args(selected_args)
    else:
        args = parser.parse_args()

    if args.mode == "human-cli":
        return run_human_cli()
    if args.mode == "human-gui":
        return run_human_gui()
    if args.mode == "ai-cli":
        return run_ai_cli(args)
    if args.mode == "ai-gui":
        return run_ai_gui(args)
    if args.mode == "train":
        return run_train(args)
    if args.mode == "eval":
        return run_eval(args)
    if args.mode == "test":
        return run_tests()

    parser.error(f"Unknown mode: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
