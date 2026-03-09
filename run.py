from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANSI_OPTION = "\033[96m"
ANSI_RESET = "\033[0m"
ANSI_PRIMARY = "\033[38;5;208m"


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
        str(ROOT / "experiments" / "train_extended.py"),
        "--episodes",
        str(args.episodes),
        "--gamma",
        str(args.gamma),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-games",
        str(args.eval_games),
        "--seed",
        str(args.seed),
        "--out",
        args.out,
    ]
    return _run(cmd, headless=True)


def run_eval(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "evaluate_extended_agents.py"),
        "--q-table",
        args.q_table,
        "--games",
        str(args.games),
        "--seed",
        str(args.seed),
        "--num-seeds",
        str(args.num_seeds),
        "--alternate-start" if args.alternate_start else "--no-alternate-start",
        "--out",
        args.out,
    ]
    return _run(cmd, headless=True)


def run_tests() -> int:
    return _run([sys.executable, "-m", "pytest", "-q"], headless=True)


def run_train_extended(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "train_extended.py"),
        "--episodes",
        str(args.episodes),
        "--gamma",
        str(args.gamma),
        "--eval-interval",
        str(args.eval_interval),
        "--eval-games",
        str(args.eval_games),
        "--seed",
        str(args.seed),
        "--out",
        args.out,
    ]
    return _run(cmd, headless=True)


def run_train_legacy(args: argparse.Namespace) -> int:
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


def run_eval_legacy(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "evaluate_agents.py"),
        "--q-table",
        args.q_table,
        "--games",
        str(args.games),
        "--seed",
        str(args.seed),
        "--num-seeds",
        str(args.num_seeds),
        "--alternate-start" if args.alternate_start else "--no-alternate-start",
        "--out",
        args.out,
    ]
    return _run(cmd, headless=True)


def run_play_extended(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "play" / "evaluate_extended.py"),
        "--q-table",
        args.q_table,
        "--episodes",
        str(args.episodes),
        "--opponent",
        args.opponent,
        "--agent-color",
        args.agent_color,
        "--render" if args.render else "--no-render",
        "--sleep",
        str(args.sleep),
        "--seed",
        str(args.seed),
    ]
    return _run(cmd)


def run_plots_extended(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "plots_extended.py"),
        "--metrics",
        args.metrics,
        "--out",
        args.out,
        "--window",
        str(args.window),
    ]
    return _run(cmd, headless=True)


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

    p_train = sub.add_parser("train", help="Train RL (extended default)")
    p_train.add_argument("--episodes", type=int, default=20000)
    p_train.add_argument("--gamma", type=float, default=0.99)
    p_train.add_argument("--eval-interval", type=int, default=1000)
    p_train.add_argument("--eval-games", type=int, default=80)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--out", type=str, default="experiments/results")

    p_eval = sub.add_parser("eval", help="Evaluate RL vs heuristic/random (extended)")
    p_eval.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    p_eval.add_argument("--games", type=int, default=300)
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--num-seeds", type=int, default=5)
    p_eval.add_argument("--alternate-start", action=argparse.BooleanOptionalAction, default=True)
    p_eval.add_argument("--out", type=str, default="experiments/results")

    p_train_legacy = sub.add_parser("train-legacy", help="Legacy baseline training")
    p_train_legacy.add_argument("--episodes", type=int, default=8000)
    p_train_legacy.add_argument("--seed", type=int, default=42)
    p_train_legacy.add_argument("--opponent", choices=["random", "heuristic"], default="heuristic")
    p_train_legacy.add_argument("--out", type=str, default="experiments/results")

    p_eval_legacy = sub.add_parser("eval-legacy", help="Legacy baseline evaluation")
    p_eval_legacy.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    p_eval_legacy.add_argument("--games", type=int, default=300)
    p_eval_legacy.add_argument("--seed", type=int, default=42)
    p_eval_legacy.add_argument("--num-seeds", type=int, default=5)
    p_eval_legacy.add_argument("--alternate-start", action=argparse.BooleanOptionalAction, default=True)
    p_eval_legacy.add_argument("--out", type=str, default="experiments/results")

    p_train_ext = sub.add_parser("train-extended", help="Train with adaptive curriculum/self-play")
    p_train_ext.add_argument("--episodes", type=int, default=20000)
    p_train_ext.add_argument("--gamma", type=float, default=0.99)
    p_train_ext.add_argument("--eval-interval", type=int, default=1000)
    p_train_ext.add_argument("--eval-games", type=int, default=80)
    p_train_ext.add_argument("--seed", type=int, default=42)
    p_train_ext.add_argument("--out", type=str, default="experiments/results")

    p_play_ext = sub.add_parser("play-extended", help="Evaluate trained Q-table (extended mode)")
    p_play_ext.add_argument("--q-table", type=str, default="experiments/results/q_table.npy")
    p_play_ext.add_argument("--episodes", type=int, default=20)
    p_play_ext.add_argument("--opponent", choices=["random", "heuristic"], default="random")
    p_play_ext.add_argument("--agent-color", choices=["b", "r"], default="b")
    p_play_ext.add_argument("--render", action=argparse.BooleanOptionalAction, default=False)
    p_play_ext.add_argument("--sleep", type=float, default=0.15)
    p_play_ext.add_argument("--seed", type=int, default=42)

    p_plots_ext = sub.add_parser("plots-extended", help="Generate extended training plots")
    p_plots_ext.add_argument("--metrics", type=str, default="experiments/results/training_metrics.npz")
    p_plots_ext.add_argument("--out", type=str, default="experiments/results")
    p_plots_ext.add_argument("--window", type=int, default=500)

    sub.add_parser("test", help="Run unit tests")
    return parser


def _print_line(num: str, title: str, desc: str) -> None:
    print(f"  {num} {ANSI_OPTION}{title}{ANSI_RESET} - {desc}")


def _print_primary_line(num: str, title: str, desc: str) -> None:
    print(f"  {num} {ANSI_PRIMARY}{title}{ANSI_RESET} - {desc}")


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
        "5": "play-extended",
        "6": "plots-extended",
        "7": "train-legacy",
        "8": "eval-legacy",
        "9": "test",
        "0": "exit",
    }
    print("Choose mode:")
    print("  -------------------------------------------")
    print(f"  {ANSI_PRIMARY}Primary Workflow{ANSI_RESET}")
    _print_line("0)", "Exit", "Quit the launcher")
    _print_line("1)", "Play locally", "Human vs human")
    _print_line("2)", "Play against computer", "Choose UI and AI type")
    _print_primary_line("3)", "Train RL", "Extended RL training (default)")
    _print_primary_line("4)", "Evaluate RL", "RL vs Heuristic/Random (extended)")
    _print_primary_line("6)", "Plots extended", "Generate extended plot set")
    print("  -------------------------------------------")
    print("  Additional/Advanced Options")
    _print_line("5)", "Play extended", "Run trained model vs random/heuristic")
    _print_line("7)", "Train legacy", "Old baseline training")
    _print_line("8)", "Evaluate legacy", "Old baseline evaluation")
    _print_line("9)", "Run tests", "Execute unit tests")

    mode = _prompt_choice(
        "Enter number (0-9): ",
        options,
        "Invalid choice. Please enter 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9.",
    )
    if mode in {"exit", "train", "eval", "play-extended", "plots-extended", "train-legacy", "eval-legacy", "test"}:
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
        if Path(q_default).exists():
            print(f"Using default Q-table: {q_default}")
            args.extend(["--q-table", q_default])
        else:
            q_path = input(f"Q-table path [{q_default}] (required): ").strip()
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
    if args.mode == "train-legacy":
        return run_train_legacy(args)
    if args.mode == "eval-legacy":
        return run_eval_legacy(args)
    if args.mode == "train-extended":
        return run_train_extended(args)
    if args.mode == "play-extended":
        return run_play_extended(args)
    if args.mode == "plots-extended":
        return run_plots_extended(args)
    if args.mode == "test":
        return run_tests()

    parser.error(f"Unknown mode: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
