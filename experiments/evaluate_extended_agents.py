"""Extended evaluation harness built on top of the external core engine."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "chandan_core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


def _load_module(module_name: str, path: Path):
    """Import ``module_name`` from ``path`` using importlib."""

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_core_qtable(q_table_path: Path, core_q_path: Path) -> None:
    """Write the Q-table in pickle format if the provided path is ``.npy``."""

    if q_table_path.suffix == ".pkl":
        shutil.copy2(q_table_path, core_q_path)
        return
    items = np.load(q_table_path, allow_pickle=True)
    q_table = dict(items.tolist())
    with core_q_path.open("wb") as f:
        pickle.dump(q_table, f)


def _wilson_interval(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    """Approximate a binomial confidence interval (95% by default)."""

    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _evaluate_matchup(
    play_mod,
    games: int,
    seed: int,
    opponent_type: str,
    alternate_start: bool,
) -> dict[str, float]:
    """Play ``games`` matches and collect win/loss/draw stats."""

    wins = 0
    losses = 0
    draws = 0
    lengths: list[int] = []
    env = play_mod.make_env()
    agent = play_mod.load_agent(env)

    for i in range(games):
        agent_player_id = 0 if (not alternate_start or i % 2 == 0) else 1
        winner = play_mod.run_episode_play(
            env,
            agent,
            opponent_type=opponent_type,
            render=False,
            sleep_sec=0.0,
            seed=seed + i,
            agent_player_id=agent_player_id,
        )
        lengths.append(int(getattr(env, "step_count", 0)))

        if winner == -1:
            draws += 1
        elif winner == agent_player_id:
            wins += 1
        else:
            losses += 1
        if (i + 1) % max(10, games // 5) == 0 or (i + 1) == games:
            print(
                f"  {opponent_type}: {i+1}/{games} games | "
                f"win={wins/(i+1):.3f}, loss={losses/(i+1):.3f}, draw={draws/(i+1):.3f}"
            )

    n = max(1, games)
    return {
        "games": float(games),
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "draw_rate": draws / n,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_return": float((wins - losses) / n),
    }


def _aggregate_over_seeds(per_seed: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Mean/std aggregate for each matchup label across evaluated seeds."""

    labels = next(iter(per_seed.values())).keys()
    aggregate: dict[str, dict[str, float]] = {}
    for label in labels:
        metrics = next(iter(per_seed.values()))[label].keys()
        stats: dict[str, float] = {}
        for m in metrics:
            vals = np.array([per_seed[s][label][m] for s in per_seed], dtype=np.float64)
            stats[f"{m}_mean"] = float(np.mean(vals))
            stats[f"{m}_std"] = float(np.std(vals))

        pooled_games = float(sum(per_seed[s][label]["games"] for s in per_seed))
        pooled_wins = float(sum(per_seed[s][label]["wins"] for s in per_seed))
        ci_low, ci_high = _wilson_interval(pooled_wins, pooled_games)
        stats["pooled_games"] = pooled_games
        stats["pooled_wins"] = pooled_wins
        stats["win_rate_ci95_low"] = ci_low
        stats["win_rate_ci95_high"] = ci_high
        aggregate[label] = stats
    return aggregate


def _bar_plot(results: dict[str, float], out_path: Path) -> None:
    """Create a bar chart summarizing win rates per opponent."""

    labels = list(results.keys())
    values = [results[k] for k in labels]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Winrate")
    plt.title("Agent comparison (Chandan core)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    """CLI arguments for the extended evaluation harness."""

    p = argparse.ArgumentParser(description="Evaluate RL vs random/heuristic using Chandan core")
    p.add_argument("--q-table", type=str, default="experiments/results/q_table.pkl")
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--alternate-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=str, default="experiments/results")
    return p.parse_args()


def main() -> None:
    """Evaluate requested seeds/opponents and persist summary artifacts."""

    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_core_qtable(Path(args.q_table), CORE_DIR / "q_table.pkl")
    play_mod = _load_module("chandan_play", CORE_DIR / "play.py")

    seeds = [args.seed + 10_000 * i for i in range(args.num_seeds)]
    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for idx, s in enumerate(seeds, start=1):
        print(f"Seed {idx}/{len(seeds)} ({s})")
        per_seed[f"seed_{s}"] = {
            "RL vs Random": _evaluate_matchup(play_mod, args.games, s + 1000, "random", args.alternate_start),
            "RL vs Heuristic": _evaluate_matchup(play_mod, args.games, s + 2000, "heuristic", args.alternate_start),
        }

    aggregate = _aggregate_over_seeds(per_seed)

    for label, stats in aggregate.items():
        print(
            f"{label}: "
            f"win={stats['win_rate_mean']:.3f}, "
            f"loss={stats['loss_rate_mean']:.3f}, "
            f"draw={stats['draw_rate_mean']:.3f}, "
            f"avg_len={stats['avg_length_mean']:.2f}, "
            f"return={stats['mean_return_mean']:.3f}, "
            f"ci95=[{stats['win_rate_ci95_low']:.3f},{stats['win_rate_ci95_high']:.3f}]"
        )

    _bar_plot({label: stats["win_rate_mean"] for label, stats in aggregate.items()}, out_dir / "head_to_head_winrates.png")

    summary = {
        "config": {
            "games": args.games,
            "base_seed": args.seed,
            "num_seeds": args.num_seeds,
            "alternate_start": args.alternate_start,
            "mode": "chandan_core",
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
