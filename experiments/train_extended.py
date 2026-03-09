from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import AdaptiveQTableAgent, PriorityHeuristicAgent, RandomAgent, canonical_state_hash, move_to_action
from src.checkers.env import Checkers6x6Env


@dataclass
class ExtendedTrainConfig:
    episodes: int = 20000
    gamma: float = 0.99
    eval_interval: int = 1000
    eval_games: int = 80
    seed: int = 42


def choose_opponent(curriculum_phase: int, rng: np.random.Generator) -> str:
    if curriculum_phase == 0:
        probs = [0.8, 0.2, 0.0]
    elif curriculum_phase == 1:
        probs = [0.1, 0.8, 0.1]
    else:
        probs = [0.0, 0.2, 0.8]
    return str(rng.choice(["random", "heuristic", "self_play"], p=probs))


def pick_self_play_snapshot(
    snapshots_recent: list[dict], snapshots_hist: list[dict], rng: np.random.Generator
) -> dict | None:
    if not snapshots_recent and not snapshots_hist:
        return None
    if snapshots_recent and snapshots_hist:
        if rng.random() < 0.7:
            return snapshots_recent[int(rng.integers(0, len(snapshots_recent)))]
        return snapshots_hist[int(rng.integers(0, len(snapshots_hist)))]
    pool = snapshots_recent if snapshots_recent else snapshots_hist
    return pool[int(rng.integers(0, len(pool)))]


def _play_opponent_turn(
    env: Checkers6x6Env,
    opponent_type: str,
    rng: np.random.Generator,
    snapshot_q: dict | None,
) -> tuple[float, bool]:
    if not env.legal_moves:
        return 0.0, True

    if opponent_type == "random":
        opp_idx = int(rng.integers(0, len(env.legal_moves)))
    elif opponent_type == "heuristic":
        opp_idx = PriorityHeuristicAgent(player=env.player).select_action(env)
    else:
        if snapshot_q is None:
            opp_idx = int(rng.integers(0, len(env.legal_moves)))
        else:
            s_opp = canonical_state_hash(env._obs())
            vals = [snapshot_q.get((s_opp, move_to_action(m)), 0.0) for m in env.legal_moves]
            opp_idx = int(np.argmax(vals))

    _, r_opp, term, trunc, _ = env.step(opp_idx)
    return -float(r_opp), bool(term or trunc)


def run_episode(
    env: Checkers6x6Env,
    agent: AdaptiveQTableAgent,
    opponent_type: str,
    rng: np.random.Generator,
    gamma: float,
    agent_player: str,
    snapshot_q: dict | None,
    update_q: bool,
) -> tuple[float, str]:
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0
    winner = "draw"

    while not done:
        if env.player != agent_player:
            r, done = _play_opponent_turn(env, opponent_type, rng, snapshot_q)
            ep_reward += r
            if done:
                winner = "r" if agent_player == "b" else "b"
            continue

        if not env.legal_moves:
            winner = "r" if agent_player == "b" else "b"
            break

        s = canonical_state_hash(obs)
        a_idx = agent.select_move_index(obs, env.legal_moves, exploit_only=(not update_q))
        action_key = move_to_action(env.legal_moves[a_idx])

        obs_after_agent, r_agent, term, trunc, info = env.step(a_idx)
        ep_reward += float(r_agent)
        done = bool(term or trunc)

        if done:
            winner = str(info.get("winner", "draw"))
            if update_q:
                agent.update_q(s, action_key, float(r_agent), None, [], gamma)
            break

        # Handle chained captures where same player moves again.
        if env.player == agent_player:
            s_next = canonical_state_hash(obs_after_agent)
            if update_q:
                agent.update_q(s, action_key, float(r_agent), s_next, env.legal_moves, gamma)
            obs = obs_after_agent
            continue

        r_opp_cycle, done_after_opp = _play_opponent_turn(env, opponent_type, rng, snapshot_q)
        cycle_reward = float(r_agent) + float(r_opp_cycle)
        ep_reward += float(r_opp_cycle)

        if done_after_opp:
            winner = "b" if agent_player == "b" else "r"
            if update_q:
                agent.update_q(s, action_key, cycle_reward, None, [], gamma)
            break

        obs = env._obs()
        s_next = canonical_state_hash(obs)
        if update_q:
            agent.update_q(s, action_key, cycle_reward, s_next, env.legal_moves, gamma)

    return ep_reward, winner


def evaluate_agent(
    q: dict,
    opponent_type: str,
    games: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    wins_as_black = 0
    wins_as_red = 0
    n_side = max(1, games // 2)

    for agent_player in ("b", "r"):
        for _ in range(n_side):
            env = Checkers6x6Env(seed=int(rng.integers(0, 1_000_000)))
            agent = AdaptiveQTableAgent(q_table=q, seed=int(rng.integers(0, 1_000_000)))
            _, winner = run_episode(
                env,
                agent,
                opponent_type=opponent_type,
                rng=rng,
                gamma=0.0,
                agent_player=agent_player,
                snapshot_q=None,
                update_q=False,
            )
            if winner == agent_player:
                if agent_player == "b":
                    wins_as_black += 1
                else:
                    wins_as_red += 1

    return wins_as_black / n_side, wins_as_red / n_side


def save_q_table(q: dict, path: Path) -> None:
    arr = np.array(list(q.items()), dtype=object)
    np.save(path, arr, allow_pickle=True)


def train_extended(cfg: ExtendedTrainConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    env = Checkers6x6Env(seed=cfg.seed)
    agent = AdaptiveQTableAgent(seed=cfg.seed)

    rewards = np.zeros(cfg.episodes, dtype=np.float64)
    winners = np.empty(cfg.episodes, dtype="U8")
    episode_lengths = np.zeros(cfg.episodes, dtype=np.int32)

    eval_win_random: list[float] = []
    eval_win_heuristic: list[float] = []
    eval_win_black_heur: list[float] = []
    eval_win_red_heur: list[float] = []
    q_table_sizes: list[int] = []

    recent_snapshots: list[dict] = []
    historical_snapshots: list[dict] = []
    curriculum_phase = 0
    last_black = 0.0
    last_red = 0.0

    for ep in range(cfg.episodes):
        opponent_type = choose_opponent(curriculum_phase, rng)
        snapshot_q = pick_self_play_snapshot(recent_snapshots, historical_snapshots, rng)

        if (last_black - last_red) > 0.15:
            agent_player = str(rng.choice(["b", "r"], p=[0.25, 0.75]))
        else:
            agent_player = str(rng.choice(["b", "r"]))

        ep_reward, winner = run_episode(
            env,
            agent,
            opponent_type=opponent_type,
            rng=rng,
            gamma=cfg.gamma,
            agent_player=agent_player,
            snapshot_q=snapshot_q,
            update_q=True,
        )

        rewards[ep] = ep_reward
        winners[ep] = winner
        episode_lengths[ep] = env._move_count

        if (ep + 1) % cfg.eval_interval == 0:
            wr_black_rand, wr_red_rand = evaluate_agent(agent.q, "random", cfg.eval_games, cfg.seed + ep + 11)
            wr_black_heur, wr_red_heur = evaluate_agent(agent.q, "heuristic", cfg.eval_games, cfg.seed + ep + 97)
            eval_win_random.append(0.5 * (wr_black_rand + wr_red_rand))
            eval_win_heuristic.append(0.5 * (wr_black_heur + wr_red_heur))
            eval_win_black_heur.append(wr_black_heur)
            eval_win_red_heur.append(wr_red_heur)
            q_table_sizes.append(len(agent.q))
            last_black = wr_black_heur
            last_red = wr_red_heur

            if wr_black_heur > 0.75 and wr_red_heur > 0.60 and curriculum_phase < 2:
                curriculum_phase += 1

            start = max(0, ep + 1 - cfg.eval_interval) + 1
            print(
                f"Episodes {start:6d}-{ep+1:6d} | eval-random={eval_win_random[-1]:.3f} "
                f"| eval-heur-b={wr_black_heur:.3f} | eval-heur-r={wr_red_heur:.3f} "
                f"| q={len(agent.q)} | phase={curriculum_phase}"
            )

        if (ep + 1) % 5000 == 0:
            snapshot = copy.deepcopy(agent.q)
            recent_snapshots.append(snapshot)
            if len(recent_snapshots) > 5:
                recent_snapshots.pop(0)
            if (ep + 1) in {5000, 10000, 25000, 50000}:
                historical_snapshots.append(snapshot)

    save_q_table(agent.q, out_dir / "q_table.npy")
    np.savez_compressed(
        out_dir / "training_metrics.npz",
        rewards=rewards,
        winners=winners,
        episode_lengths=episode_lengths,
        num_episodes=cfg.episodes,
        eval_win_random=np.array(eval_win_random, dtype=np.float64),
        eval_win_heuristic=np.array(eval_win_heuristic, dtype=np.float64),
        eval_win_black_heuristic=np.array(eval_win_black_heur, dtype=np.float64),
        eval_win_red_heuristic=np.array(eval_win_red_heur, dtype=np.float64),
        q_table_sizes=np.array(q_table_sizes, dtype=np.int32),
    )
    print(f"Saved extended training artifacts to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended curriculum training for 6x6 checkers")
    p.add_argument("--episodes", type=int, default=20000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--eval-games", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="experiments/results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExtendedTrainConfig(
        episodes=args.episodes,
        gamma=args.gamma,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        seed=args.seed,
    )
    train_extended(cfg, Path(args.out))


if __name__ == "__main__":
    main()
