"""
Train a 1v1 Rocket League bot with PPO self-play in RocketSim.

    python train.py --run my_bot                # start (or resume) a run
    python train.py --run my_bot --workers 4    # fewer worker processes

Ctrl+C saves a checkpoint and exits; rerunning the same command resumes.
"""
import argparse
import csv
import json
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from rocketbot.arena import init_rocketsim
from rocketbot.collector import Collector
from rocketbot.config import Config
from rocketbot.policy import export_policy
from rocketbot.ppo import PPO

RUNS_DIR = Path(__file__).resolve().parent / "runs"
TICKS_PER_STEP = 8
SECONDS_PER_STEP = TICKS_PER_STEP / 120


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="default", help="run name (directory under runs/)")
    p.add_argument("--workers", type=int, help="worker processes (default: CPU cores - 2)")
    p.add_argument("--envs-per-worker", type=int)
    p.add_argument("--steps-per-iteration", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--device", help="cpu or mps")
    p.add_argument("--max-steps", type=float, default=float("inf"), help="stop after this many total agent steps")
    return p.parse_args()


def load_or_create_config(run_dir: Path, args) -> Config:
    config_path = run_dir / "config.json"
    cfg = Config.from_dict(json.loads(config_path.read_text())) if config_path.exists() else Config(run_name=args.run)
    overrides = {"num_workers": args.workers, "envs_per_worker": args.envs_per_worker,
                 "steps_per_iteration": args.steps_per_iteration, "lr": args.lr, "device": args.device}
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    config_path.write_text(json.dumps(cfg.to_dict(), indent=2))
    return cfg


def save_checkpoint(run_dir: Path, cfg: Config, ppo: PPO, step: int, iteration: int, pool: deque) -> None:
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    state = {"ppo": ppo.state_dict(), "step": step, "iteration": iteration, "pool": list(pool)}
    path = ckpt_dir / f"{step:012d}.pt"
    torch.save(state, path)
    torch.save(state, run_dir / "latest.pt")
    export_policy(ppo.policy.cpu(), run_dir / "policy.pt")
    ppo.policy.to(ppo.device)
    for old in sorted(ckpt_dir.glob("*.pt"))[:-cfg.keep_checkpoints]:
        old.unlink()


def main():
    args = parse_args()
    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_or_create_config(run_dir, args)
    torch.set_num_threads(2)  # Leave the cores to the workers

    mesh_source = init_rocketsim()
    print(f"Arena meshes: {mesh_source}" + ("" if mesh_source == "real" else
          " (approximate; drop real meshes in collision_meshes/soccar/ to use them)"))

    collector = Collector(cfg)
    ppo = PPO(cfg, cfg.num_workers, collector.streams_per_worker)
    pool = deque(maxlen=cfg.opponent_pool_size)
    step, iteration = 0, 0

    latest = run_dir / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=cfg.device, weights_only=False)
        ppo.load_state_dict(state["ppo"])
        step, iteration = state["step"], state["iteration"]
        pool.extend(state["pool"])
        for snapshot in pool:
            collector.broadcast("add_opponent", snapshot)
        print(f"Resumed {args.run} at {step:,} steps (iteration {iteration})")

    steps_per_worker = max(1, cfg.steps_per_iteration // (cfg.num_workers * collector.streams_per_worker))
    reward_weights = None
    stop = False

    def request_stop(*_):
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True
        print("\nStopping after this iteration (Ctrl+C again to quit immediately)...")

    signal.signal(signal.SIGINT, request_stop)

    log_path = run_dir / "log.csv"
    log_file = open(log_path, "a", newline="")
    log_writer = None
    recent_episodes = deque(maxlen=500)

    print(f"{cfg.num_workers} workers x {cfg.envs_per_worker} envs, "
          f"{steps_per_worker * cfg.num_workers * collector.streams_per_worker:,} agent steps per iteration")
    try:
        while not stop and step < args.max_steps:
            weights = cfg.reward_weights_at(step)
            if weights != reward_weights:
                collector.broadcast("reward_weights", weights)
                if reward_weights is not None:
                    print(f"Reward curriculum: switching to {weights}")
                reward_weights = weights

            t0 = time.time()
            collector.broadcast("weights", ppo.policy_state_cpu())
            batches = collector.collect(steps_per_worker)
            t1 = time.time()
            stats = ppo.update(batches)
            t2 = time.time()

            collected = sum(b["obs"].shape[0] * b["obs"].shape[1] for b in batches)
            step += collected
            iteration += 1
            for b in batches:
                recent_episodes.extend(b["episodes"])

            if iteration % cfg.opponent_snapshot_every == 0:
                snapshot = ppo.policy_state_cpu()
                pool.append(snapshot)
                collector.broadcast("add_opponent", snapshot)

            eps = list(recent_episodes)
            pool_eps = [e for e in eps if e["vs_pool"] and e["goal"]]
            row = {
                "iteration": iteration,
                "step": step,
                "sps": collected / (t2 - t0),
                "collect_s": t1 - t0,
                "learn_s": t2 - t1,
                "mean_reward": float(np.mean([b["rewards"][b["train_mask"]].mean() for b in batches])),
                "episode_seconds": float(np.mean([e["length"] for e in eps])) * SECONDS_PER_STEP if eps else 0,
                "goal_rate": float(np.mean([e["goal"] for e in eps])) if eps else 0,
                "touches_per_min": (sum(e["touches"] for e in eps) / max(1, sum(e["length"] for e in eps))
                                    / SECONDS_PER_STEP * 60) if eps else 0,
                "win_rate_vs_pool": (1 - float(np.mean([e["pool_won"] for e in pool_eps]))) if pool_eps else "",
                **stats,
            }
            if log_writer is None:
                log_writer = csv.DictWriter(log_file, fieldnames=list(row))
                if log_path.stat().st_size == 0:
                    log_writer.writeheader()
            log_writer.writerow(row)
            log_file.flush()

            vs_pool = f" | win vs past selves {row['win_rate_vs_pool']:.0%}" if pool_eps else ""
            print(f"it {iteration:5d} | {step / 1e6:8.2f}M steps | {row['sps']:6.0f} sps | "
                  f"rew {row['mean_reward']:+.4f} | ep {row['episode_seconds']:4.1f}s | "
                  f"goals/ep {row['goal_rate']:.2f} | touches/min {row['touches_per_min']:4.1f} | "
                  f"ent {stats['entropy']:.2f} | kl {stats['kl']:.4f} | ev {stats['explained_var']:+.2f}{vs_pool}")

            if iteration % cfg.checkpoint_every == 0:
                save_checkpoint(run_dir, cfg, ppo, step, iteration, pool)
    finally:
        save_checkpoint(run_dir, cfg, ppo, step, iteration, pool)
        print(f"Saved checkpoint at {step:,} steps to {run_dir}")
        log_file.close()
        collector.close()


if __name__ == "__main__":
    main()
