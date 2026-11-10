"""
Parallel experience collection. Each worker process runs several environments and its own copy of the
policy, and returns a fixed-length block of experience per iteration. Environments keep running across
iterations (they're only reset when an episode ends).

Experience is organized as "streams": one per (environment, car), so a worker with E envs has S = 2E streams,
indexed env * 2 + (0 for blue, 1 for orange).
"""
import multiprocessing as mp
import random
import signal
from collections import deque
from typing import Dict, List

import numpy as np
import torch

from .actions import NUM_ACTIONS
from .config import Config
from .obs import OBS_SIZE
from .policy import Policy
from .rewards import build_reward

AGENTS = ("blue-0", "orange-0")


class _Worker:
    def __init__(self, cfg: Config, seed: int):
        from .env import make_env

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(1)

        self.cfg = cfg
        self.envs = [make_env(cfg.reward_weights_at(0), cfg.kickoff_prob, cfg.no_touch_timeout_seconds,
                              cfg.episode_timeout_seconds) for _ in range(cfg.envs_per_worker)]
        self.policy = Policy(OBS_SIZE, NUM_ACTIONS, cfg.policy_layers)
        self.pool: deque = deque(maxlen=cfg.opponent_pool_size)
        self.obs = [env.reset() for env in self.envs]
        self.opponent = [None] * len(self.envs)  # Pool policy driving orange, or None for self-play
        self.ep_len = [0] * len(self.envs)
        self.ep_touches = [[0, 0] for _ in self.envs]

    def set_reward_weights(self, weights: Dict[str, float]) -> None:
        for env in self.envs:
            env.reward_fn = build_reward(weights)

    def add_opponent(self, state_dict) -> None:
        opponent = Policy(OBS_SIZE, NUM_ACTIONS, self.cfg.policy_layers)
        opponent.load_state_dict(state_dict)
        self.pool.append(opponent)

    def _pick_opponent(self):
        if self.pool and random.random() < self.cfg.opponent_pool_prob:
            return random.choice(self.pool)
        return None

    def collect(self, n_steps: int) -> Dict[str, np.ndarray]:
        num_envs = len(self.envs)
        num_streams = 2 * num_envs
        obs_buf = np.zeros((n_steps, num_streams, OBS_SIZE), dtype=np.float32)
        act_buf = np.zeros((n_steps, num_streams), dtype=np.int64)
        logp_buf = np.zeros((n_steps, num_streams), dtype=np.float32)
        rew_buf = np.zeros((n_steps, num_streams), dtype=np.float32)
        done_buf = np.zeros((n_steps, num_streams), dtype=bool)
        trunc_buf = np.zeros((n_steps, num_streams), dtype=bool)
        train_mask = np.ones((n_steps, num_streams), dtype=bool)
        trunc_idx: List[tuple] = []
        trunc_obs: List[np.ndarray] = []
        episodes = []

        for t in range(n_steps):
            obs = np.stack([self.obs[e][a] for e in range(num_envs) for a in AGENTS])
            actions, logps = self.policy.act(obs)
            for e, opponent in enumerate(self.opponent):
                if opponent is not None:
                    s = 2 * e + 1
                    a, lp = opponent.act(obs[s:s + 1])
                    actions[s], logps[s] = a[0], lp[0]
                    train_mask[t, s] = False
            obs_buf[t] = obs
            act_buf[t] = actions
            logp_buf[t] = logps

            for e, env in enumerate(self.envs):
                step_actions = {a: actions[2 * e + i:2 * e + i + 1] for i, a in enumerate(AGENTS)}
                next_obs, rewards, terminated, truncated = env.step(step_actions)
                self.ep_len[e] += 1
                for i, a in enumerate(AGENTS):
                    rew_buf[t, 2 * e + i] = rewards[a]
                    self.ep_touches[e][i] += env.state.cars[a].ball_touches > 0

                term, trunc = any(terminated.values()), any(truncated.values())
                if term or trunc:
                    done_buf[t, 2 * e:2 * e + 2] = True
                    if trunc and not term:
                        trunc_buf[t, 2 * e:2 * e + 2] = True
                        for i, a in enumerate(AGENTS):
                            trunc_idx.append((t, 2 * e + i))
                            trunc_obs.append(next_obs[a])
                    episodes.append({
                        "length": self.ep_len[e],
                        "goal": bool(env.state.goal_scored),
                        "touches": sum(self.ep_touches[e]),
                        "vs_pool": self.opponent[e] is not None,
                        "pool_won": self.opponent[e] is not None and env.state.scoring_team == 1,
                    })
                    self.ep_len[e] = 0
                    self.ep_touches[e] = [0, 0]
                    self.opponent[e] = self._pick_opponent()
                    next_obs = env.reset()
                self.obs[e] = next_obs

        last_obs = np.stack([self.obs[e][a] for e in range(num_envs) for a in AGENTS])
        return {
            "obs": obs_buf, "actions": act_buf, "logp": logp_buf, "rewards": rew_buf,
            "dones": done_buf, "truncated": trunc_buf, "train_mask": train_mask, "last_obs": last_obs,
            "trunc_idx": np.array(trunc_idx, dtype=np.int64).reshape(-1, 2),
            "trunc_obs": np.array(trunc_obs, dtype=np.float32).reshape(-1, OBS_SIZE),
            "episodes": episodes,
        }


def _worker_main(conn, cfg_dict: dict, seed: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # The trainer handles Ctrl+C and shuts workers down cleanly
    worker = _Worker(Config.from_dict(cfg_dict), seed)
    while True:
        cmd, arg = conn.recv()
        if cmd == "collect":
            conn.send(worker.collect(arg))
        elif cmd == "weights":
            worker.policy.load_state_dict(arg)
        elif cmd == "reward_weights":
            worker.set_reward_weights(arg)
        elif cmd == "add_opponent":
            worker.add_opponent(arg)
        elif cmd == "close":
            conn.close()
            return


class Collector:
    def __init__(self, cfg: Config, seed: int = 0):
        ctx = mp.get_context("spawn")
        self.conns = []
        self.procs = []
        for i in range(cfg.num_workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker_main, args=(child, cfg.to_dict(), seed * 1000 + i), daemon=True)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)
        self.streams_per_worker = 2 * cfg.envs_per_worker

    def broadcast(self, cmd: str, arg=None) -> None:
        for conn in self.conns:
            conn.send((cmd, arg))

    def collect(self, steps_per_worker: int) -> List[Dict[str, np.ndarray]]:
        self.broadcast("collect", steps_per_worker)
        return [conn.recv() for conn in self.conns]

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
