from typing import Dict, List

import numpy as np
import torch

from .actions import NUM_ACTIONS
from .config import Config
from .obs import OBS_SIZE
from .policy import Critic, Policy


class RunningStd:
    """Running variance (Welford/Chan merge) of a scalar quantity."""

    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean, batch_var, n = float(x.mean()), float(x.var()), x.size
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var) + 1e-8)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class PPO:
    def __init__(self, cfg: Config, num_workers: int, streams_per_worker: int):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.policy = Policy(OBS_SIZE, NUM_ACTIONS, cfg.policy_layers).to(self.device)
        self.critic = Critic(OBS_SIZE, cfg.critic_layers).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.critic.parameters()), lr=cfg.lr)
        # Rewards are scaled by the std of the discounted return so the value targets stay O(1)
        self.return_std = RunningStd()
        self.return_acc = np.zeros((num_workers, streams_per_worker), dtype=np.float64)

    def policy_state_cpu(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.policy.state_dict().items()}

    @torch.no_grad()
    def _values(self, obs: np.ndarray) -> np.ndarray:
        out = []
        flat = obs.reshape(-1, OBS_SIZE)
        for i in range(0, len(flat), 50_000):
            out.append(self.critic(torch.as_tensor(flat[i:i + 50_000], device=self.device)).cpu().numpy())
        return np.concatenate(out).reshape(obs.shape[:-1]) if out else np.zeros(obs.shape[:-1], dtype=np.float32)

    def _scale_rewards(self, worker: int, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        acc = self.return_acc[worker]
        discounted = np.empty_like(rewards, dtype=np.float64)
        for t in range(len(rewards)):
            acc = acc * self.cfg.gamma + rewards[t]
            discounted[t] = acc
            acc = np.where(dones[t], 0.0, acc)
        self.return_acc[worker] = acc
        self.return_std.update(discounted)
        return rewards / self.return_std.std

    def _gae(self, batch: Dict[str, np.ndarray], rewards: np.ndarray):
        cfg = self.cfg
        values = self._values(batch["obs"])
        next_values = np.empty_like(values)
        next_values[:-1] = values[1:]
        next_values[-1] = self._values(batch["last_obs"])
        # Episode boundaries: no bootstrap after a goal, bootstrap from the final obs after a timeout
        next_values[batch["dones"]] = 0.0
        if len(batch["trunc_idx"]):
            t, s = batch["trunc_idx"].T
            next_values[t, s] = self._values(batch["trunc_obs"])

        advantages = np.zeros_like(values)
        gae = np.zeros(values.shape[1], dtype=np.float32)
        for t in reversed(range(len(values))):
            delta = rewards[t] + cfg.gamma * next_values[t] - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * (~batch["dones"][t]) * gae
            advantages[t] = gae
        return advantages, advantages + values, values

    def update(self, batches: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
        cfg = self.cfg
        obs, actions, old_logp, advantages, returns, values = [], [], [], [], [], []
        for w, batch in enumerate(batches):
            rewards = self._scale_rewards(w, batch["rewards"], batch["dones"])
            adv, ret, val = self._gae(batch, rewards)
            mask = batch["train_mask"]
            obs.append(batch["obs"][mask])
            actions.append(batch["actions"][mask])
            old_logp.append(batch["logp"][mask])
            advantages.append(adv[mask])
            returns.append(ret[mask])
            values.append(val[mask])

        def cat(xs):
            return torch.as_tensor(np.concatenate(xs), device=self.device)

        obs, actions, old_logp = cat(obs), cat(actions), cat(old_logp)
        advantages, returns, values_np = cat(advantages), cat(returns), np.concatenate(values)
        n = len(obs)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0}
        num_updates = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                adv = advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                dist = self.policy(obs[idx])
                logp = dist.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                entropy = dist.entropy().mean()
                value_loss = (self.critic(obs[idx]) - returns[idx]).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.critic.parameters()),
                                               cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = logp - old_logp[idx]
                    stats["kl"] += ((ratio - 1) - log_ratio).mean().item()
                    stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                num_updates += 1

        stats = {k: v / max(num_updates, 1) for k, v in stats.items()}
        returns_np = returns.cpu().numpy()
        stats["explained_var"] = float(1 - np.var(returns_np - values_np) / (np.var(returns_np) + 1e-8))
        stats["trained_steps"] = n
        return stats

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "return_std": self.return_std.state_dict(),
        }

    def load_state_dict(self, d: dict) -> None:
        self.policy.load_state_dict(d["policy"])
        self.critic.load_state_dict(d["critic"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.return_std.load_state_dict(d["return_std"])
