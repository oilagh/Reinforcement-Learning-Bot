from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn


def mlp(in_size: int, layers: List[int], out_size: int) -> nn.Sequential:
    modules = []
    for size in layers:
        modules += [nn.Linear(in_size, size), nn.ReLU()]
        in_size = size
    modules.append(nn.Linear(in_size, out_size))
    return nn.Sequential(*modules)


class Policy(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, layers: List[int]):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.layers = list(layers)
        self.net = mlp(obs_size, layers, num_actions)
        # Small final layer => near-uniform initial action distribution, which helps early exploration
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.net(obs))

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        dist = self(torch.as_tensor(obs, dtype=torch.float32))
        actions = dist.probs.argmax(-1) if deterministic else dist.sample()
        return actions.numpy(), dist.log_prob(actions).numpy()


class Critic(nn.Module):
    def __init__(self, obs_size: int, layers: List[int]):
        super().__init__()
        self.net = mlp(obs_size, layers, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def export_policy(policy: Policy, path: Path) -> None:
    """Writes just what the live bot needs: weights and architecture."""
    torch.save({
        "state_dict": policy.state_dict(),
        "obs_size": policy.obs_size,
        "num_actions": policy.num_actions,
        "layers": policy.layers,
    }, path)


def load_exported_policy(path: Path) -> Policy:
    data = torch.load(path, map_location="cpu", weights_only=True)
    policy = Policy(data["obs_size"], data["num_actions"], data["layers"])
    policy.load_state_dict(data["state_dict"])
    policy.eval()
    return policy
