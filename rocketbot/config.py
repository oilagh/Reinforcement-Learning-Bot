import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class RewardStage:
    """Reward weights used from `start_step` (total agent steps) onward."""
    start_step: int
    weights: Dict[str, float]


# Curriculum: first learn to chase and hit the ball, then shift the emphasis to scoring and defending.
DEFAULT_STAGES = [
    RewardStage(0, {
        "goal": 10.0,
        "touch": 3.0,
        "vel_player_to_ball": 0.5,
        "vel_ball_to_goal": 1.0,
        "face_ball": 0.1,
        "in_air": 0.03,
        "save_boost": 0.1,
    }),
    RewardStage(150_000_000, {
        "goal": 20.0,
        "touch": 1.0,
        "vel_player_to_ball": 0.25,
        "vel_ball_to_goal": 2.0,
        "face_ball": 0.03,
        "in_air": 0.02,
        "save_boost": 0.2,
        "ball_in_opponent_half": 0.05,
        "demo": 2.0,
    }),
    RewardStage(600_000_000, {
        "goal": 30.0,
        "touch": 0.3,
        "vel_player_to_ball": 0.1,
        "vel_ball_to_goal": 2.0,
        "save_boost": 0.2,
        "ball_in_opponent_half": 0.05,
        "demo": 2.0,
    }),
]


@dataclass
class Config:
    run_name: str = "default"

    # Collection
    num_workers: int = max(1, (os.cpu_count() or 4) - 2)
    envs_per_worker: int = 4
    steps_per_iteration: int = 100_000  # agent steps (both cars count) collected per PPO update
    kickoff_prob: float = 0.4  # remaining resets use random states
    no_touch_timeout_seconds: float = 20.0
    episode_timeout_seconds: float = 300.0

    # Self-play: chance that the orange car is driven by a frozen past policy instead of the current one
    opponent_pool_prob: float = 0.2
    opponent_pool_size: int = 10
    opponent_snapshot_every: int = 50  # iterations

    # Model
    policy_layers: List[int] = field(default_factory=lambda: [512, 512, 256])
    critic_layers: List[int] = field(default_factory=lambda: [512, 512, 256])

    # PPO
    lr: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 2
    minibatch_size: int = 25_000
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    device: str = "cpu"

    # Bookkeeping
    checkpoint_every: int = 10  # iterations
    keep_checkpoints: int = 20

    stages: List[RewardStage] = field(default_factory=lambda: list(DEFAULT_STAGES))

    def reward_weights_at(self, step: int) -> Dict[str, float]:
        weights = self.stages[0].weights
        for stage in self.stages:
            if step >= stage.start_step:
                weights = stage.weights
        return weights

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        d = dict(d)
        d["stages"] = [RewardStage(**s) for s in d.get("stages", [])] or list(DEFAULT_STAGES)
        known = Config.__dataclass_fields__.keys()
        return Config(**{k: v for k, v in d.items() if k in known})
