import random
from typing import Any, Dict, Tuple

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_RADIUS, CEILING_Z


class WeightedMutator(StateMutator[GameState]):
    """Picks one of several mutators at random each reset."""

    def __init__(self, *mutators_and_weights: Tuple[StateMutator[GameState], float]):
        self.mutators = [m for m, _ in mutators_and_weights]
        self.weights = [w for _, w in mutators_and_weights]

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        random.choices(self.mutators, self.weights)[0].apply(state, shared_info)


class RandomStateMutator(StateMutator[GameState]):
    """
    Random ball and car positions/velocities. Exposes the policy to situations kickoffs alone rarely reach
    (ball in the air, ball on the wall side, defending a fast shot, etc).
    """

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        ball = state.ball
        ball.position = np.array([random.uniform(-3600, 3600), random.uniform(-4600, 4600),
                                  random.uniform(BALL_RADIUS + 2, CEILING_Z - 400)], dtype=np.float32)
        ball.linear_velocity = _random_vec(random.uniform(0, 2500))
        ball.angular_velocity = _random_vec(random.uniform(0, 4))

        for car in state.cars.values():
            yaw = random.uniform(-np.pi, np.pi)
            speed = random.uniform(0, 1800)
            car.physics.position = np.array([random.uniform(-3800, 3800), random.uniform(-4800, 4800), 17],
                                            dtype=np.float32)
            car.physics.linear_velocity = np.array([np.cos(yaw) * speed, np.sin(yaw) * speed, 0], dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0, yaw, 0], dtype=np.float32)
            car.boost_amount = random.uniform(0, 100)


def _random_vec(magnitude: float) -> np.ndarray:
    v = np.random.normal(size=3)
    return (v / np.linalg.norm(v) * magnitude).astype(np.float32)
