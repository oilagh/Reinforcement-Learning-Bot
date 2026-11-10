"""
Observation encoding for 1v1. The same `encode_obs` is used in training (from RLGym GameStates) and in
the live RLBot bot (from game packets), so the two must only ever be fed through `CarView`.

Everything is encoded from the agent's team perspective: orange agents see a mirrored field so the policy
always attacks toward +y.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from rlgym.api import AgentID, ObsBuilder
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import BOOST_LOCATIONS, ORANGE_TEAM

POS_COEF = 1 / 2300
VEL_COEF = 1 / 2300
ANG_VEL_COEF = 1 / math.pi
NUM_PADS = len(BOOST_LOCATIONS)

BALL_OBS_SIZE = 9
CAR_OBS_SIZE = 25
OBS_SIZE = BALL_OBS_SIZE + 8 + NUM_PADS + 2 * CAR_OBS_SIZE


@dataclass
class CarView:
    position: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    rotation_mtx: np.ndarray  # columns: forward, right, up
    boost: float  # 0-100
    on_ground: bool
    has_flip: bool
    is_demoed: bool

    @staticmethod
    def from_rlgym(car: Car, inverted: bool) -> "CarView":
        phys = car.inverted_physics if inverted else car.physics
        return CarView(phys.position, phys.linear_velocity, phys.angular_velocity, phys.rotation_mtx,
                       car.boost_amount, car.on_ground, car.has_flip, car.is_demoed)


def _car_obs(car: CarView, target_pos: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
    rot = car.rotation_mtx
    # Target relative to the car, expressed in the car's local frame (forward, right, up)
    rel_pos = rot.T @ (target_pos - car.position)
    rel_vel = rot.T @ (target_vel - car.linear_velocity)
    return np.concatenate([
        car.position * POS_COEF,
        rot[:, 0],
        rot[:, 2],
        car.linear_velocity * VEL_COEF,
        car.angular_velocity * ANG_VEL_COEF,
        rel_pos * POS_COEF,
        rel_vel * VEL_COEF,
        [car.boost / 100, float(car.on_ground), float(car.has_flip), float(car.is_demoed)],
    ])


def encode_obs(ball: PhysicsObject, me: CarView, opponent: CarView, pads_available: np.ndarray,
               prev_action: np.ndarray) -> np.ndarray:
    """All inputs must already be in the agent's team perspective (see module docstring)."""
    obs = np.concatenate([
        ball.position * POS_COEF,
        ball.linear_velocity * VEL_COEF,
        ball.angular_velocity * ANG_VEL_COEF,
        prev_action,
        pads_available,
        _car_obs(me, ball.position, ball.linear_velocity),
        _car_obs(opponent, me.position, me.linear_velocity),
    ]).astype(np.float32)
    return obs


class OneVsOneObs(ObsBuilder[AgentID, np.ndarray, GameState, Tuple[str, int]]):
    """Reads previous actions from shared_info["prev_actions"], written by RecordingLookupTableAction."""

    def get_obs_space(self, agent: AgentID) -> Tuple[str, int]:
        return "real", OBS_SIZE

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        prev_actions = shared_info.get("prev_actions", {})
        obs = {}
        for agent in agents:
            car = state.cars[agent]
            inverted = car.team_num == ORANGE_TEAM
            ball = state.inverted_ball if inverted else state.ball
            timers = state.inverted_boost_pad_timers if inverted else state.boost_pad_timers
            opponent = next(c for a, c in state.cars.items() if a != agent)
            obs[agent] = encode_obs(
                ball,
                CarView.from_rlgym(car, inverted),
                CarView.from_rlgym(opponent, inverted),
                (timers == 0).astype(np.float32),
                prev_actions.get(agent, np.zeros(8, dtype=np.float32)),
            )
        return obs
