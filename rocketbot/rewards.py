"""
Reward terms. Each returns a per-agent float; `build_reward` combines them with weights from the config.
Directional terms are computed from the agent's own team perspective, so they're symmetric for both teams.
"""
from typing import Any, Dict, List

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import (BACK_NET_Y, BALL_MAX_SPEED, BLUE_TEAM, CAR_MAX_SPEED, CEILING_Z)
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward

ORANGE_GOAL = np.array([0, BACK_NET_Y, 320], dtype=np.float32)
BLUE_GOAL = np.array([0, -BACK_NET_Y, 320], dtype=np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros_like(v)


class _PerAgentReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(self._reward(agent, state)) for agent in agents}

    def _reward(self, agent: AgentID, state: GameState) -> float:
        raise NotImplementedError


class TouchReward(_PerAgentReward):
    """Rewards touching the ball, more for touches higher up to encourage aerial play."""

    def _reward(self, agent, state):
        if state.cars[agent].ball_touches == 0:
            return 0.
        return 1. + state.ball.position[2] / CEILING_Z


class VelocityPlayerToBall(_PerAgentReward):
    def _reward(self, agent, state):
        car = state.cars[agent].physics
        return np.dot(_unit(state.ball.position - car.position), car.linear_velocity) / CAR_MAX_SPEED


class VelocityBallToGoal(_PerAgentReward):
    def _reward(self, agent, state):
        goal = ORANGE_GOAL if state.cars[agent].team_num == BLUE_TEAM else BLUE_GOAL
        ball = state.ball
        return np.dot(_unit(goal - ball.position), ball.linear_velocity) / BALL_MAX_SPEED


class FaceBall(_PerAgentReward):
    def _reward(self, agent, state):
        car = state.cars[agent].physics
        return np.dot(_unit(state.ball.position - car.position), car.forward)


class InAir(_PerAgentReward):
    def _reward(self, agent, state):
        return float(not state.cars[agent].on_ground)


class SaveBoost(_PerAgentReward):
    """Concave in boost amount: the first bits of boost are worth more than the last."""

    def _reward(self, agent, state):
        return np.sqrt(state.cars[agent].boost_amount / 100)


class BallInOpponentHalf(_PerAgentReward):
    """Small positional pressure term: positive while the ball is on the opponent's side."""

    def _reward(self, agent, state):
        y = state.ball.position[1]
        return np.sign(y) if state.cars[agent].team_num == BLUE_TEAM else -np.sign(y)


class Demo(_PerAgentReward):
    def _reward(self, agent, state):
        victim = state.cars[agent].bump_victim_id
        return float(victim is not None and state.cars[victim].is_demoed
                     and state.cars[victim].team_num != state.cars[agent].team_num)


REWARD_TERMS = {
    "goal": GoalReward,
    "touch": TouchReward,
    "vel_player_to_ball": VelocityPlayerToBall,
    "vel_ball_to_goal": VelocityBallToGoal,
    "face_ball": FaceBall,
    "in_air": InAir,
    "save_boost": SaveBoost,
    "ball_in_opponent_half": BallInOpponentHalf,
    "demo": Demo,
}


def build_reward(weights: Dict[str, float]) -> CombinedReward:
    unknown = set(weights) - set(REWARD_TERMS)
    if unknown:
        raise ValueError(f"Unknown reward terms: {sorted(unknown)}")
    return CombinedReward(*[(REWARD_TERMS[name](), w) for name, w in weights.items() if w != 0])

