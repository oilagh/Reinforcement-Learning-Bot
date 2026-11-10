from typing import Dict

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import RepeatAction
from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence

from .actions import TICK_SKIP, RecordingLookupTableAction
from .arena import init_rocketsim
from .mutators import RandomStateMutator, WeightedMutator
from .obs import OneVsOneObs
from .rewards import build_reward


def make_env(reward_weights: Dict[str, float], kickoff_prob: float = 0.4, no_touch_timeout_seconds: float = 20.0,
             episode_timeout_seconds: float = 300.0) -> RLGym:
    init_rocketsim()
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            WeightedMutator((KickoffMutator(), kickoff_prob), (RandomStateMutator(), 1 - kickoff_prob)),
        ),
        obs_builder=OneVsOneObs(),
        action_parser=RepeatAction(RecordingLookupTableAction(), repeats=TICK_SKIP),
        reward_fn=build_reward(reward_weights),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(no_touch_timeout_seconds),
            TimeoutCondition(episode_timeout_seconds),
        ),
        transition_engine=RocketSimEngine(),
    )
