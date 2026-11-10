from typing import Any, Dict, List

import numpy as np
from rlgym.api import AgentID
from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.api import GameState

LOOKUP_TABLE = LookupTableAction.make_lookup_table().astype(np.float32)
NUM_ACTIONS = len(LOOKUP_TABLE)
TICK_SKIP = 8


class RecordingLookupTableAction(LookupTableAction):
    """LookupTableAction that also stores each agent's last controls in shared_info for the obs builder."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        shared_info["prev_actions"] = {agent: np.zeros(8, dtype=np.float32) for agent in agents}

    def parse_actions(self, actions: Dict[AgentID, np.ndarray], state: GameState,
                      shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        parsed = super().parse_actions(actions, state, shared_info)
        shared_info["prev_actions"] = {agent: a.reshape(8).astype(np.float32) for agent, a in parsed.items()}
        return parsed
