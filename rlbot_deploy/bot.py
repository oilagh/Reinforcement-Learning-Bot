"""
RLBot v5 bot that plays Rocket League with a policy trained by train.py.

It rebuilds the exact training observation from each game packet (via rocketbot.obs.encode_obs) and picks
a new action every TICK_SKIP physics ticks, just like in training.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from rlbot.flat import AirState, ControllerState, GamePacket, PlayerInfo  # noqa: E402
from rlbot.managers import Bot  # noqa: E402
from rlgym.rocket_league.api import PhysicsObject  # noqa: E402
from rlgym.rocket_league.common_values import BOOST_LOCATIONS  # noqa: E402
from rlgym.rocket_league.math import euler_to_rotation  # noqa: E402

from rocketbot.actions import LOOKUP_TABLE, TICK_SKIP  # noqa: E402
from rocketbot.obs import CarView, encode_obs  # noqa: E402
from rocketbot.policy import load_exported_policy  # noqa: E402

POLICY_PATH = Path(__file__).resolve().parent / "policy.pt"
INV_VEC = PhysicsObject.INV_VEC
INV_MTX = PhysicsObject.INV_MTX


def _vec(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z], dtype=np.float32)


def _car_view(player: PlayerInfo, inverted: bool) -> CarView:
    phys = player.physics
    pos, vel, ang_vel = _vec(phys.location), _vec(phys.velocity), _vec(phys.angular_velocity)
    rot = euler_to_rotation(np.array([phys.rotation.pitch, phys.rotation.yaw, phys.rotation.roll]))
    if inverted:
        pos, vel, ang_vel, rot = pos * INV_VEC, vel * INV_VEC, ang_vel * INV_VEC, rot * INV_MTX
    on_ground = player.air_state == AirState.OnGround
    # Mirrors RLGym's Car.has_flip: a dodge is available until it's used or the post-jump timer runs out
    has_flip = on_ground or (not player.has_double_jumped and not player.has_dodged
                             and (not player.has_jumped or player.dodge_timeout > 0))
    return CarView(pos, vel, ang_vel, rot, player.boost, on_ground, has_flip, player.demolished_timeout > 0)


class RocketBot(Bot):
    def initialize(self):
        self.policy = load_exported_policy(POLICY_PATH)
        # RLBot's pad order isn't guaranteed to match RLGym's, so map by position
        rlbot_pads = np.array([_vec(p.location)[:2] for p in self.field_info.boost_pads])
        self.pad_index = [int(np.argmin(np.linalg.norm(rlbot_pads - np.array(loc[:2]), axis=1)))
                          for loc in BOOST_LOCATIONS]
        self.prev_action = np.zeros(8, dtype=np.float32)
        self.controls = ControllerState()
        self.last_action_frame = None
        self.logger.info("Loaded policy from %s", POLICY_PATH)

    def get_output(self, packet: GamePacket) -> ControllerState:
        frame = packet.match_info.frame_num
        if self.last_action_frame is not None and 0 <= frame - self.last_action_frame < TICK_SKIP:
            return self.controls
        self.last_action_frame = frame

        if not packet.balls or len(packet.players) <= self.index:
            return self.controls
        me = packet.players[self.index]
        opponents = [p for p in packet.players if p.team != self.team]
        if not opponents:
            return self.controls
        opponent = min(opponents, key=lambda p: np.linalg.norm(_vec(p.physics.location) - _vec(me.physics.location)))

        inverted = self.team == 1
        ball = PhysicsObject()
        ball.position = _vec(packet.balls[0].physics.location)
        ball.linear_velocity = _vec(packet.balls[0].physics.velocity)
        ball.angular_velocity = _vec(packet.balls[0].physics.angular_velocity)
        if inverted:
            ball = ball.inverted()

        pads = np.array([float(packet.boost_pads[i].is_active) if i < len(packet.boost_pads) else 1.0
                         for i in self.pad_index], dtype=np.float32)
        if inverted:
            pads = pads[::-1].copy()

        obs = encode_obs(ball, _car_view(me, inverted), _car_view(opponent, inverted), pads, self.prev_action)
        action, _ = self.policy.act(obs[None], deterministic=True)
        controls = LOOKUP_TABLE[action[0]]
        self.prev_action = controls

        self.controls = ControllerState(
            throttle=float(controls[0]), steer=float(controls[1]), pitch=float(controls[2]),
            yaw=float(controls[3]), roll=float(controls[4]), jump=bool(controls[5]),
            boost=bool(controls[6]), handbrake=bool(controls[7]),
        )
        return self.controls


if __name__ == "__main__":
    RocketBot("rocketbot/ppo-1v1").run()
