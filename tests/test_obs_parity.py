"""
Checks that the RLBot bot (rlbot_deploy/bot.py) builds the same observation from a game packet as training
builds from the simulator, for both teams and arbitrary car orientations. RLBot itself only runs on
Windows/Linux, so its flatbuffer types are stubbed with plain objects carrying the same field names.
"""
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _install_rlbot_stubs():
    flat = types.ModuleType("rlbot.flat")
    flat.AirState = NS(OnGround=0, InAir=4)
    flat.ControllerState = lambda **kw: NS(**kw)
    flat.GamePacket = flat.PlayerInfo = object
    managers = types.ModuleType("rlbot.managers")

    class Bot:
        def __init__(self, *_):
            pass

    managers.Bot = Bot
    sys.modules.update({"rlbot": types.ModuleType("rlbot"), "rlbot.flat": flat, "rlbot.managers": managers})


_install_rlbot_stubs()
sys.path.insert(0, str(ROOT / "rlbot_deploy"))
import bot as deploy_bot  # noqa: E402
from rlgym.rocket_league.common_values import BOOST_LOCATIONS, DOUBLEJUMP_MAX_DELAY  # noqa: E402

from rocketbot.actions import NUM_ACTIONS  # noqa: E402
from rocketbot.config import Config  # noqa: E402
from rocketbot.env import make_env  # noqa: E402
from rocketbot.obs import OBS_SIZE  # noqa: E402
from rocketbot.policy import Policy, export_policy  # noqa: E402


def _v(a):
    return NS(x=float(a[0]), y=float(a[1]), z=float(a[2]))


def _packet_player(car, team):
    p = car.physics
    pitch, yaw, roll = p.euler_angles
    return NS(
        physics=NS(location=_v(p.position), velocity=_v(p.linear_velocity),
                   angular_velocity=_v(p.angular_velocity), rotation=NS(pitch=pitch, yaw=yaw, roll=roll)),
        air_state=0 if car.on_ground else 4,
        has_jumped=car.has_jumped, has_double_jumped=car.has_double_jumped, has_dodged=car.has_flipped,
        dodge_timeout=DOUBLEJUMP_MAX_DELAY - car.air_time_since_jump if car.has_jumped else -1,
        demolished_timeout=car.demo_respawn_timer if car.is_demoed else -1,
        boost=car.boost_amount, team=team,
    )


def _make_bot(team, index, prev_action):
    b = deploy_bot.RocketBot.__new__(deploy_bot.RocketBot)
    b.team, b.index = team, index
    # Shuffle RLBot's pad order to exercise the position-based mapping
    order = np.random.permutation(len(BOOST_LOCATIONS))
    b.field_info = NS(boost_pads=[NS(location=_v(BOOST_LOCATIONS[i])) for i in order])
    b.logger = NS(info=lambda *a: None)
    b.initialize()
    b.prev_action = prev_action
    return b, order


def test_obs_parity(steps=400):
    np.random.seed(0)
    deploy_bot.POLICY_PATH = Path(tempfile.mkdtemp()) / "policy.pt"
    export_policy(Policy(OBS_SIZE, NUM_ACTIONS, [64]), deploy_bot.POLICY_PATH)
    env = make_env(Config().reward_weights_at(0), kickoff_prob=0.0)
    obs = env.reset()
    agents = ["blue-0", "orange-0"]
    checked = 0
    for _ in range(steps):
        state = env.state
        players = [_packet_player(state.cars[a], i) for i, a in enumerate(agents)]
        for idx, agent in enumerate(agents):
            bot, order = _make_bot(idx, idx, env.shared_info["prev_actions"][agent])
            pad_states = [NS(is_active=bool(state.boost_pad_timers[i] == 0)) for i in order]
            ball = state.ball
            packet = NS(match_info=NS(frame_num=0), players=players, boost_pads=pad_states,
                        balls=[NS(physics=NS(location=_v(ball.position), velocity=_v(ball.linear_velocity),
                                             angular_velocity=_v(ball.angular_velocity)))])
            captured = {}
            original = deploy_bot.encode_obs
            deploy_bot.encode_obs = lambda *a: captured.setdefault("obs", original(*a))
            try:
                bot.get_output(packet)
            finally:
                deploy_bot.encode_obs = original
            np.testing.assert_allclose(captured["obs"], obs[agent], atol=1e-4, err_msg=f"{agent} obs mismatch")
            checked += 1
        actions = {a: np.array([np.random.randint(90)]) for a in agents}
        obs, _, term, trunc = env.step(actions)
        if any(term.values()) or any(trunc.values()):
            obs = env.reset()
    print(f"obs parity OK over {checked} agent observations")


if __name__ == "__main__":
    test_obs_parity()
