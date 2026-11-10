"""
Evaluate a trained policy in the simulator and optionally save a replay you can watch in a browser.

    python evaluate.py --run my_bot                          # vs the scripted ball-chaser
    python evaluate.py --run my_bot --opponent random
    python evaluate.py --run my_bot --opponent runs/old_bot/policy.pt
    python evaluate.py --run my_bot --replay replay.html     # save a top-down replay of the first episodes

Every episode starts from a kickoff and ends on a goal or after --timeout seconds. The trained bot plays blue.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rocketbot.actions import LOOKUP_TABLE, NUM_ACTIONS
from rocketbot.env import make_env
from rocketbot.policy import load_exported_policy

ROOT = Path(__file__).resolve().parent
SECONDS_PER_STEP = 8 / 120
VIEWER_TEMPLATE = ROOT / "rocketbot" / "replay_viewer.html"


class RandomAgent:
    def act(self, obs, deterministic=False):
        return np.random.randint(NUM_ACTIONS, size=len(obs)), None


class ChaserAgent:
    """Scripted baseline: drives at the ball, boosting when roughly facing it. Reads the env state directly."""

    # Indices into the lookup table for ground actions: (throttle, steer, boost)
    def __init__(self):
        self.table = {}
        for i, a in enumerate(LOOKUP_TABLE):
            if a[2] == 0 and a[4] == 0 and a[5] == 0 and a[7] == 0:  # no pitch/roll/jump/handbrake
                self.table[(a[0], a[1], a[6])] = i

    def act_for(self, state, agent):
        car = state.cars[agent].physics
        to_ball = state.ball.position - car.position
        local = car.rotation_mtx.T @ to_ball
        angle = np.arctan2(local[1], local[0])
        steer = float(np.clip(np.round(angle * 2), -1, 1))
        boost = float(abs(angle) < 0.3)
        return self.table[(1.0, steer, boost)]


def load_agent(spec: str):
    if spec == "random":
        return RandomAgent()
    if spec == "chaser":
        return ChaserAgent()
    path = Path(spec)
    if path.is_dir():
        path = path / "policy.pt"
    return load_exported_policy(path)


def resolve_policy(run: str, policy: str):
    if policy:
        return load_exported_policy(Path(policy))
    return load_exported_policy(ROOT / "runs" / run / "policy.pt")


def frame(state):
    def car(c):
        p = c.physics
        return [*np.round(p.position, 1).tolist(), *np.round(p.forward, 3).tolist(), round(c.boost_amount)]
    return {"b": np.round(state.ball.position, 1).tolist(), "c": [car(state.cars["blue-0"]), car(state.cars["orange-0"])]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="default", help="run whose policy.pt plays blue")
    p.add_argument("--policy", help="explicit policy.pt path for blue (overrides --run)")
    p.add_argument("--opponent", default="chaser", help="chaser, random, or a policy.pt / run directory")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--timeout", type=float, default=30.0, help="seconds without a touch before an episode ends")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of taking the most likely")
    p.add_argument("--replay", help="write an HTML replay of the first --replay-episodes episodes here")
    p.add_argument("--replay-episodes", type=int, default=5)
    args = p.parse_args()

    blue = resolve_policy(args.run, args.policy)
    orange = load_agent(args.opponent)
    env = make_env({"goal": 1.0}, kickoff_prob=1.0, no_touch_timeout_seconds=args.timeout)
    deterministic = not args.stochastic

    results = {"blue": 0, "orange": 0, "timeout": 0}
    replay = []
    for ep in range(args.episodes):
        obs = env.reset()
        frames = [frame(env.state)]
        steps = 0
        while True:
            a_blue, _ = blue.act(obs["blue-0"][None], deterministic)
            if isinstance(orange, ChaserAgent):
                a_orange = orange.act_for(env.state, "orange-0")
            else:
                a_orange = orange.act(obs["orange-0"][None], deterministic)[0][0]
            obs, _, term, trunc = env.step({"blue-0": np.array([a_blue[0]]), "orange-0": np.array([a_orange])})
            steps += 1
            if ep < args.replay_episodes:
                frames.append(frame(env.state))
            if any(term.values()) or any(trunc.values()):
                break
        if env.state.goal_scored:
            winner = "blue" if env.state.scoring_team == 0 else "orange"
        else:
            winner = "timeout"
        results[winner] += 1
        if args.replay and ep < args.replay_episodes:
            replay.append({"frames": frames, "result": winner})
        print(f"episode {ep + 1:3d}: {winner:7s} after {steps * SECONDS_PER_STEP:5.1f}s")

    decided = results["blue"] + results["orange"]
    print(f"\nBlue (trained) {results['blue']} - {results['orange']} Orange ({args.opponent}), "
          f"{results['timeout']} timeouts")
    if decided:
        print(f"Blue wins {results['blue'] / decided:.0%} of decided episodes")

    if args.replay:
        html = VIEWER_TEMPLATE.read_text().replace("__REPLAY_DATA__", json.dumps({
            "dt": SECONDS_PER_STEP, "blue": args.policy or args.run, "orange": args.opponent, "episodes": replay}))
        Path(args.replay).write_text(html)
        print(f"Replay written to {Path(args.replay).resolve()}")


if __name__ == "__main__":
    main()
