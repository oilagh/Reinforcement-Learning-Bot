# RocketBot

A 1v1 Rocket League bot trained from scratch with reinforcement learning (PPO self-play). It trains
headlessly in [RocketSim](https://github.com/ZealanL/RocketSim) through [RLGym](https://rlgym.org), so
training runs on any machine (including macOS), and it plays in the real game through
[RLBot v5](https://rlbot.org).

RLBot is for offline matches only (exhibitions, LAN, bot tournaments). Never use a bot in online matches.

## Setup

Requires Python 3.12. RLGym needs `numpy<2`, which has no wheels for Python 3.13.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Training

```bash
.venv/bin/python train.py --run my_bot
```

- Ctrl+C saves a checkpoint and exits. Running the same command again resumes.
- Everything for a run lives in `runs/<name>/`: `config.json` (all hyperparameters, editable between
  restarts), `log.csv`, `checkpoints/`, `latest.pt` (to resume), and `policy.pt` (the exported bot).
- `--workers N` sets the number of collection processes. The default is CPU cores minus 2. Use fewer if
  the machine is needed for other work.

### Reading the log line

| column | meaning | what you want to see |
|---|---|---|
| `sps` | agent steps per second | ~20k on an M2 |
| `rew` | mean reward per step | rises, then jumps around when the reward curriculum changes stage |
| `ep` | mean episode length | drops as the bots learn to score (episodes end on goals) |
| `goals/ep` | fraction of episodes that end in a goal | rises steadily |
| `touches/min` | ball touches per minute, both cars | rises quickly early on |
| `ent` | policy entropy (max 4.50 = random) | falls slowly; a crash toward 0 means it collapsed |
| `kl` | how far each update moves the policy | roughly 0.002–0.02 |
| `ev` | how well the critic predicts returns | approaches 1 |
| `win vs past selves` | win rate against frozen older versions | stays above 50%; it should keep beating its old selves |

### What to expect

The skill timeline is approximate, at ~20k steps/s on an M2 MacBook:

| steps | time | behaviour |
|---|---|---|
| 10–50M | 10–45 min | drives at the ball and hits it consistently |
| 100–300M | 1.5–4 h | pushes the ball toward the opponent's net, wins kickoffs, scores |
| 500M–1B | 7–14 h | shoots deliberately, defends, uses boost sensibly |
| 2B+ | days | starts flipping into shots, early aerial touches, challenges |

The reward curriculum in `rocketbot/config.py` changes automatically at 150M and 600M steps. Early on it
rewards approaching and touching the ball; later stages mostly reward scoring and moving the ball toward
the opponent's goal. Top community bots were trained on tens of billions of steps on GPUs, so the
longer this trains, the better it gets.

## Watching it play

The real game doesn't run on macOS, so `evaluate.py` plays matches in the simulator and can save a
top-down replay to open in a browser:

```bash
.venv/bin/python evaluate.py --run my_bot                              # vs a scripted ball-chaser
.venv/bin/python evaluate.py --run my_bot --opponent random
.venv/bin/python evaluate.py --run my_bot --opponent runs/older_run    # vs another trained bot
.venv/bin/python evaluate.py --run my_bot --replay replay.html && open replay.html
```

Beating the scripted `chaser` most of the time is a good first milestone.

## The arena (and macOS)

RocketSim normally loads collision meshes dumped from a Windows install of Rocket League. When they're
missing, `rocketbot/arena.py` generates an approximate arena from the official field dimensions: walls,
chamfered corners, goals, and curved floor/wall/ceiling ramps. Cars drive up walls and the ball bounces
and scores as expected, but the curvature isn't an exact copy of the real mesh.

If you get access to a Windows PC with Rocket League, run
[RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper) and put the resulting `.cmf`
files in `collision_meshes/soccar/`. Training picks them up automatically, and the first line of
`train.py`'s output shows which meshes are in use. Fine-tuning for a while on the real meshes before
deploying helps with wall and corner play.

## Playing it in Rocket League

This needs a Windows PC (or Linux with Proton) with Rocket League and RLBot v5 installed.

1. Copy this repository to that machine, and copy `runs/<name>/policy.pt` into `rlbot_deploy/`.
2. Create the bot's environment in the repository root (Python 3.11 or 3.12):
   ```
   python -m venv .venv
   .venv\Scripts\pip install --pre -r rlbot_deploy\requirements.txt
   ```
3. In RLBot, add `rlbot_deploy/bot.toml` as a bot and start a 1v1 match against it.

The bot rebuilds the exact training observation from each game packet and acts every 8 ticks, as in
training. `tests/test_obs_parity.py` checks that the bot and the simulator produce identical
observations; run it after changing `rocketbot/obs.py`.

## Layout

```
train.py                 training entry point
evaluate.py              sim matches, win rates, HTML replays
rocketbot/
  arena.py               mesh loading / procedural arena generation
  env.py                 RLGym environment (1v1, kickoffs + random states)
  obs.py                 observation encoding, shared by training and the live bot
  actions.py             90-action lookup table, 8-tick action repeat
  rewards.py             reward terms
  config.py              hyperparameters and the reward curriculum
  collector.py           parallel self-play workers + opponent pool
  ppo.py                 PPO learner
  policy.py              actor/critic networks, policy export
  replay_viewer.html     template for evaluate.py --replay
rlbot_deploy/            RLBot v5 bot (bot.toml, bot.py)
tests/test_obs_parity.py sim vs. RLBot observation check
```
