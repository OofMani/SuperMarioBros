"""Record demonstrations for the reverse curriculum.

A demo is the exact controller byte sent to the NES on every frame, from the
level start to the flag, with game-accurate timing. Because it's raw bytes it
replays identically whatever action set the model uses.

    python make_demo.py overland --ckpt models/ppo/solved.pt   # a trained Mario's overland run
    python make_demo.py pipe     --ckpt models/ppo/solved.pt   # scripted pipe route (see below)

Output goes to demos/<name>.json and is verified by replaying it.

The pipe demo is deliberately clumsy: the checkpoint plays up to the 4th pipe,
then a script lands on the pipe, presses Down, crosses the bonus room (jump onto
the brick block, brake, walk into the side pipe), and the checkpoint finishes the
level from the exit. It only proves the route exists; training should find a
much better line through it.
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

from ppo_model import ActorCritic
from wrappers import make_env

DOWN, LEFT, NOOP, RUN, A = 0b00100000, 0b01000000, 0, 0b10000010, 0b00000001


def load_policy(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # our own file
    model = ActorCritic(ckpt["obs_shape"][0], ckpt["n_actions"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["args"]


def recording_env(train_args):
    """An accurate-timing env whose NES records every controller byte it receives."""
    env = make_env(level=train_args["level"], actions=train_args["actions"],
                   skip=train_args.get("skip", 4), accurate_timing=True, stuck_steps=0)
    nes = env.nes
    log = []
    inner_step = nes.step

    def logged_step(b):
        log.append(int(b))
        return inner_step(b)

    nes.step = logged_step
    return env, nes, log


@torch.no_grad()
def policy_steps(model, env, obs, max_steps, stop=None):
    info = {}
    for _ in range(max_steps):
        a = model.act(torch.from_numpy(obs).unsqueeze(0), greedy=True)[0].item()
        obs, _, term, trunc, info = env.step(a)
        if term or trunc or (stop and stop(info)):
            break
    return obs, info


def refresh_stack(env):
    for _ in range(env.stack):
        env.frames.append(env._process(env.nes.screen))
    return np.stack(env.frames)


def overland(model, train_args):
    env, nes, log = recording_env(train_args)
    obs, _ = env.reset()
    log.clear()
    _, info = policy_steps(model, env, obs, 5000)
    return log, info


def pipe(model, train_args):
    env, nes, log = recording_env(train_args)
    obs, _ = env.reset()
    log.clear()
    # 1. play until Mario is in the air just before the enterable (4th) pipe
    obs, _ = policy_steps(model, env, obs, 500, stop=lambda i: i["x_pos"] >= 881)
    # 2. let go so he lands on the pipe top, then press Down until he's in the bonus room
    for _ in range(120):
        if nes.ram[0x001D] == 0:
            break
        nes.step(NOOP)
    for _ in range(300):
        nes.step(DOWN)
        if nes._x_position < 200 and nes.ram[0x000E] == 0x08:
            break
    if nes._x_position >= 200:
        raise SystemExit("Mario didn't land in the pipe's entry window; this checkpoint needs a different stop point.")
    # 3. cross the bonus room and walk into the side pipe, until he's back outside
    dropped = False
    for f in range(2000):
        x, grounded = int(nes._x_position), nes.ram[0x001D] == 0
        if x < 120:
            b = RUN | A if (f % 60) < 35 else RUN       # jump up onto the brick block
        elif x < 150 and not dropped:
            b = RUN                                       # run across the top of the block
        elif not dropped:
            b = LEFT                                      # brake so he drops to the floor
            dropped = grounded and nes._y_position <= 79
        else:
            b = RUN                                       # walk into the side pipe
        nes.step(b)
        if x > 2000 and nes.ram[0x000E] == 0x08 and grounded:
            break
    # 4. let the checkpoint finish the level from the exit
    env.a_held = False
    _, info = policy_steps(model, env, refresh_stack(env), 1000)
    return log, info


def verify(path, train_args):
    """Replay the bytes from a fresh reset and check the run is reproduced."""
    demo = json.load(open(path))
    env = make_env(level=train_args["level"], actions=train_args["actions"], accurate_timing=True)
    env.reset()
    info, done = {}, False
    for b in demo["bytes"]:
        _, _, done, info = env.nes.step(b)
        if done:
            break
    return info.get("flag_get", False), int(info.get("x_pos", 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["overland", "pipe"])
    p.add_argument("--ckpt", default="models/ppo/solved.pt")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model, train_args = load_policy(args.ckpt)
    log, info = (overland if args.kind == "overland" else pipe)(model, train_args)
    if not info.get("flag_get"):
        raise SystemExit(f"The run didn't reach the flag (x={info.get('x_pos')}); not saving a demo.")
    out = args.out or f"demos/{args.kind}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"kind": args.kind, "source": args.ckpt, "accurate_timing": True,
                   "frames": len(log), "timer_at_flag": int(info.get("time", 0)), "bytes": log}, f)
    flag, x = verify(out, train_args)
    print(f"{args.kind} demo: {len(log)} frames, timer at flag {info.get('time')} -> {out} "
          f"(replay check: {'reaches the flag' if flag else f'FAILED at x={x}'})")


if __name__ == "__main__":
    main()
