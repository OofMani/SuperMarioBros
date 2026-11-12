"""Time how fast a trained Mario finishes the level (speedrun phase 1: baseline).

    python speedrun_eval.py                          # times the newest record (pipe, speedrun, then ppo/solved.pt)
    python speedrun_eval.py --ckpt models/ppo/best.pt
    python speedrun_eval.py --delays 20              # also check 20 random start delays
    python speedrun_eval.py --save runs/speedrun/baseline.json

The main run starts on frame 0 with no delay, like a real speedrun, and
always picks the network's top action, so the result is exactly repeatable.
The button presses of that run are saved too, so later phases can replay it.

Times:
  frames      NES frames from level start to grabbing the flagpole (the main score)
  seconds     frames / 60.0988 (NTSC NES frame rate)
  timer used  in-game timer at the start (400) minus the timer at the flag
  frame rule  SMB1 rounds level ends to 21-frame blocks; frames // 21 is an
              approximation (the real boundary depends on the global frame counter)
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

from ppo_model import ActorCritic, pick_device
from wrappers import make_env, play_env_kwargs

NES_FPS = 60.0988
FRAME_RULE = 21


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own file
    model = ActorCritic(ckpt["obs_shape"][0], ckpt["n_actions"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["args"]


@torch.no_grad()
def run(model, device, env, delay=0, max_steps=5000):
    """One greedy run. Returns a result dict with timing and the action list."""
    obs, _ = env.reset()
    actions = []
    info = {}
    start_time = int(env.nes._time)
    for t in range(max_steps):
        if t < delay:
            a = 0
        else:
            a = model.act(torch.from_numpy(obs).unsqueeze(0).to(device), greedy=True)[0].item()
        actions.append(a)
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    frames = info.get("frames", 0)
    return {
        "delay": delay,
        "cleared": bool(info.get("flag_get", False)),
        "pipe": bool(info.get("pipe", False)),
        "x_pos": int(info.get("x_pos", 0)),
        "frames": int(frames),
        "seconds": round(frames / NES_FPS, 3),
        "timer_used": start_time - int(info.get("time", start_time)),
        "timer_at_flag": int(info.get("time", 0)),
        "frame_rules": frames // FRAME_RULE,
        "skipped_frames": int(info.get("skipped_frames", 0)),
        "actions": actions,
    }


def describe(r):
    if not r["cleared"]:
        return f"did not clear (died or stuck at x={r['x_pos']})"
    note = "  (WARNING: went through a skipped animation, time not game-accurate)" if r["skipped_frames"] else ""
    note = (" via the PIPE" if r["pipe"] else " overland") + note
    return (f"{r['frames']} frames = {r['seconds']:.2f} s | timer at flag {r['timer_at_flag']} (used {r['timer_used']}) | "
            f"~{r['frame_rules']} frame rules{note}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--delays", type=int, default=0, help="also test this many random start delays (0-30 steps)")
    p.add_argument("--save", default=None, help="write the results (and the frame-0 run's actions) to this JSON file")
    p.add_argument("--device", default="cpu", help="cpu keeps runs bit-for-bit repeatable")
    args = p.parse_args()

    ckpt_path = args.ckpt or next(
        (f for f in ("models/pipe/fastest.pt", "models/speedrun/fastest.pt", "models/ppo/solved.pt", "models/ppo/best.pt")
         if os.path.exists(f)), None)
    if ckpt_path is None:
        raise SystemExit("No checkpoint found. Pass --ckpt.")
    device = pick_device(args.device)
    model, train_args = load(ckpt_path, device)
    env = make_env(**play_env_kwargs(train_args))

    print(f"checkpoint: {ckpt_path}   level {train_args['level']}")
    main_run = run(model, device, env, delay=0)
    print(f"frame-0 run: {describe(main_run)}")

    delay_runs = []
    if args.delays:
        rng = np.random.default_rng(0)
        for d in rng.integers(1, 31, size=args.delays):
            delay_runs.append(run(model, device, env, delay=int(d)))
        cleared = [r for r in delay_runs if r["cleared"]]
        print(f"random delays: cleared {len(cleared)}/{len(delay_runs)}")
        if cleared:
            # subtract the waiting so we compare how fast Mario plays, not how long he waited
            played = [r["frames"] - r["delay"] * env.skip for r in cleared]
            print(f"  frames after the delay: best {min(played)}, median {int(np.median(played))}, worst {max(played)}")
    env.close()

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump({"checkpoint": ckpt_path, "level": train_args["level"],
                       "actions_set": train_args["actions"], "frame0": main_run,
                       "delays": [{k: v for k, v in r.items() if k != "actions"} for r in delay_runs]},
                      f, indent=1)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
