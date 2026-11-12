"""Watch a trained Mario's run at real NES speed (60 fps), and optionally save it as a video.

    python watch_run.py                              # newest record: pipe, then speedrun, then ppo/solved.pt
    python watch_run.py --ckpt models/ppo/solved.pt
    python watch_run.py --mp4 runs/speedrun/best_run.mp4          # save a video too
    python watch_run.py --mp4 runs/speedrun/best_run.mp4 --no-window   # just the video
    python watch_run.py --speed 0.5                  # slow motion
    python watch_run.py --ckpt a.pt b.pt c.pt        # play all quietly, show the one that got closest

Every NES frame is shown (not just one per agent step), with a frame counter, and a
virtual NES controller underneath that lights up exactly the buttons the AI is holding
on that frame (a white ring marks a fresh press). Press q in the window to quit.
"""
import argparse
import json
import os
import shutil
import time
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

from ppo_model import ActorCritic, pick_device
from wrappers import ACTION_SETS, make_env, play_env_kwargs

NES_FPS = 60.0988

# BGR colours
PAD_BG, BODY, DARK, DIM = (30, 30, 30), (200, 200, 200), (70, 70, 70), (120, 120, 120)
LIT_DPAD, LIT_AB, AB_OFF, RING = (0, 215, 255), (40, 40, 235), (95, 95, 110), (255, 255, 255)


def draw_gamepad(width, s, held, fresh):
    """An NES controller, `width` pixels wide and scaled by s. held/fresh are sets of button names."""
    h = 40 * s
    pad = np.full((h, width, 3), PAD_BG, dtype=np.uint8)
    cx, cy, arm, t = 34 * s, h // 2, 11 * s, 4 * s
    cv2.rectangle(pad, (6 * s, 4 * s), (width - 6 * s, h - 4 * s), BODY, -1)
    # D-pad: one rectangle per direction, lit when held
    arms = {"up": ((cx - t, cy - arm), (cx + t, cy - t)), "down": ((cx - t, cy + t), (cx + t, cy + arm)),
            "left": ((cx - arm, cy - t), (cx - t, cy + t)), "right": ((cx + t, cy - t), (cx + arm, cy + t))}
    cv2.rectangle(pad, (cx - t, cy - t), (cx + t, cy + t), DARK, -1)
    for name, (p1, p2) in arms.items():
        cv2.rectangle(pad, p1, p2, LIT_DPAD if name in held else DARK, -1)
        if name in fresh:
            cv2.rectangle(pad, p1, p2, RING, max(1, s // 2))
    # Select / Start (the AI never uses them, drawn dimmed)
    for i, name in enumerate(("SELECT", "START")):
        x = width // 2 - 22 * s + i * 26 * s
        cv2.rectangle(pad, (x, cy - 2 * s), (x + 16 * s, cy + 2 * s), DIM, -1)
        cv2.putText(pad, name, (x - s, cy + 9 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.12 * s, DIM, 1)
    # B and A
    for name, x, role in (("B", width - 58 * s, "run"), ("A", width - 30 * s, "jump")):
        cv2.circle(pad, (x, cy - 2 * s), 9 * s, LIT_AB if name in held else AB_OFF, -1)
        if name in fresh:
            cv2.circle(pad, (x, cy - 2 * s), 9 * s, RING, max(1, s // 2))
        cv2.putText(pad, name, (x - 3 * s, cy + 1 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.25 * s, (255, 255, 255), max(1, s // 2))
        cv2.putText(pad, role, (x - 6 * s, cy + 15 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.13 * s, DARK, 1)
    # what's held, in words
    words = "+".join(b for b in ("left", "right", "up", "down", "B", "A") if b in held) or "nothing"
    cv2.putText(pad, words, (cx + 16 * s, h - 7 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.15 * s, DARK, 1)
    return pad


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # our own file
    model = ActorCritic(ckpt["obs_shape"][0], ckpt["n_actions"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["args"]


@torch.no_grad()
def closeness(path, device):
    """Play a checkpoint greedily without recording. Higher is better: any finish beats any
    non-finish, faster finishes beat slower ones, and otherwise further is better."""
    model, train_args = load_model(path, device)
    env = make_env(**play_env_kwargs(train_args))
    obs, _ = env.reset()
    info, done = {}, False
    while not done:
        action = model.act(torch.from_numpy(obs).unsqueeze(0).to(device), greedy=True)[0].item()
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    env.close()
    if info.get("flag_get"):
        return (1, -info["frames"])
    return (0, info.get("x_pos", 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", default=None,
                   help="one checkpoint, or several: they're all played quietly and the closest one is shown")
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    p.add_argument("--scale", type=int, default=3, help="window/video size multiplier")
    p.add_argument("--mp4", default=None, help="also save the run to this video file")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--device", default="cpu", help="cpu matches speedrun_eval.py exactly")
    p.add_argument("--label", default="", help="text shown at the top of every frame")
    p.add_argument("--best-file", default=None,
                   help="personal-best mode: only make a video if the run beats the score stored in this "
                        "JSON file; the new best score and video are then saved next to it")
    args = p.parse_args()

    device = pick_device(args.device)
    candidates = args.ckpt or [next(
        (f for f in ("models/pipe/fastest.pt", "models/speedrun/fastest.pt", "models/ppo/solved.pt", "models/ppo/best.pt")
         if os.path.exists(f)), None)]
    if candidates[0] is None:
        raise SystemExit("No checkpoint found. Pass --ckpt.")
    if len(candidates) == 1 and not args.best_file:
        ckpt_path = candidates[0]
    else:
        scores = {c: closeness(c, device) for c in candidates}
        ckpt_path = max(candidates, key=scores.get)
        if args.best_file:
            previous = None
            if os.path.exists(args.best_file):
                with open(args.best_file) as f:
                    previous = tuple(json.load(f)["score"])
            if previous is not None and scores[ckpt_path] <= previous:
                print(f"no new personal best ({scores[ckpt_path]} vs {previous}); no video")
                return
    model, train_args = load_model(ckpt_path, device)

    env = make_env(**play_env_kwargs(train_args))

    # Grab every NES frame and the buttons held on it: hook the layer the wrapper steps once per
    # frame (it receives the per-frame action, after auto-release has let go of A if needed)
    frames, held = [], []
    inner_step = env.env.step
    buttons = ACTION_SETS[train_args["actions"]]

    def recording_step(action):
        result = inner_step(action)
        frames.append(env.nes.screen.copy())
        held.append(set(buttons[action]) - {"NOOP"})
        return result

    env.env.step = recording_step

    obs, _ = env.reset()
    frames.append(env.nes.screen.copy())
    held.append(set())
    info, done = {}, False
    with torch.no_grad():
        while not done:
            action = model.act(torch.from_numpy(obs).unsqueeze(0).to(device), greedy=True)[0].item()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
    env.close()

    cleared = info.get("flag_get", False)
    total = info.get("frames", len(frames) - 1)
    route = " via the PIPE" if info.get("pipe") else ""
    result = (f"cleared in {total} frames ({total / NES_FPS:.2f} s){route}" if cleared
              else f"did NOT clear, reached x={info.get('x_pos')}{route}")
    print(f"{ckpt_path}: {result}")
    picked = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("update")
    if len(candidates) > 1 and picked is not None:
        result += f" (update {picked})"
    top_label = f"{args.label}  |  {result}" if args.label else ""

    h, w, _ = frames[0].shape
    size = (w * args.scale, h * args.scale)
    pad_h = 40 * args.scale
    out_size = (size[0], size[1] + pad_h)  # game on top, controller underneath
    writer = None
    if args.mp4:
        os.makedirs(os.path.dirname(args.mp4) or ".", exist_ok=True)
        tmp_mp4 = args.mp4[:-4] + ".partial.mp4"  # renamed when done, so watchers never see half a file
        writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"), NES_FPS, out_size)

    frame_time = 1.0 / (NES_FPS * args.speed)
    start = time.perf_counter()
    for i, f in enumerate(frames):
        img = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), size, interpolation=cv2.INTER_NEAREST)
        label = f"frame {i}  {i / NES_FPS:5.2f}s"
        cv2.putText(img, label, (10, size[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * args.scale / 2,
                    (255, 255, 255), max(1, args.scale // 2))
        if top_label:
            cv2.rectangle(img, (0, 0), (size[0], 14 * args.scale), (0, 0, 0), -1)
            cv2.putText(img, top_label, (8, 10 * args.scale), cv2.FONT_HERSHEY_SIMPLEX, 0.15 * args.scale,
                        (255, 255, 255), 1)
        fresh = held[i] - (held[i - 1] if i else set())
        img = np.vstack([img, draw_gamepad(size[0], args.scale, held[i], fresh)])
        if writer is not None:
            writer.write(img)
        if not args.no_window:
            cv2.imshow("Mario run (q to quit)", img)
            # wait until this frame's real-time slot so playback runs at true speed
            delay = start + (i + 1) * frame_time - time.perf_counter()
            if cv2.waitKey(max(1, int(delay * 1000))) & 0xFF == ord("q"):
                break
    if writer is not None:
        writer.release()
        os.replace(tmp_mp4, args.mp4)
        if args.best_file:
            # remember the new personal best (score + a copy of its video) next to the checkpoints
            shutil.copy(args.mp4, os.path.splitext(args.best_file)[0] + ".mp4")
            with open(args.best_file, "w") as f:
                json.dump({"score": list(scores[ckpt_path]), "label": top_label}, f)
        print(f"saved {args.mp4}")
    if not args.no_window:
        cv2.waitKey(1500)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
