"""Watch a trained Mario's run at real NES speed (60 fps), and optionally save it as a video.

    python watch_run.py                              # models/speedrun/fastest.pt, else ppo/solved.pt
    python watch_run.py --ckpt models/ppo/solved.pt
    python watch_run.py --mp4 runs/speedrun/best_run.mp4          # save a video too
    python watch_run.py --mp4 runs/speedrun/best_run.mp4 --no-window   # just the video
    python watch_run.py --speed 0.5                  # slow motion

Every NES frame is shown (not just one per agent step), with a frame counter.
Press q in the window to quit.
"""
import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

from ppo_model import ActorCritic, pick_device
from wrappers import make_env

NES_FPS = 60.0988


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    p.add_argument("--scale", type=int, default=3, help="window/video size multiplier")
    p.add_argument("--mp4", default=None, help="also save the run to this video file")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--device", default="cpu", help="cpu matches speedrun_eval.py exactly")
    args = p.parse_args()

    ckpt_path = args.ckpt or next(
        (f for f in ("models/speedrun/fastest.pt", "models/ppo/solved.pt", "models/ppo/best.pt")
         if os.path.exists(f)), None)
    if ckpt_path is None:
        raise SystemExit("No checkpoint found. Pass --ckpt.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own file
    train_args = ckpt["args"]
    device = pick_device(args.device)
    model = ActorCritic(ckpt["obs_shape"][0], ckpt["n_actions"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env = make_env(level=train_args["level"], actions=train_args["actions"], skip=train_args.get("skip", 4))

    # Grab every NES frame: hook the layer the wrapper steps once per frame
    frames = []
    inner_step = env.env.step

    def recording_step(action):
        result = inner_step(action)
        frames.append(env.nes.screen.copy())
        return result

    env.env.step = recording_step

    obs, _ = env.reset()
    frames.append(env.nes.screen.copy())
    info, done = {}, False
    with torch.no_grad():
        while not done:
            action = model.act(torch.from_numpy(obs).unsqueeze(0).to(device), greedy=True)[0].item()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
    env.close()

    cleared = info.get("flag_get", False)
    total = info.get("frames", len(frames) - 1)
    print(f"{ckpt_path}: {'cleared' if cleared else 'did NOT clear'} in {total} frames "
          f"({total / NES_FPS:.2f} s)" + ("" if cleared else f", reached x={info.get('x_pos')}"))

    h, w, _ = frames[0].shape
    size = (w * args.scale, h * args.scale)
    writer = None
    if args.mp4:
        os.makedirs(os.path.dirname(args.mp4) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.mp4, cv2.VideoWriter_fourcc(*"mp4v"), NES_FPS, size)

    frame_time = 1.0 / (NES_FPS * args.speed)
    start = time.perf_counter()
    for i, f in enumerate(frames):
        img = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), size, interpolation=cv2.INTER_NEAREST)
        label = f"frame {i}  {i / NES_FPS:5.2f}s"
        cv2.putText(img, label, (10, size[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * args.scale / 2,
                    (255, 255, 255), max(1, args.scale // 2))
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
        print(f"saved {args.mp4}")
    if not args.no_window:
        cv2.waitKey(1500)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
