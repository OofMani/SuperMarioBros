"""Watch a trained PPO Mario play.

    python play_ppo.py                          # uses models/ppo/solved.pt, else best.pt
    python play_ppo.py --ckpt models/ppo/latest.pt --episodes 3
"""
import argparse
import os

import torch

from ppo_model import ActorCritic, pick_device
from wrappers import make_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--sample", action="store_true", help="sample actions instead of always picking the best")
    args = p.parse_args()

    ckpt_path = args.ckpt or next(
        (f for f in ("models/ppo/solved.pt", "models/ppo/best.pt", "models/ppo/latest.pt") if os.path.exists(f)),
        None,
    )
    if ckpt_path is None:
        raise SystemExit("No checkpoint found. Train first with: python train_ppo.py")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # our own file
    train_args = ckpt["args"]

    device = pick_device()
    model = ActorCritic(ckpt["obs_shape"][0], ckpt["n_actions"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    env = make_env(level=train_args["level"], actions=train_args["actions"], render_mode="human",
                   skip=train_args.get("skip", 4))
    print(f"Playing {ckpt_path}")
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, info = False, {}
        while not done:
            with torch.no_grad():
                action, _, _ = model.act(torch.from_numpy(obs).unsqueeze(0).to(device), greedy=not args.sample)
            obs, _, term, trunc, info = env.step(action.item())
            done = term or trunc
        print(f"episode {ep + 1}: x={info.get('x_pos')}  flag={'YES' if info.get('flag_get') else 'no'}")
    env.close()


if __name__ == "__main__":
    main()
