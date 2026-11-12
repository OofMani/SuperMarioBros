"""Train the (fixed-up) DQN agent on one Mario level.

    python main.py                 # train, no window (much faster)
    python main.py --display       # watch while it trains
    python main.py --resume        # continue from models/dqn/model.pt
    tensorboard --logdir runs      # graphs

For the faster, parallel trainer see train_ppo.py.
"""
import argparse
import os
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from wrappers import make_env

p = argparse.ArgumentParser()
p.add_argument("--level", default="1-1")
p.add_argument("--episodes", type=int, default=20_000)
p.add_argument("--display", action="store_true")
p.add_argument("--resume", action="store_true")
p.add_argument("--save-every", type=int, default=100)
p.add_argument("--eval-every", type=int, default=50)
p.add_argument("--save-dir", default="models/dqn")
args = p.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
model_path = os.path.join(args.save_dir, "model.pt")

env = make_env(level=args.level, render_mode="human" if args.display else "rgb_array")
eval_env = make_env(level=args.level)
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
print("device:", agent.device)
if args.resume and os.path.exists(model_path):
    agent.load_model(model_path)
    agent.epsilon = agent.eps_start = 0.1  # it already knows something; explore less
    print("resumed from", model_path)

writer = SummaryWriter(f"runs/dqn_{args.level}_{time.strftime('%Y%m%d-%H%M%S')}")


def evaluate():
    """One run with no randomness. The emulator is deterministic, so if this
    reaches the flag, the same policy will reach it every time."""
    state, _ = eval_env.reset()
    done, info = False, {}
    while not done:
        state, _, term, trunc, info = eval_env.step(agent.choose_action(state, greedy=True))
        done = term or trunc
    return info.get("flag_get", False), info.get("x_pos", 0)


for ep in range(1, args.episodes + 1):
    state, _ = env.reset()
    done, total_reward, info = False, 0.0, {}
    while not done:
        a = agent.choose_action(state)
        new_state, reward, term, trunc, info = env.step(a)
        done = term or trunc
        # Only a real ending (death/flag) cuts off the future; a stuck-timeout doesn't
        agent.store_in_memory(state, a, reward, new_state, term)
        agent.learn()
        state = new_state
        total_reward += reward

    writer.add_scalar("episode/return", total_reward, agent.step_counter)
    writer.add_scalar("episode/x_pos", info.get("x_pos", 0), agent.step_counter)
    writer.add_scalar("episode/epsilon", agent.epsilon, agent.step_counter)
    print(f"ep {ep:6d} | return {total_reward:7.1f} | x {info.get('x_pos', 0):5d} | "
          f"eps {agent.epsilon:.3f} | steps {agent.step_counter}")

    if ep % args.save_every == 0:
        agent.save_model(model_path)

    if ep % args.eval_every == 0:
        flag, x = evaluate()
        writer.add_scalar("eval/x_pos", x, agent.step_counter)
        print(f"  greedy eval: x={x} flag={'YES' if flag else 'no'}")
        if flag:
            agent.save_model(os.path.join(args.save_dir, "solved.pt"))
            print("  Cleared the level with no randomness! Saved solved.pt")
            break

env.close()
eval_env.close()
writer.close()
