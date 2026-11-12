"""Train a PPO agent on Super Mario Bros with many Marios playing in parallel.

    python train_ppo.py                 # train fast, no window
    python train_ppo.py --watch         # also show every Mario tiled in one window
    python train_ppo.py --resume        # continue from models/ppo/latest.pt
    tensorboard --logdir runs           # graphs

Speed tricks:
  - The envs are split into two groups. While one group's games are running,
    the network picks actions for the other group, so nothing waits around.
  - Some training episodes start from save states further into the level
    (--savestate-prob), so Mario practises the hard parts more often.
    Needs the patched emulator: bash tools/build_nes_py.sh
  - Evaluation runs all its games in parallel too.

Training stops on its own once the greedy policy clears the level on every
evaluation run (with random start delays) several evaluations in a row.
"""
import argparse
import os
import time
import warnings
from collections import deque

warnings.filterwarnings("ignore")  # old gym prints lots of harmless deprecation warnings
os.environ.setdefault("PYTHONWARNINGS", "ignore")  # ...and the same for the worker processes

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ppo_model import ActorCritic, pick_device
from vec_env import SubprocVecEnv, tile_frames
from wrappers import HAS_SNAPSHOTS, make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="1-1")
    p.add_argument("--actions", default="right_only", choices=["right_only", "simple"])
    p.add_argument("--num-envs", type=int, default=16, help="must be even (two groups)")
    p.add_argument("--num-steps", type=int, default=128, help="steps per env per rollout")
    p.add_argument("--total-steps", type=int, default=10_000_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=0.1)
    p.add_argument("--savestate-prob", type=float, default=0.3,
                   help="chance a training episode starts from a mid-level save (0 = off)")
    p.add_argument("--eval-every", type=int, default=10, help="updates between greedy evaluations")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--solve-streak", type=int, default=3, help="perfect evals in a row to stop")
    p.add_argument("--watch", action="store_true", help="show all Marios tiled in a window")
    p.add_argument("--render-every", type=int, default=2, help="steps between window refreshes")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-dir", default="models/ppo")
    p.add_argument("--device", default="auto")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, device, eval_envs, rng, max_steps=3000):
    """Play greedily (always the best-scoring action, no randomness), all runs at once.

    The emulator is deterministic, so each run first waits a random 0-30
    steps; a policy that only memorised one exact path fails this.
    """
    n = eval_envs.num_envs
    obs = eval_envs.reset()
    noops = rng.integers(0, 31, size=n)
    finished = np.zeros(n, dtype=bool)
    wins = np.zeros(n, dtype=bool)
    xs = np.zeros(n)
    for t in range(max_steps):
        action, _, _ = model.act(torch.from_numpy(obs).to(device), greedy=True)
        action = action.cpu().numpy()
        action[t < noops] = 0
        obs, _, done, infos = eval_envs.step(action)
        for i in np.flatnonzero(done & ~finished):
            finished[i] = True
            wins[i] = infos[i]["flag_get"]
            xs[i] = infos[i]["x_pos"]
        if finished.all():
            break
    return wins.mean(), xs.mean()


def main():
    args = parse_args()
    assert args.num_envs % 2 == 0, "--num-envs must be even"
    device = pick_device(args.device)
    torch.set_num_threads(2)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.savestate_prob > 0 and not HAS_SNAPSHOTS:
        print("Save-state starts need the patched emulator (bash tools/build_nes_py.sh); running without them.")
    print(f"device={device}  envs={args.num_envs}  level={args.level}  savestate_prob={args.savestate_prob}")

    env_kwargs = dict(level=args.level, actions=args.actions)
    N, T = args.num_envs, args.num_steps
    half = N // 2
    groups = [SubprocVecEnv(half, dict(env_kwargs, savestate_prob=args.savestate_prob)) for _ in range(2)]
    slices = [slice(0, half), slice(half, N)]
    eval_envs = SubprocVecEnv(args.eval_episodes, env_kwargs)  # always from the real start

    probe = make_env(**env_kwargs)
    obs_shape, n_actions = probe.observation_space.shape, probe.action_space.n
    probe.close()

    model = ActorCritic(obs_shape[0], n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    global_step, update, streak, best_eval = 0, 0, 0, -1.0
    latest = os.path.join(args.save_dir, "latest.pt")
    if args.resume and os.path.exists(latest):
        ckpt = torch.load(latest, map_location=device, weights_only=False)  # our own file
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step, update = ckpt["global_step"], ckpt["update"]
        streak, best_eval = ckpt.get("streak", 0), ckpt.get("best_eval", -1.0)
        print(f"resumed from step {global_step}")

    writer = SummaryWriter(f"runs/ppo_{args.level}_{time.strftime('%Y%m%d-%H%M%S')}")
    rng = np.random.default_rng()

    batch = N * T
    mb_size = batch // args.minibatches
    num_updates = args.total_steps // batch

    # Rollout storage (observations stay uint8 to save memory)
    obs_buf = torch.zeros((T, N, *obs_shape), dtype=torch.uint8, device=device)
    act_buf = torch.zeros((T, N), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N), device=device)
    rew_buf = torch.zeros((T, N), device=device)
    done_buf = torch.zeros((T, N), device=device)
    val_buf = torch.zeros((T, N), device=device)

    next_obs = [torch.from_numpy(g.reset()).to(device) for g in groups]
    next_done = [torch.zeros(half, device=device) for _ in groups]
    ep_return = np.zeros(N)
    recent_returns, recent_x, recent_flags = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    watching = args.watch
    frames_cache, infos_cache = [None, None], [[{"x_pos": 0}] * half, [{"x_pos": 0}] * half]
    t_start, steps_at_start = time.time(), global_step

    def collect(g, t):
        """Receive group g's results for step t and record them."""
        nonlocal global_step, watching
        obs, reward, done, infos = groups[g].step_wait()
        sl = slices[g]
        global_step += half
        rew_buf[t, sl] = torch.from_numpy(reward * args.reward_scale).to(device)
        next_obs[g] = torch.from_numpy(obs).to(device)
        next_done[g] = torch.from_numpy(done.astype(np.float32)).to(device)
        ep_return[sl] += reward
        for i in np.flatnonzero(done):
            j = sl.start + i
            if not infos[i]["from_save"]:  # only full runs count toward these stats
                recent_returns.append(ep_return[j])
                recent_x.append(infos[i]["x_pos"])
                recent_flags.append(float(infos[i]["flag_get"]))
                writer.add_scalar("episode/return", ep_return[j], global_step)
                writer.add_scalar("episode/x_pos", infos[i]["x_pos"], global_step)
            ep_return[j] = 0
        infos_cache[g] = infos
        if watching and t % args.render_every == 0:
            frames_cache[g] = groups[g].render_frames()  # safe: this group is idle right now
            if g == 1 and frames_cache[0] is not None:
                cv2.imshow("Marios (q to hide)", tile_frames(frames_cache[0] + frames_cache[1],
                                                             infos_cache[0] + infos_cache[1]))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    watching = False
                    cv2.destroyAllWindows()

    def send(g, t):
        """Pick actions for group g at step t and start its games running."""
        sl = slices[g]
        obs_buf[t, sl] = next_obs[g]
        done_buf[t, sl] = next_done[g]
        with torch.no_grad():
            action, logp, value = model.act(next_obs[g])
        act_buf[t, sl], logp_buf[t, sl], val_buf[t, sl] = action, logp, value
        groups[g].step_async(action.cpu().numpy())

    while update < num_updates:
        update += 1
        optimizer.param_groups[0]["lr"] = args.lr * (1.0 - (update - 1) / num_updates)

        # ---- 1. Play: the two groups take turns, so the network and the games overlap ----
        model.eval()
        send(0, 0)
        send(1, 0)
        for t in range(1, T):
            collect(0, t - 1); send(0, t)   # group 1's games keep running meanwhile
            collect(1, t - 1); send(1, t)   # group 0's games keep running meanwhile
        collect(0, T - 1)
        collect(1, T - 1)
        rollout_obs, rollout_done = torch.cat(next_obs), torch.cat(next_done)

        # ---- 2. Score: work out how much better each action was than expected (GAE) ----
        with torch.no_grad():
            _, next_value = model(rollout_obs)
            adv = torch.zeros_like(rew_buf)
            last_gae = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                if t == T - 1:
                    not_done, nv = 1.0 - rollout_done, next_value
                else:
                    not_done, nv = 1.0 - done_buf[t + 1], val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nv * not_done - val_buf[t]
                last_gae = delta + args.gamma * args.gae_lambda * not_done * last_gae
                adv[t] = last_gae
            returns = adv + val_buf

        # ---- 3. Learn: several passes over the batch with the clipped PPO loss ----
        model.train()
        b_obs = obs_buf.reshape(batch, *obs_shape)
        b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), returns.reshape(-1)
        for _ in range(args.epochs):
            perm = torch.randperm(batch, device=device)
            for start in range(0, batch, mb_size):
                idx = perm[start:start + mb_size]
                logits, value = model(b_obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_act[idx])
                ratio = (new_logp - b_logp[idx]).exp()

                mb_adv = b_adv[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = -torch.min(
                    ratio * mb_adv, ratio.clamp(1 - args.clip, 1 + args.clip) * mb_adv
                ).mean()
                v_loss = 0.5 * (value - b_ret[idx]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        # ---- Logging ----
        sps = (global_step - steps_at_start) / (time.time() - t_start)
        writer.add_scalar("loss/policy", pg_loss.item(), global_step)
        writer.add_scalar("loss/value", v_loss.item(), global_step)
        writer.add_scalar("loss/entropy", entropy.item(), global_step)
        writer.add_scalar("perf/steps_per_sec", sps, global_step)
        if recent_returns:
            writer.add_scalar("train/flag_rate_100", np.mean(recent_flags), global_step)
            print(
                f"update {update:5d} | step {global_step:9d} | {sps:5.0f} sps | "
                f"return {np.mean(recent_returns):7.1f} | x {np.mean(recent_x):6.0f} | "
                f"flag {100 * np.mean(recent_flags):5.1f}%"
            )

        # ---- Greedy evaluation: is it actually beating the level? ----
        if update % args.eval_every == 0:
            model.eval()
            win_rate, mean_x = map(float, evaluate(model, device, eval_envs, rng))
            writer.add_scalar("eval/win_rate", win_rate, global_step)
            writer.add_scalar("eval/x_pos", mean_x, global_step)
            print(f"  eval: cleared {100 * win_rate:.0f}% of {args.eval_episodes} runs, mean x {mean_x:.0f}")
            score = float(win_rate * 10_000 + mean_x)
            if score > best_eval:
                best_eval = score
                torch.save({"model": model.state_dict(), "n_actions": n_actions,
                            "obs_shape": obs_shape, "args": vars(args)},
                           os.path.join(args.save_dir, "best.pt"))
            streak = streak + 1 if win_rate == 1.0 else 0

        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "global_step": global_step, "update": update, "streak": streak,
             "best_eval": best_eval, "args": vars(args),
             "n_actions": n_actions, "obs_shape": obs_shape},
            latest,
        )

        if streak >= args.solve_streak:
            torch.save({"model": model.state_dict(), "n_actions": n_actions,
                        "obs_shape": obs_shape, "args": vars(args)},
                       os.path.join(args.save_dir, "solved.pt"))
            print(f"Solved! {streak} perfect evaluations in a row. Saved {args.save_dir}/solved.pt")
            break

    for g in groups:
        g.close()
    eval_envs.close()
    writer.close()


if __name__ == "__main__":
    main()
