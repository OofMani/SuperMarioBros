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

Speedrun mode (fine-tune a solved Mario to finish faster):

    python train_ppo.py --speedrun --init-from models/ppo/solved.pt

  - adds a small cost per NES frame and a flag bonus that grows the faster he
    finishes (--time-cost, --speed-bonus, --ref-frames)
  - after changing reward settings on a trained model, add --critic-warmup 30
    so the critic relearns what situations are worth before the actor moves
  - uses gamma 0.997 so the finish-time bonus reaches back to early decisions,
    and the gentle fine-tuning settings lr 5e-5 / ent-coef 0.005
  - one evaluation run always starts on frame 0 (a real speedrun start); every
    new record is saved as fastest.pt. It never stops on its own.

Pipe mode (let Mario find the 1-1 pipe shortcut):

    python train_ppo.py --pipe-mode --init-from models/ppo/solved.pt
    python train_ppo.py --pipe-mode                                   # from scratch (recommended)

  - all buttons (incl. Down); a 5-button checkpoint gets its action layer expanded
  - game-accurate timing (no skipped pipe animations)
  - route-neutral reward: finishing (big bonus + speed bonus) and dying count, plus 20% of
    the old "pixels moved right" reward as a steady nudge, a small cost per frame (0.05) and
    a 50-step stuck timer. Without those, he learned to stand still to put off the death
    penalty. The pipe skips ~1,700 px (-340) but saves ~140 frames (+700 speed bonus),
    so it still pays clearly.
  - reverse curriculum: half the episodes start from save states along a demo
    (demos/overland.json by default), beginning near the flag and moving back
    as Mario gets reliable; the other half start at the real level start
  - evaluation prints whether each record run used the pipe
"""
import argparse
import os
import subprocess
import sys
import time
import warnings
from collections import deque

warnings.filterwarnings("ignore")  # old gym prints lots of harmless deprecation warnings
os.environ.setdefault("PYTHONWARNINGS", "ignore")  # ...and the same for the worker processes

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ppo_model import ActorCritic, expand_actions, pick_device
from vec_env import SubprocVecEnv, tile_frames
from wrappers import ACTION_SETS, HAS_SNAPSHOTS, make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--level", default="1-1")
    p.add_argument("--actions", default="right_only", choices=["right_only", "simple", "complex"])
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
    p.add_argument("--preview-every", type=int, default=5,
                   help="every N updates, replay the networks from those last N updates (max 10) and save "
                        "a video of the one that got closest to the flag (0 = off)")
    p.add_argument("--no-preview-window", action="store_true",
                   help="still save preview videos, but don't open the window that plays them")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-dir", default="models/ppo")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip", type=int, default=4, help="NES frames per agent step")
    # speedrun mode
    p.add_argument("--speedrun", action="store_true", help="optimise finish time (see top of file)")
    p.add_argument("--init-from", default=None, help="start from these weights (fresh optimizer and step count)")
    p.add_argument("--time-cost", type=float, default=None, help="reward lost per NES frame (speedrun: 0.25)")
    p.add_argument("--speed-bonus", type=float, default=None,
                   help="flag bonus per frame finished under --ref-frames (speedrun: 1.0)")
    p.add_argument("--ref-frames", type=int, default=1400)
    p.add_argument("--stuck-steps", type=int, default=150,
                   help="end an episode after this many steps without new progress (pipe mode: 50)")
    p.add_argument("--critic-warmup", type=int, default=0,
                   help="first N updates of this run train only the critic's output layer (actor frozen). "
                        "Use after changing the reward on a trained model.")
    p.add_argument("--warmup-lr", type=float, default=1e-3)
    # pipe mode
    p.add_argument("--pipe-mode", action="store_true", help="find the pipe shortcut (see top of file)")
    p.add_argument("--accurate-timing", action="store_true", help="no skipped pipe/area animations")
    p.add_argument("--reward", default="x", choices=["x", "finish"], help="x = env reward, finish = route-neutral")
    p.add_argument("--finish-bonus", type=float, default=1000.0)
    p.add_argument("--death-penalty", type=float, default=500.0)
    p.add_argument("--progress-weight", type=float, default=0.0, help="finish reward: weight of the env's x reward")
    p.add_argument("--curriculum", default=None, help="demo JSON for reverse-curriculum starts")
    p.add_argument("--curriculum-prob", type=float, default=0.5)
    p.add_argument("--curriculum-threshold", type=float, default=0.7,
                   help="move the curriculum start back once this share of curriculum episodes finish")
    p.add_argument("--curriculum-step", type=float, default=0.05)
    p.add_argument("--curriculum-window", type=int, default=40, help="curriculum episodes per check")
    p.add_argument("--new-action-bias", type=float, default=-3.0,
                   help="starting bias for buttons the --init-from checkpoint didn't have")
    args = p.parse_args()
    given = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
    if args.pipe_mode:
        pipe_defaults = {"--actions": ("actions", "complex"), "--accurate-timing": ("accurate_timing", True),
                         "--reward": ("reward", "finish"), "--speed-bonus": ("speed_bonus", 5.0),
                         # a small cost per frame and a short stuck timer, so standing still to put
                         # off the death penalty doesn't pay (he learned to procrastinate without them)
                         "--time-cost": ("time_cost", 0.05), "--stuck-steps": ("stuck_steps", 50),
                         # holding right+A (endless bunny-hop, thanks to auto-release) clears 1-1 in
                         # ~1,950 frames and a fresh network finds it within ~10 updates, so the speed
                         # bonus has to pay for every frame below that, not only below 1,400
                         "--ref-frames": ("ref_frames", 3000),
                         # dying is a small nudge, not the main signal: at -500 it swamped the progress
                         # reward near the start, so running into the first Goomba looked no better
                         # than standing still until the stuck timer
                         "--death-penalty": ("death_penalty", 50.0),
                         "--reward-scale": ("reward_scale", 0.01),
                         "--curriculum": ("curriculum", "demos/overland.json"),
                         "--savestate-prob": ("savestate_prob", 0.0), "--gamma": ("gamma", 0.997),
                         # fine-tuning a checkpoint needs gentle steps; from scratch uses the
                         # settings that originally solved 1-1 (lower ent-coef later if the
                         # greedy test gets stuck deciding, as happened at the first pipe)
                         "--lr": ("lr", 5e-5 if args.init_from else 2.5e-4),
                         "--ent-coef": ("ent_coef", 0.005 if args.init_from else 0.01),
                         "--progress-weight": ("progress_weight", 0.2),
                         "--new-action-bias": ("new_action_bias", -2.0),
                         "--save-dir": ("save_dir", "models/pipe")}
        if args.init_from and "--critic-warmup" not in given:
            args.critic_warmup = 30
        for flag, (name, value) in pipe_defaults.items():
            if flag not in given:
                setattr(args, name, value)
        args.speedrun = True  # frame-0 evaluation, fastest.pt records, no auto-stop
    elif args.speedrun:
        # defaults for speedrun mode, used only for flags not given on the command line
        speedrun_defaults = {"--time-cost": ("time_cost", 0.25), "--speed-bonus": ("speed_bonus", 1.0),
                             "--gamma": ("gamma", 0.997), "--lr": ("lr", 5e-5),
                             "--ent-coef": ("ent_coef", 0.005), "--save-dir": ("save_dir", "models/speedrun")}
        for flag, (name, value) in speedrun_defaults.items():
            if flag not in given:
                setattr(args, name, value)
    args.time_cost = args.time_cost or 0.0
    args.speed_bonus = args.speed_bonus or 0.0
    return args


@torch.no_grad()
def evaluate(model, device, eval_envs, rng, max_steps=3000, frame0=False):
    """Play greedily (always the best-scoring action, no randomness), all runs at once.

    The emulator is deterministic, so each run first waits a random 0-30
    steps; a policy that only memorised one exact path fails this.
    With frame0=True, run 0 starts with no wait (the speedrun start).
    Returns (win rate, mean x, frames of run 0 or None if it didn't finish, run 0 used the pipe).
    """
    n = eval_envs.num_envs
    obs = eval_envs.reset()
    noops = rng.integers(0, 31, size=n)
    if frame0:
        noops[0] = 0
    finished = np.zeros(n, dtype=bool)
    wins = np.zeros(n, dtype=bool)
    xs = np.zeros(n)
    frames0, pipe0 = None, False
    for t in range(max_steps):
        action, _, _ = model.act(torch.from_numpy(obs).to(device), greedy=True)
        action = action.cpu().numpy()
        action[t < noops] = 0
        obs, _, done, infos = eval_envs.step(action)
        for i in np.flatnonzero(done & ~finished):
            finished[i] = True
            wins[i] = infos[i]["flag_get"]
            xs[i] = infos[i]["x_pos"]
            if i == 0 and wins[i]:
                frames0, pipe0 = infos[i]["frames"], infos[i]["pipe"]
        if finished.all():
            break
    return float(wins.mean()), float(xs.mean()), frames0, pipe0


def main():
    args = parse_args()
    assert args.num_envs % 2 == 0, "--num-envs must be even"
    device = pick_device(args.device)
    torch.set_num_threads(2)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.savestate_prob > 0 and not HAS_SNAPSHOTS:
        print("Save-state starts need the patched emulator (bash tools/build_nes_py.sh); running without them.")
    print(f"device={device}  envs={args.num_envs}  level={args.level}  savestate_prob={args.savestate_prob}")
    if args.speedrun:
        print(f"speedrun mode: time_cost={args.time_cost}/frame  speed_bonus={args.speed_bonus}/frame "
              f"under {args.ref_frames}  gamma={args.gamma}  lr={args.lr}  ent_coef={args.ent_coef}  "
              f"skip={args.skip}  save_dir={args.save_dir}")
    if args.pipe_mode or args.reward == "finish" or args.curriculum:
        print(f"pipe settings: actions={args.actions}  accurate_timing={args.accurate_timing}  reward={args.reward} "
              f"(finish {args.finish_bonus}, death -{args.death_penalty}, progress x{args.progress_weight})  "
              f"reward_scale={args.reward_scale}  curriculum={args.curriculum} (p={args.curriculum_prob})  "
              f"critic_warmup={args.critic_warmup}")

    env_kwargs = dict(level=args.level, actions=args.actions, skip=args.skip,
                      time_cost=args.time_cost, speed_bonus=args.speed_bonus, ref_frames=args.ref_frames,
                      accurate_timing=args.accurate_timing, reward_mode=args.reward,
                      finish_bonus=args.finish_bonus, death_penalty=args.death_penalty,
                      progress_weight=args.progress_weight, stuck_steps=args.stuck_steps)
    curriculum_start = 1.0  # fraction of the demo where episodes may start (1 = only the end)
    N, T = args.num_envs, args.num_steps
    half = N // 2
    train_kwargs = dict(env_kwargs, savestate_prob=args.savestate_prob)
    if args.curriculum:
        train_kwargs.update(curriculum=args.curriculum, curriculum_prob=args.curriculum_prob,
                            curriculum_start=curriculum_start)
    groups = [SubprocVecEnv(half, train_kwargs) for _ in range(2)]
    slices = [slice(0, half), slice(half, N)]
    eval_envs = SubprocVecEnv(args.eval_episodes, env_kwargs)  # always from the real start

    probe = make_env(**env_kwargs)
    obs_shape, n_actions = probe.observation_space.shape, probe.action_space.n
    probe.close()

    model = ActorCritic(obs_shape[0], n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    # Critic warmup: only the critic's last layer learns, so Mario's play (actor) and the
    # shared vision layers stay exactly as they were while the value estimates catch up
    critic_opt = torch.optim.Adam(model.critic.parameters(), lr=args.warmup_lr, eps=1e-5)
    warmup_left = args.critic_warmup

    global_step, update, streak, best_eval = 0, 0, 0, -1.0
    best_frames = None  # fastest frame-0 finish so far (speedrun mode)
    latest = os.path.join(args.save_dir, "latest.pt")
    if args.init_from and not (args.resume and os.path.exists(latest)):
        init = torch.load(args.init_from, map_location=device, weights_only=False)  # our own file
        weights = init["model"]
        if init["n_actions"] != n_actions:
            old_buttons = ACTION_SETS[init["args"]["actions"]]
            weights = expand_actions(weights, old_buttons, ACTION_SETS[args.actions], args.new_action_bias)
            print(f"expanded the action layer from {init['n_actions']} to {n_actions} buttons "
                  f"(new ones start at bias {args.new_action_bias})")
        model.load_state_dict(weights)
        print(f"starting from the weights in {args.init_from}")
    if args.resume and os.path.exists(latest):
        ckpt = torch.load(latest, map_location=device, weights_only=False)  # our own file
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step, update = ckpt["global_step"], ckpt["update"]
        streak, best_eval = ckpt.get("streak", 0), ckpt.get("best_eval", -1.0)
        best_frames = ckpt.get("best_frames")
        curriculum_start = ckpt.get("curriculum_start", curriculum_start)
        if args.curriculum:
            for g in groups:
                g.set_attr("curriculum_start", curriculum_start)
        print(f"resumed from step {global_step}")

    run_kind = "pipe" if args.pipe_mode else ("speedrun" if args.speedrun else "ppo")
    run_dir = f"runs/{run_kind}_{args.level}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(run_dir)

    # Previews: every --preview-every updates, the networks from those last updates are replayed
    # in a background process (training doesn't wait), and a video of the one that got closest
    # to the flag is saved in <run_dir>/previews. A separate window loops the newest one.
    here = os.path.dirname(os.path.abspath(__file__))
    preview_dir = os.path.join(run_dir, "previews")
    preview_render = None
    recent_nets = deque(maxlen=max(1, min(args.preview_every, 10)))  # (update, CPU copy of the weights)
    if args.preview_every:
        os.makedirs(preview_dir, exist_ok=True)
        print(f"preview every {args.preview_every} updates: the closest run of the last "
              f"{recent_nets.maxlen} updates -> {preview_dir}")
        if not args.no_preview_window:
            subprocess.Popen([sys.executable, os.path.join(here, "preview_window.py"), preview_dir])
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
    recent_finishes = deque(maxlen=100)  # per full run: (frames, used the pipe) if it finished, else None
    curriculum_results = []            # finished? for episodes that started on the curriculum
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
                recent_finishes.append((infos[i]["frames"], float(infos[i]["pipe"]))
                                       if infos[i]["flag_get"] else None)
                writer.add_scalar("episode/return", ep_return[j], global_step)
                writer.add_scalar("episode/x_pos", infos[i]["x_pos"], global_step)
            if infos[i]["from_curriculum"]:
                curriculum_results.append(float(infos[i]["flag_get"]))
                if infos[i]["pipe"]:
                    writer.add_scalar("curriculum/pipe_start_x", infos[i]["start_x"], global_step)
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
        warming_up = warmup_left > 0
        if warming_up:
            warmup_left -= 1
            if warmup_left == 0:
                print("  critic warmup finished: the actor learns from the next update")
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
                if warming_up:
                    critic_opt.zero_grad()
                    v_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.critic.parameters(), args.max_grad_norm)
                    critic_opt.step()
                    continue

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
        finishes = [r for r in recent_finishes if r is not None]
        recent_frames = [r[0] for r in finishes]
        recent_pipe = [r[1] for r in finishes]
        if recent_returns:
            writer.add_scalar("train/flag_rate_100", np.mean(recent_flags), global_step)
            print(
                f"update {update:5d} | step {global_step:9d} | {sps:5.0f} sps | "
                f"return {np.mean(recent_returns):7.1f} | x {np.mean(recent_x):6.0f} | "
                f"flag {100 * np.mean(recent_flags):5.1f}%"
                + (f" | frames {int(np.median(recent_frames))}" if args.speedrun and recent_frames else "")
                + (f" | pipe {100 * np.mean(recent_pipe):.0f}% of finishes" if args.pipe_mode and recent_pipe else "")
                + (f" | start {curriculum_start:.2f}" if args.curriculum else "")
                + (f" | critic warmup, value loss {v_loss.item():.3f}" if warming_up else "")
            )
            if recent_pipe:
                writer.add_scalar("train/pipe_rate_of_finishes", np.mean(recent_pipe), global_step)
            if recent_frames:
                writer.add_scalar("train/frames_to_flag_median", float(np.median(recent_frames)), global_step)

        # ---- Reverse curriculum: once Mario finishes reliably from here, start him earlier ----
        if args.curriculum and len(curriculum_results) >= args.curriculum_window:
            rate = float(np.mean(curriculum_results))
            curriculum_results.clear()
            writer.add_scalar("curriculum/finish_rate", rate, global_step)
            if rate >= args.curriculum_threshold and curriculum_start > 0:
                curriculum_start = max(0.0, curriculum_start - args.curriculum_step)
                for g in groups:
                    g.set_attr("curriculum_start", curriculum_start)  # groups are idle between rollouts
                print(f"  curriculum: {100 * rate:.0f}% of curriculum starts finished, "
                      f"now starting from the last {100 * (1 - curriculum_start):.0f}% of the demo")
            writer.add_scalar("curriculum/start", curriculum_start, global_step)

        # ---- Greedy evaluation: is it actually beating the level? ----
        if update % args.eval_every == 0:
            model.eval()
            win_rate, mean_x, frames0, pipe0 = evaluate(model, device, eval_envs, rng, frame0=args.speedrun)
            writer.add_scalar("eval/win_rate", win_rate, global_step)
            writer.add_scalar("eval/x_pos", mean_x, global_step)
            print(f"  eval: cleared {100 * win_rate:.0f}% of {args.eval_episodes} runs, mean x {mean_x:.0f}"
                  + (f" | frame-0 run: {frames0 if frames0 else 'did not finish'} frames" if args.speedrun else "")
                  + (" via the PIPE" if pipe0 else ""))
            if args.speedrun and frames0:
                writer.add_scalar("eval/frames_to_flag", frames0, global_step)
                if best_frames is None or frames0 < best_frames:
                    best_frames = frames0
                    torch.save({"model": model.state_dict(), "n_actions": n_actions, "obs_shape": obs_shape,
                                "args": vars(args), "frames": frames0, "pipe": pipe0},
                               os.path.join(args.save_dir, "fastest.pt"))
                    print(f"  new record: {frames0} frames ({frames0 / 60.0988:.2f} s)"
                          f"{' through the pipe' if pipe0 else ''}, saved fastest.pt")
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
             "best_eval": best_eval, "best_frames": best_frames, "curriculum_start": curriculum_start,
             "args": vars(args),
             "n_actions": n_actions, "obs_shape": obs_shape},
            latest,
        )

        if args.preview_every:
            recent_nets.append((update, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        if args.preview_every and update % args.preview_every == 0 and \
                (preview_render is None or preview_render.poll() is not None):
            # write the recent networks out; the preview process plays each one and shows the closest
            candidates = []
            for k, (u, weights) in enumerate(recent_nets):
                path = os.path.join(preview_dir, f"_candidate{k}.pt")
                torch.save({"model": weights, "n_actions": n_actions, "obs_shape": obs_shape,
                            "args": vars(args), "update": u}, path)
                candidates.append(path)
            first = recent_nets[0][0]
            label = f"closest of updates {first}-{update}, step {global_step:,}"
            preview_render = subprocess.Popen(
                [sys.executable, os.path.join(here, "watch_run.py"), "--ckpt", *candidates, "--no-window",
                 "--scale", "2", "--label", label, "--mp4", os.path.join(preview_dir, f"update_{update:05d}.mp4")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if streak >= args.solve_streak and not args.speedrun:
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
