# Super Mario Bros Reinforcement Learning

An AI that learns to play Super Mario Bros (NES) World 1-1 from the screen pixels.

It started years ago as a DQN agent, inspired by a YouTube video
[here](https://youtu.be/_gmQZToTMac), that never finished the level. It was rebuilt
with PPO and parallel training, and it now clears 1-1 every time and speedruns it
in **1,172 frames (19.50 s)** overland. It's now learning to find the pipe shortcut.

## Results so far

| Milestone | Result |
|---|---|
| Original DQN (≈2022) | Never finished 1-1 |
| PPO, first clear with no randomness | ~1.39M training steps |
| PPO, first 5/5 perfect evaluation | ~1.66M steps (update 810) |
| `solved.pt` (3 perfect evaluations in a row) | Clears 1-1 from 20/20 random start delays |
| Solved run, timed | 1,240 frames = 20.63 s (in-game timer 339 at the flag) |
| Speedrun fine-tune, first round | 1,208 frames = 20.10 s |
| **Speedrun with critic warmup + 5× speed bonus, best** | **1,172 frames = 19.50 s** (stable 1,177 on 100% of tests) |

## Setup

```bash
conda create -n mario-ppo python=3.11
conda activate mario-ppo
pip install -r requirements.txt
bash tools/build_nes_py.sh     # recommended: ~46% faster emulator + multiple save states
```

## Scripts

| Script | What it does |
|---|---|
| `train_ppo.py` | PPO training with 16 parallel Marios (`--watch` to see them tiled, `--resume` to continue) |
| `train_ppo.py --speedrun --init-from <ckpt>` | Fine-tune a solved Mario to finish faster (see below) |
| `play_ppo.py` | Watch a checkpoint play (`solved.pt` → `best.pt` → `latest.pt`) |
| `speedrun_eval.py` | Time a checkpoint exactly from frame 0 (`--delays 20` checks robustness, `--save` writes JSON) |
| `preview_window.py <run>/previews` | Loop the newest training preview at real speed (opened automatically by training) |
| `make_demo.py overland\|pipe` | Record a frame-exact demo for the reverse curriculum |
| `watch_run.py` | Play a run at true 60 fps with a frame counter and a virtual NES controller showing the buttons held on every frame (`--mp4` saves a video, `--speed 0.25` slow motion) |
| `main.py` | The original DQN approach, fixed up |
| `tensorboard --logdir runs` | Training charts (`eval/win_rate`, `eval/x_pos`, `eval/frames_to_flag`, `train/flag_rate_100`, ...) |

## How it works

- **PPO with an actor-critic CNN** (`ppo_model.py`). The input is the last 4 frames, grayscale, 84×84.
  The actor picks a button combo and the critic estimates how good the situation is.
- **16 Marios in parallel** (`vec_env.py`), split into two groups that take turns, so the network
  picks actions for one group while the other group's games run.
- **Mario wrapper** (`wrappers.py`):
  - frame skip 4, frame stack, and stuck detection
  - a flag bonus
  - **auto-release of A** (see the lessons below)
  - frame counting for speedruns
  - optional speed rewards
- **Save-state starts**: 30% of training episodes start mid-level from saved states, so Mario
  practises the hard parts. This needs the patched emulator, which can hold many save states.
- **Evaluation** always plays the network's top action, with no randomness, from random start
  delays. The emulator is deterministic, so a policy that memorised one exact path fails this test.

## Progress log

### 1. Reviving the old DQN
The original agent had several problems that stopped it from ever finishing:
- `gamma=0.9`: it only planned about 40 frames ahead.
- `eps_min=0.1`: 10% of moves stayed random forever, even when testing.
- It rendered every frame while training.
- A float32 replay buffer that would have needed about 22 GB.
- No Apple GPU support.
- No detection of a stuck Mario.

All of these were fixed in `agent.py` and `main.py`. The main effort then moved to PPO.

### 2. PPO and speed
- The first PPO version with 8 parallel Marios ran at about 300 steps/s.
- The emulator turned out to be the bottleneck, at only 784 NES frames/s per core.
  Rebuilding nes-py with Apple Silicon optimisations raised that to **1,146 frames/s (+46%)**.
- A small C++ patch added multiple save states, which made the mid-level starts possible.
- Splitting 16 Marios into two groups that take turns, plus running the evaluation games in
  parallel, brought training to about **600–700 steps/s**. That's close to the Mac's practical
  limit; 6,000 steps/s would need roughly 40 or more CPU cores.

### 3. Getting it to clear 1-1
| Problem | Symptom | Fix |
|---|---|---|
| Greedy test Mario undecided between "run" and "jump" | The test always died at the first pipe (x=434) for ~400k steps | Lowered the entropy bonus from 0.01 to 0.005 |
| **A-held trap**: the NES only jumps when A is *pressed*, and Mario can't see whether A is held | The test pushed against the tall pipe (x≈722) until the timer ran out | Wrapper releases A for 1 frame when Mario is on the ground |
| Evaluations swinging (100% → 0%) after solving | The learning rate was still high | Resumed with `--lr 5e-5`, and the solve criterion was met |

The A-held fix was the turning point. Within about 20k steps the test distance jumped from 722 to
2,082, Mario made his first clear, and the training flag rate went from 0% to 12% and then on to 98%.

### 4. Speedrun (branch `speedrun-1-1`)
- **Measurement.** The wrapper counts every NES frame from the level start to grabbing the flagpole.
  The in-game timer ticks every 20 frames in this emulator. Runs that went through the
  emulator's skipped animations are flagged, because their times wouldn't match the real game.
- **Baseline.** `solved.pt` finishes in **1,240 frames**. It held up for every random start
  delay (best 1,239 frames, worst 1,303).
- **Speedrun fine-tune** (`--speedrun`): a per-frame time cost, a flag bonus that grows the faster
  he finishes, gamma 0.997 and lr 5e-5, starting from `solved.pt`. Result: **1,208 frames (20.10 s)**.
- **Pushing further didn't work**:
  - Speed bonus 50, reward scale 0.01 and lr 1e-4: **collapsed** (flag rate 98% → 2%).
  - Speed bonus 5: slid from 100% to 52%.
  - Cause: changing the reward made the critic's value estimates wrong (value loss about 40),
    so PPO judged every move against wrong expectations and wrecked the actor.
  - **Fix**: `--critic-warmup N` trains only the critic's last layer for N updates while the
    actor stays frozen. It stopped the collapse, with the flag rate staying at 88–97%.
    But the speed still didn't improve past 1,208, and the training median drifted from about
    1,250 to about 1,270 frames, which looked like a plateau.
  - **It wasn't a plateau, just a slow start.** The same run kept going, and the median fell to
    about 1,190. Records came in at 1,197 → 1,177 → **1,172 frames (19.50 s)**, and the frame-0
    test settled on a stable 1,177 that cleared every evaluation. After that it really did
    plateau, at about 3 SMB1 frame rules faster than the solved baseline.

### Lessons
1. **What the agent can observe and how its actions work matter as much as the algorithm.**
   The A-held trap cost about a million training steps. No amount of hyperparameter tuning
   fixes an action whose result depends on something the agent can't see.
2. **The entropy bonus pays the agent for being undecided.** That's good early on, but it can
   keep the greedy "always pick the top move" policy stuck exactly where decisions are close.
3. **Changing the reward on a trained agent requires recalibrating the critic first**,
   or PPO will damage the policy.
4. **RL plateaus a few frames short of optimal.** 4-frame steps can't express frame-perfect
   timing, which is where search (TAS-style) or a different route comes in.

## Pipe route (branch `pipe-route`)

### What the emulator investigation showed
- The solved Mario **never stands on the enterable pipe** (the 4th one, at x≈912). He jumps
  clean over it, so he'd never get the chance to press Down.
- Landing on the pipe top and pressing Down works, but only in a **narrow window**:
  x=911 worked, while x=923–935 on the same pipe top did nothing.
- The bonus room needs a precise sequence: jump onto the brick block, drop to the floor
  (running off the block at full speed lands you *on top of* the exit pipe), then walk
  into the sideways exit. Mario comes back out at **x=2,616**.
- gym-super-mario-bros **skips 138 frames** of pipe animations by editing game timers, so
  pipe times are only honest with those skips turned off (`--accurate-timing`).

| Route (game-accurate timing) | Frames to flag | Timer at flag |
|---|---|---|
| Overland, `solved.pt` | 1,240 | 338 |
| Pipe, clumsy script + solved Mario | 1,098 | 361 |
| Speedrunners' reference | — | 370 |

### Pipe mode
```bash
python make_demo.py overland --ckpt models/ppo/solved.pt   # demos/overland.json (committed)
python make_demo.py pipe     --ckpt models/ppo/solved.pt   # demos/pipe.json (committed, scripted route)
python train_ppo.py --pipe-mode --init-from models/ppo/solved.pt
```
- **All 12 button combos**, including Down. The 5-button solved model is expanded with
  action-layer surgery: its rows are copied, and the new buttons start at bias -2, so it
  plays exactly like `solved.pt` at first.
- **Game-accurate timing**, so no animation skips.
- **Reward**: a finish bonus of 1,000, plus 5 per frame under 1,400, -500 for dying or getting
  stuck, plus 10% of the old x reward as a nudge. The pipe still pays clearly overall.
- **Reverse curriculum**: half the episodes start from save states along a demo, at first
  only near the flag. The start moves back by 5% of the demo whenever 70% of those
  episodes finish.
- Critic warmup (30 updates) runs automatically when starting from a checkpoint.
- Training lines show `pipe N%` (the share of finished runs that used the pipe), and
  evaluations say `via the PIPE` when the record run takes it.
- **Previews**: every 30 updates (`--preview-every`), the best run so far (`fastest.pt`, or
  else `best.pt`) is replayed in the background and saved to `runs/<run>/previews/update_N.mp4`,
  labelled with the update, the step, and whether it cleared and took the pipe. A window
  (`preview_window.py`) loops the newest preview at real speed. Use `--no-preview-window` to
  only save the videos.
- `--curriculum demos/pipe.json` is the "assisted" variant. It shows Mario that the
  route exists without telling him to take it.

## The original plan for the pipe-route branch
Speedrunners reach about 370 on the in-game timer in 1-1 by taking the enterable pipe to the
underground bonus room. Our overland run finishes at 339. The next branch will try to let the AI
find that route:
- **all buttons**, including Down, added by "surgery" on the solved model's action layer
  (the existing 5 actions are copied, and the new ones start with a strongly negative score)
- a **route-neutral reward** with a massive finish and speed bonus, because the current
  "pixels moved right" reward punishes the pipe shortcut
- **game-accurate pipe timing**, by turning off the emulator's animation skipping
- a **reverse curriculum** from save states, because a finish-only reward is too sparse to learn from scratch
