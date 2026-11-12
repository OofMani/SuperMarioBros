# Super Mario Bros Reinforcement Learning

This is based of a DQN Model designed to play Super Mario Bros.

This was inspired by a YouTube video [here](https://youtu.be/_gmQZToTMac).

## Setup

```bash
conda create -n mario-ppo python=3.11
conda activate mario-ppo
pip install -r requirements.txt
bash tools/build_nes_py.sh     # optional but recommended: ~45% faster emulator + save states
```

## PPO with parallel Marios (recommended)

```bash
python train_ppo.py            # 16 Marios in parallel, no window (fastest)
python train_ppo.py --watch    # same, but shows them all tiled in one window (press q to hide)
python train_ppo.py --resume   # continue from models/ppo/latest.pt
python play_ppo.py             # watch the trained agent play
tensorboard --logdir runs      # training graphs
```

Speed tricks: the Marios run in two groups that take turns, so the network picks
actions for one group while the other group's games are running. About 30% of training
episodes start from save states further into the level (`--savestate-prob`), so Mario
practises the hard parts instead of replaying the start every time.

Training stops by itself once the agent clears the level with no randomness on
every evaluation run, 3 evaluations in a row, and saves `models/ppo/solved.pt`.

## DQN (the original approach, fixed up)

```bash
python main.py                 # add --display to watch
```

Changes from the original: gamma 0.99 instead of 0.9, epsilon goes from 1.0 down to 0.05
and evaluation uses no randomness, Double DQN with Huber loss, a uint8 replay buffer
(about 6x less memory), training on the Apple GPU (MPS), and episodes that end early
when Mario is stuck.
