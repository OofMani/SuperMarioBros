"""Run several Mario games at once, each in its own process.

Each worker owns one emulator. The main process sends actions to all of them
at the same time and gets back a batch of observations, so the network sees
8 games' worth of experience for the cost of roughly one.
"""
import multiprocessing as mp

import cv2
import numpy as np


def _worker(remote, env_kwargs):
    from wrappers import make_env  # imported here so each process builds its own emulator

    env = make_env(**env_kwargs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(int(data))
                done = terminated or truncated
                small = {
                    "x_pos": int(info.get("x_pos", 0)),
                    "flag_get": bool(info.get("flag_get", False)),
                    "from_save": bool(info.get("from_save", False)),
                    "frames": int(info.get("frames", 0)),
                }
                if done:
                    obs, _ = env.reset()
                remote.send((obs, reward, done, small))
            elif cmd == "reset":
                obs, _ = env.reset()
                remote.send(obs)
            elif cmd == "render":
                remote.send(env.raw)
            elif cmd == "close":
                break
    finally:
        env.close()
        remote.close()


class SubprocVecEnv:
    def __init__(self, num_envs, env_kwargs):
        ctx = mp.get_context("spawn")
        self.num_envs = num_envs
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.procs = []
        for i, wr in enumerate(work_remotes):
            p = ctx.Process(target=_worker, args=(wr, env_kwargs), daemon=True)
            p.start()
            wr.close()
            self.procs.append(p)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.stack([r.recv() for r in self.remotes])

    def step_async(self, actions):
        """Send actions and return immediately; the games run while we do other work."""
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))

    def step_wait(self):
        obs, rewards, dones, infos = zip(*[r.recv() for r in self.remotes])
        return np.stack(obs), np.array(rewards, dtype=np.float32), np.array(dones), list(infos)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render_frames(self):
        for r in self.remotes:
            r.send(("render", None))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=2)


def tile_frames(frames, infos=None, cols=4, scale=1.0):
    """Arrange the colour frames from every env into one grid image (BGR for OpenCV)."""
    h, w, _ = frames[0].shape
    rows = int(np.ceil(len(frames) / cols))
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
        label = f"#{i}"
        if infos is not None:
            label += f"  x={infos[i]['x_pos']}"
        cv2.putText(grid, label, (c * w + 6, r * h + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    if scale != 1.0:
        grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return grid
