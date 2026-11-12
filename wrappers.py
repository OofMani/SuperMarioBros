"""Environment setup shared by the DQN and PPO trainers.

Everything Mario-specific lives in one wrapper so it is fast (no LazyFrames,
no lz4) and the observation is a plain uint8 array of shape (4, 84, 84).
"""
import random
from collections import deque

import cv2
import gym
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.nes_env import _LIB
from nes_py.wrappers import JoypadSpace

ACTION_SETS = {"right_only": RIGHT_ONLY, "simple": SIMPLE_MOVEMENT}

# Multiple save states need the patched nes-py from tools/build_nes_py.sh
import ctypes

HAS_SNAPSHOTS = hasattr(_LIB, "Snapshot")
if HAS_SNAPSHOTS:
    _LIB.Snapshot.argtypes = [ctypes.c_void_p]
    _LIB.Snapshot.restype = ctypes.c_void_p
    _LIB.LoadSnapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _LIB.LoadSnapshot.restype = None
    _LIB.FreeSnapshot.argtypes = [ctypes.c_void_p]
    _LIB.FreeSnapshot.restype = None


class MarioWrapper(gym.Wrapper):
    """Frame skip + grayscale/resize + frame stack + stuck detection + flag bonus.

    - skip:          repeat each action for this many NES frames
    - stack:         number of past frames the agent sees (lets it see motion)
    - stuck_steps:   end the episode if Mario's best x position hasn't improved
                     for this many agent steps (stops wasting time at a pipe)
    - flag_bonus:    extra reward for reaching the flag
    - savestate_prob: chance that an episode starts from a save state made at
                     a point Mario reached before, instead of the level start.
                     He then practises the later parts of the level without
                     replaying the start every time. Evaluation keeps this at 0.
    - savestate_spacing: keep at most one save per this many pixels of level
    - buttons:       the action list given to JoypadSpace (e.g. RIGHT_ONLY). Used for
                     auto-release: the NES only starts a jump when A is *pressed*, so
                     if Mario is standing on the ground with A still held from before,
                     A is let go for one frame first. Otherwise a policy that keeps
                     choosing right+A gets stuck pushing against a tall pipe forever.
    """

    def __init__(self, env, skip=4, stack=4, size=84, stuck_steps=150, flag_bonus=50.0,
                 savestate_prob=0.0, savestate_spacing=200, buttons=None):
        super().__init__(env)
        # For each action that holds A: the same action without A (None if it has no A)
        self.release_a = {}
        if buttons is not None:
            sets = [frozenset(b) - {"NOOP"} for b in buttons]
            for i, b in enumerate(sets):
                if "A" in b:
                    self.release_a[i] = next((j for j, c in enumerate(sets) if c == b - {"A"}), 0)
        self.a_held = False
        self.skip = skip
        self.stack = stack
        self.size = size
        self.stuck_steps = stuck_steps
        self.flag_bonus = flag_bonus
        self.savestate_prob = savestate_prob if HAS_SNAPSHOTS else 0.0
        self.savestate_spacing = savestate_spacing
        self.saves = {}  # bucket (x // spacing) -> snapshot handle
        self.from_save = False
        self.frames = deque(maxlen=stack)
        self.raw = None  # last full-colour frame, used for the tiled view
        self.observation_space = gym.spaces.Box(0, 255, (stack, size, size), dtype=np.uint8)
        # The actual emulator env, under gym's wrappers (needed for save states)
        nes = env
        while not hasattr(nes, "_x_position"):
            nes = nes.env
        self.nes = nes

    def _process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def _obs(self):
        return np.stack(self.frames, axis=0)

    # ---- save states ----
    def _maybe_save(self, info, done):
        if done or self.savestate_prob <= 0:
            return
        nes = self.nes
        ram = nes.ram
        on_ground = ram[0x001D] == 0   # not jumping or falling
        normal = ram[0x000E] == 0x08   # normal player state (not dying, not in a pipe)
        bucket = info.get("x_pos", 0) // self.savestate_spacing
        if bucket == 0 or not (on_ground and normal) or bucket in self.saves:
            return
        self.saves[bucket] = _LIB.Snapshot(nes._env)

    def _load_random_save(self):
        nes = self.nes
        _LIB.LoadSnapshot(nes._env, self.saves[random.choice(list(self.saves))])
        nes._did_reset()  # resync the reward bookkeeping (x, time, score) to the loaded state
        return nes.screen.copy()

    def close(self):
        for handle in self.saves.values():
            _LIB.FreeSnapshot(handle)
        self.saves.clear()
        super().close()

    # ---- gym API ----
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.from_save = bool(self.saves) and random.random() < self.savestate_prob
        if self.from_save:
            obs = self._load_random_save()
        self.raw = obs
        self.best_x = int(self.nes._x_position)
        self.no_progress = 0
        self.a_held = False
        frame = self._process(obs)
        for _ in range(self.stack):
            self.frames.append(frame)
        return self._obs(), info

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        action = int(action)
        presses_a = action in self.release_a
        on_ground = self.nes.ram[0x001D] == 0
        for i in range(self.skip):
            frame_action = action
            if i == 0 and presses_a and self.a_held and on_ground:
                frame_action = self.release_a[action]  # let go of A so the next frame is a fresh press
            obs, reward, terminated, truncated, info = self.env.step(frame_action)
            total_reward += reward
            if terminated or truncated:
                break
        self.a_held = presses_a
        self.raw = obs
        self.frames.append(self._process(obs))

        if info.get("flag_get", False):
            total_reward += self.flag_bonus
            terminated = True

        x = info.get("x_pos", 0)
        if x > self.best_x:
            self.best_x = x
            self.no_progress = 0
        else:
            self.no_progress += 1
            if self.stuck_steps and self.no_progress >= self.stuck_steps:
                truncated = True

        self._maybe_save(info, terminated or truncated)
        info["from_save"] = self.from_save
        return self._obs(), total_reward, terminated, truncated, info


def make_env(level="1-1", actions="right_only", render_mode="rgb_array", **wrapper_kwargs):
    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{level}-v0", render_mode=render_mode, apply_api_compatibility=True
    )
    env = JoypadSpace(env, ACTION_SETS[actions])
    return MarioWrapper(env, buttons=ACTION_SETS[actions], **wrapper_kwargs)


def apply_wrappers(env):
    """Kept for the older scripts: wrap an env that already has JoypadSpace."""
    return MarioWrapper(env, buttons=RIGHT_ONLY)
