"""Environment setup shared by the DQN and PPO trainers.

Everything Mario-specific lives in one wrapper so it is fast (no LazyFrames,
no lz4) and the observation is a plain uint8 array of shape (4, 84, 84).
"""
import ctypes
import json
import random
from collections import deque

import cv2
import gym
import gym_super_mario_bros
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.nes_env import _LIB
from nes_py.wrappers import JoypadSpace

# "complex" is every useful button combo, including down (pipes) and up.
# Its first 5 entries are exactly RIGHT_ONLY, which makes action-layer surgery simple.
ACTION_SETS = {"right_only": RIGHT_ONLY, "simple": SIMPLE_MOVEMENT, "complex": COMPLEX_MOVEMENT}

# Multiple save states need the patched nes-py from tools/build_nes_py.sh
HAS_SNAPSHOTS = hasattr(_LIB, "Snapshot")
if HAS_SNAPSHOTS:
    _LIB.Snapshot.argtypes = [ctypes.c_void_p]
    _LIB.Snapshot.restype = ctypes.c_void_p
    _LIB.LoadSnapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _LIB.LoadSnapshot.restype = None
    _LIB.FreeSnapshot.argtypes = [ctypes.c_void_p]
    _LIB.FreeSnapshot.restype = None

A_BIT = 0b00000001  # the A button in the NES controller byte


class MarioWrapper(gym.Wrapper):
    """Frame skip + grayscale/resize + frame stack + stuck detection + rewards.

    - skip:          repeat each action for this many NES frames
    - stack:         number of past frames the agent sees (lets it see motion)
    - stuck_steps:   end the episode if Mario's best x position hasn't improved
                     for this many agent steps (stops wasting time at a pipe)
    - flag_bonus:    extra reward for reaching the flag (reward_mode "x")
    - savestate_prob: chance that an episode starts from a save state made at
                     a point Mario reached before, instead of the level start.
                     Evaluation keeps this at 0.
    - savestate_spacing: keep at most one save per this many pixels of level
    - buttons:       the action list given to JoypadSpace. Used for auto-release:
                     the NES only starts a jump when A is *pressed*, so if Mario is
                     standing on the ground with A still held, A is let go for one frame.
    - time_cost:     subtract this much per NES frame (0 = off)
    - speed_bonus:   at the flag, add this much per frame finished under ref_frames
    - accurate_timing: turn off gym-super-mario-bros's animation skipping (it edits game
                     timers to fast-forward pipes and area changes). Times then match
                     the real game, which matters for the pipe route.
    - reward_mode:   "x"      = the env's reward (pixels moved right, clock, death) + bonuses
                     "finish" = route-neutral: only finishing (finish_bonus + speed_bonus)
                                and dying/getting stuck (death_penalty) count, plus
                                progress_weight * the env's reward as an optional nudge.
                                The env's x reward ignores teleports, so it punishes
                                the pipe shortcut; "finish" doesn't care which way he goes.
    - curriculum:    path to a demo JSON (see make_demo.py). Save states are made along
                     the demo, and with probability curriculum_prob an episode starts at
                     one of them, chosen from the last part of the demo set by
                     curriculum_start (1.0 = only the very end, 0.0 = anywhere).
                     The trainer moves curriculum_start back as Mario gets reliable.
    """

    def __init__(self, env, skip=4, stack=4, size=84, stuck_steps=150, flag_bonus=50.0,
                 savestate_prob=0.0, savestate_spacing=200, buttons=None,
                 time_cost=0.0, speed_bonus=0.0, ref_frames=1400,
                 accurate_timing=False, reward_mode="x", finish_bonus=1000.0, death_penalty=500.0,
                 progress_weight=0.0, curriculum=None, curriculum_prob=0.0, curriculum_start=1.0,
                 curriculum_every=16):
        super().__init__(env)
        # For each action that holds A: the same action without A
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
        self.time_cost = time_cost
        self.speed_bonus = speed_bonus
        self.ref_frames = ref_frames
        self.reward_mode = reward_mode
        self.finish_bonus = finish_bonus
        self.death_penalty = death_penalty
        self.progress_weight = progress_weight
        self.savestate_prob = savestate_prob if HAS_SNAPSHOTS else 0.0
        self.savestate_spacing = savestate_spacing
        self.saves = {}  # bucket (x // spacing) -> (snapshot handle, frames elapsed when saved)
        self.from_save = False
        self.frames = deque(maxlen=stack)
        self.raw = None  # last full-colour frame, used for the tiled view
        self.observation_space = gym.spaces.Box(0, 255, (stack, size, size), dtype=np.uint8)
        # The actual emulator env, under gym's wrappers (needed for save states)
        nes = env
        while not hasattr(nes, "_x_position"):
            nes = nes.env
        self.nes = nes

        # Speedrun timing: count every NES frame since the level started.
        # step() runs one frame; the env may also run extra frames itself through
        # _frame_advance while skipping animations, counted separately.
        self.frame_count = 0
        self.skipped_frames = 0
        original_advance = nes._frame_advance

        def counted_advance(action):
            self.skipped_frames += 1
            original_advance(action)

        nes._frame_advance = counted_advance

        self.accurate_timing = accurate_timing
        if accurate_timing:
            nes._skip_change_area = lambda: None
            nes._skip_occupied_states = lambda: None

        # Reverse curriculum from a demo
        self.curriculum_path = curriculum if HAS_SNAPSHOTS else None
        self.curriculum_prob = curriculum_prob
        self.curriculum_start = curriculum_start
        self.curriculum_every = curriculum_every
        self.curriculum_states = None  # built lazily on the first reset
        self.from_curriculum = False
        self.start_x = 0
        self.used_pipe = False

    def _process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def _obs(self):
        return np.stack(self.frames, axis=0)

    # ---- save states ----
    def _maybe_save(self, info, done):
        if done or self.savestate_prob <= 0 or self.used_pipe:
            return
        nes = self.nes
        ram = nes.ram
        on_ground = ram[0x001D] == 0   # not jumping or falling
        normal = ram[0x000E] == 0x08   # normal player state (not dying, not in a pipe)
        bucket = info.get("x_pos", 0) // self.savestate_spacing
        if bucket == 0 or not (on_ground and normal) or self.from_save:
            return
        frames = self.frame_count + self.skipped_frames
        old = self.saves.get(bucket)
        if old is not None:
            if frames >= old[1]:
                return  # keep the save from the faster arrival
            _LIB.FreeSnapshot(old[0])
        self.saves[bucket] = (_LIB.Snapshot(nes._env), frames)

    def _load_random_save(self):
        """Load a random save. Returns the screen and the frame count it was saved at."""
        nes = self.nes
        handle, frames = self.saves[random.choice(list(self.saves))]
        _LIB.LoadSnapshot(nes._env, handle)
        nes._did_reset()  # resync the reward bookkeeping (x, time, score) to the loaded state
        return nes.screen.copy(), frames

    def _build_curriculum(self):
        """Replay the demo's controller bytes frame by frame and save states along the way."""
        with open(self.curriculum_path) as f:
            demo = json.load(f)
        assert demo.get("accurate_timing", False) == self.accurate_timing, \
            "the demo was recorded with a different accurate_timing setting"
        self.env.reset()
        nes = self.nes
        states = []
        for i, b in enumerate(demo["bytes"]):
            _, _, done, info = nes.step(b)
            if done:
                break
            if (i + 1) % self.curriculum_every == 0 and nes.ram[0x000E] == 0x08:
                states.append((_LIB.Snapshot(nes._env), i + 1, bool(b & A_BIT), int(info["x_pos"])))
        self.curriculum_states = states

    def _load_curriculum_state(self):
        n = len(self.curriculum_states)
        lo = int(self.curriculum_start * (n - 1))
        handle, frames, a_held, x = self.curriculum_states[random.randint(lo, n - 1)]
        _LIB.LoadSnapshot(self.nes._env, handle)
        self.nes._did_reset()
        return self.nes.screen.copy(), frames, a_held

    def close(self):
        for handle, _ in self.saves.values():
            _LIB.FreeSnapshot(handle)
        self.saves.clear()
        for handle, *_ in self.curriculum_states or []:
            _LIB.FreeSnapshot(handle)
        self.curriculum_states = None
        super().close()

    # ---- gym API ----
    def reset(self, **kwargs):
        if self.curriculum_path and self.curriculum_states is None:
            self._build_curriculum()
        obs, info = self.env.reset(**kwargs)
        self.frame_count = 0
        self.skipped_frames = 0
        self.a_held = False
        self.used_pipe = False
        self.from_curriculum = bool(self.curriculum_states) and random.random() < self.curriculum_prob
        self.from_save = self.from_curriculum or (bool(self.saves) and random.random() < self.savestate_prob)
        if self.from_curriculum:
            obs, self.frame_count, self.a_held = self._load_curriculum_state()
        elif self.from_save:
            # carry over the frames it took to get there, so speed rewards stay honest
            obs, self.frame_count = self._load_random_save()
        self.raw = obs
        self.best_x = self.start_x = int(self.nes._x_position)
        self.no_progress = 0
        frame = self._process(obs)
        for _ in range(self.stack):
            self.frames.append(frame)
        return self._obs(), info

    def step(self, action):
        env_reward = 0.0
        terminated = truncated = False
        action = int(action)
        presses_a = action in self.release_a
        on_ground = self.nes.ram[0x001D] == 0
        frames_before = self.frame_count + self.skipped_frames
        for i in range(self.skip):
            frame_action = action
            if i == 0 and presses_a and self.a_held and on_ground:
                frame_action = self.release_a[action]  # let go of A so the next frame is a fresh press
            obs, reward, terminated, truncated, info = self.env.step(frame_action)
            self.frame_count += 1
            env_reward += reward
            if terminated or truncated:
                break
        self.a_held = presses_a
        self.raw = obs
        self.frames.append(self._process(obs))

        frames_now = self.frame_count + self.skipped_frames
        flag = info.get("flag_get", False)
        if flag:
            terminated = True

        x = info.get("x_pos", 0)
        if x < self.best_x - 200:
            # a big jump backwards is a new area (pipe / bonus room), not lost progress
            self.used_pipe = True
            self.best_x = x
            self.no_progress = 0
        elif x > self.best_x:
            self.best_x = x
            self.no_progress = 0
        else:
            self.no_progress += 1
            if self.stuck_steps and self.no_progress >= self.stuck_steps:
                truncated = True

        time_part = -self.time_cost * (frames_now - frames_before)
        speed_part = self.speed_bonus * max(0, self.ref_frames - frames_now) if flag else 0.0
        if self.reward_mode == "finish":
            total_reward = self.progress_weight * env_reward + time_part + speed_part
            if flag:
                total_reward += self.finish_bonus
            elif terminated or truncated:
                total_reward -= self.death_penalty
        else:
            total_reward = env_reward + time_part + speed_part + (self.flag_bonus if flag else 0.0)

        self._maybe_save(info, terminated or truncated)
        info["from_save"] = self.from_save
        info["from_curriculum"] = self.from_curriculum
        info["start_x"] = self.start_x
        info["pipe"] = self.used_pipe
        info["frames"] = frames_now  # from a save: includes the frames it took to reach the save
        info["skipped_frames"] = self.skipped_frames
        return self._obs(), total_reward, terminated, truncated, info


def make_env(level="1-1", actions="right_only", render_mode="rgb_array", **wrapper_kwargs):
    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{level}-v0", render_mode=render_mode, apply_api_compatibility=True
    )
    env = JoypadSpace(env, ACTION_SETS[actions])
    return MarioWrapper(env, buttons=ACTION_SETS[actions], **wrapper_kwargs)


def play_env_kwargs(train_args):
    """The settings needed to replay a checkpoint the way it was trained (for eval/watch scripts)."""
    return dict(level=train_args["level"], actions=train_args["actions"],
                skip=train_args.get("skip", 4), accurate_timing=train_args.get("accurate_timing", False))


def apply_wrappers(env):
    """Kept for the older scripts: wrap an env that already has JoypadSpace."""
    return MarioWrapper(env, buttons=RIGHT_ONLY)
