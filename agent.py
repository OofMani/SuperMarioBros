import numpy as np
import torch

from agent_nn import AgentNN, pick_device


class ReplayBuffer:
    """Stores transitions as uint8 in plain numpy arrays.

    The next state is the current stack shifted by one frame, so only the one
    new frame is stored instead of a whole second stack. 100k transitions take
    about 3.5 GB of RAM (the old float32 version needed about 22 GB).
    """

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_frames = np.zeros((capacity, *obs_shape[1:]), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done):
        i = self.idx
        self.states[i] = state
        self.next_frames[i] = next_state[-1]
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        ids = np.random.randint(0, self.size, size=batch_size)
        states = self.states[ids]
        next_states = np.concatenate([states[:, 1:], self.next_frames[ids][:, None]], axis=1)
        to = lambda a: torch.from_numpy(a).to(device)
        return to(states), to(self.actions[ids]), to(self.rewards[ids]), to(next_states), to(self.dones[ids])


class Agent:
    def __init__(self,
                 input_dims,
                 num_actions,
                 lr=1e-4,
                 gamma=0.99,            # was 0.9: now it plans ~100 steps ahead, not ~10
                 eps_start=1.0,
                 eps_min=0.05,          # was 0.1, and evaluation now uses 0
                 eps_decay_steps=1_000_000,
                 replay_buffer_capacity=100_000,
                 batch_size=32,
                 learn_every=4,
                 learn_starts=10_000,
                 sync_network_rate=10_000,
                 device="auto"):

        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay_steps = eps_decay_steps
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.learn_starts = learn_starts
        self.sync_network_rate = sync_network_rate
        self.step_counter = 0
        self.learn_step_counter = 0

        self.device = pick_device(device)
        self.online_network = AgentNN(input_dims, num_actions).to(self.device)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=lr)
        self.loss = torch.nn.SmoothL1Loss()  # Huber loss: big mistakes don't blow up training
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity, input_dims)

    def choose_action(self, observation, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        obs = torch.from_numpy(np.asarray(observation)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_network(obs).argmax().item()

    def decay_epsilon(self):
        frac = min(1.0, self.step_counter / self.eps_decay_steps)
        self.epsilon = self.eps_start + frac * (self.eps_min - self.eps_start)

    def store_in_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(np.asarray(state), action, reward, np.asarray(next_state), done)

    def save_model(self, path):
        torch.save(self.online_network.state_dict(), path)

    def load_model(self, path):
        state = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(state)
        self.target_network.load_state_dict(state)

    def learn(self):
        """Call once per environment step; it only trains every `learn_every` steps."""
        self.step_counter += 1
        self.decay_epsilon()
        if len(self.replay_buffer) < self.learn_starts or self.step_counter % self.learn_every:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        q = self.online_network(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            # Double DQN: the online net picks the next action, the target net scores it
            next_actions = self.online_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.sync_network_rate == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
