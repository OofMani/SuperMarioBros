import torch
from torch import nn

from agent_nn import pick_device  # noqa: F401  (re-exported for train/play scripts)


def _init(layer, gain=2 ** 0.5):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """One CNN that looks at the screen, with two small heads on top:

    - actor:  a score (logit) for every button combo -> turned into probabilities
    - critic: a single number, "how good is this situation?"
    """

    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.body = nn.Sequential(
            _init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )
        self.actor = _init(nn.Linear(512, n_actions), gain=0.01)
        self.critic = _init(nn.Linear(512, 1), gain=1.0)

    def forward(self, obs_uint8):
        x = self.body(obs_uint8.float() / 255.0)
        return self.actor(x), self.critic(x).squeeze(-1)

    def act(self, obs_uint8, greedy=False):
        logits, value = self(obs_uint8)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), value


def expand_actions(state_dict, old_buttons, new_buttons, new_bias=-3.0):
    """Action-layer surgery: give a trained model more buttons without losing what it knows.

    The vision layers and the critic are kept as they are. Each old action's row of
    the actor's output layer is copied to wherever the same button combo sits in the
    new action list. New combos get small random weights and a negative bias, so at
    first they're rarely picked (Mario plays as before) but exploration can still find them.
    """
    old_sets = [frozenset(b) for b in old_buttons]
    new_sets = [frozenset(b) for b in new_buttons]
    w_old, b_old = state_dict["actor.weight"], state_dict["actor.bias"]
    w_new = torch.empty(len(new_sets), w_old.shape[1])
    nn.init.orthogonal_(w_new, 0.01)
    b_new = torch.full((len(new_sets),), float(new_bias))
    for i, s in enumerate(old_sets):
        if s not in new_sets:
            raise ValueError(f"old action {sorted(s)} is missing from the new action set")
        j = new_sets.index(s)
        w_new[j], b_new[j] = w_old[i], b_old[i]
    out = dict(state_dict)
    out["actor.weight"], out["actor.bias"] = w_new, b_new
    return out
