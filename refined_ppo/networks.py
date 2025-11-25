import torch
import torch.nn as nn

class PPOPolicy(nn.Module):
    def __init__(self, action_dim):
        super(PPOPolicy, self).__init__()

        # Shared CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU()
        )

        # Actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

        # Learnable std
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * 0.5)

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self):
        raise NotImplementedError("Use act() or get_value() instead.")

    def act(self, state):
        # state: (batch, H, W, C) from gym
        state = state.permute(0, 3, 1, 2)  # to (batch, C, H, W)
        features = self.cnn(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        value = self.critic(features)
        return action_mean, action_std, value

    def get_value(self, state):
        state = state.permute(0, 3, 1, 2)
        features = self.cnn(state)
        return self.critic(features)
