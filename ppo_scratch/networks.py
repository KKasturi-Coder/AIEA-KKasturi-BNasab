import torch
import torch.nn as nn
import numpy as np

class PPOPolicy(nn.Module):
    def __init__(self, action_dim):
        super(PPOPolicy, self).__init__()

        # 1. Shared CNN Feature Extractor
        # Input: (Batch, 3, 96, 96) -> Output: (Batch, 512)
        # We use the standard "Nature CNN" architecture often used in RL
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), # Layer 1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # Layer 2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # Layer 3
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), # Map flattened features to 512 vector
            nn.ReLU()
        )

        # 2. Actor Head (Policy)
        # Input: 512 features -> Output: action_dim (mean of actions)
        self.actor_mean = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Squashes output between -1 and 1 (required for CarRacing)
        )
        
        # Learnable standard deviation (log_std) for exploration
        # We use a parameter so it can be optimized
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # 3. Critic Head (Value Function)
        # Input: 512 features -> Output: 1 scalar value (V(s))
        self.critic = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self):
        raise NotImplementedError("Use act() or get_value() instead.")

    def act(self, state):
        """
        Given a state image, return the action distribution parameters and value.
        """
        # CRITICAL FIX: Permute dimensions from (Batch, Height, Width, Channels) 
        # to (Batch, Channels, Height, Width) which PyTorch expects.
        # Gym gives: [1, 96, 96, 3] -> PyTorch wants: [1, 3, 96, 96]
        state = state.permute(0, 3, 1, 2)
        
        # Process image through CNN
        features = self.cnn(state)
        
        # Get Action Mean
        action_mean = self.actor_mean(features)
        
        # Expand log_std to match batch size
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Get Value
        value = self.critic(features)

        return action_mean, action_std, value

    def get_value(self, state):
        """
        Helper to just get the value V(s) for GAE calculation.
        """
        # CRITICAL FIX: Apply the same permutation here
        state = state.permute(0, 3, 1, 2)
        
        features = self.cnn(state)
        return self.critic(features)
