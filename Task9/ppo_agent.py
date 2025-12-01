import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import PPOPolicy

class PPOAgent:
    def __init__(self, action_dim, lr=3e-4, gamma=0.99, clip_range=0.2, n_epochs=10):
        self.device = torch.device("cpu")
        
        # Initialize policy network
        self.policy = PPOPolicy(action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        
        # Loss coefficients
        self.c1 = 0.5   # Value loss
        self.c2 = 0.02  # Increased entropy bonus for exploration

    def select_action(self, state):
        """Selects an action for the environment loop."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action distribution
            action_mean, action_std, state_value = self.policy.act(state)
            dist = torch.distributions.Normal(action_mean, action_std)
            
            # Sample action
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(axis=-1)
        
        return action.cpu().numpy().flatten(), action_logprob.cpu().item(), state_value.cpu().item()

    def update(self, buffer):
        """Core PPO Update Loop"""
        # Convert lists to tensors
        old_states = torch.FloatTensor(np.array(buffer.states)).to(self.device)
        old_actions = torch.FloatTensor(np.array(buffer.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(buffer.logprobs)).to(self.device)
        
        # Compute returns and advantages using GAE
        returns = buffer.compute_returns_and_advantages(self.gamma).to(self.device)
        
        for _ in range(self.n_epochs):
            # Evaluate old actions under current policy
            action_mean, action_std, state_values = self.policy.act(old_states)
            dist = torch.distributions.Normal(action_mean, action_std)
            
            logprobs = dist.log_prob(old_actions).sum(axis=-1)
            dist_entropy = dist.entropy().sum(axis=-1)
            state_values = state_values.squeeze()
            
            # --- PPO LOSS CALCULATION ---
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = returns - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = nn.MSELoss()(state_values, returns)
            
            loss = policy_loss + self.c1 * value_loss - self.c2 * dist_entropy.mean()
            
            # Optimize with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffer
        buffer.clear()
