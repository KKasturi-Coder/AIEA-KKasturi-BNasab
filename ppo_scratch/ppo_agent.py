import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import PPOPolicy

class PPOAgent:
    def __init__(self, action_dim, lr=3e-4, gamma=0.99, clip_range=0.2, n_epochs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy = PPOPolicy(action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        # Loss coefficients
        self.c1 = 0.5 # Value loss weight
        self.c2 = 0.01 # Entropy weight

    def select_action(self, state):
        """Selects an action for the environment loop."""
        with torch.no_grad():
            # Convert state to tensor and add batch dimension
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action distribution
            action_mean, action_std, state_value = self.policy.act(state)
            dist = torch.distributions.Normal(action_mean, action_std)
            
            # Sample action
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(axis=-1)
            
        # Return as numpy arrays/scalars for the environment
        return action.cpu().numpy().flatten(), action_logprob.cpu().item(), state_value.cpu().item()

    def update(self, buffer):
        """Core PPO Update Loop"""
        # 1. Get Batch of Data
        # Convert lists to tensors
        old_states = torch.FloatTensor(np.array(buffer.states)).to(self.device)
        old_actions = torch.FloatTensor(np.array(buffer.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(np.array(buffer.logprobs)).to(self.device)
        
        # 2. Compute Returns and Advantages (GAE)
        rewards = buffer.compute_returns_and_advantages(self.gamma)
        rewards = rewards.to(self.device)
        
        # 3. Optimize Policy for K epochs
        for _ in range(self.n_epochs):
            # Evaluate old actions under CURRENT policy
            action_mean, action_std, state_values = self.policy.act(old_states)
            dist = torch.distributions.Normal(action_mean, action_std)
            
            logprobs = dist.log_prob(old_actions).sum(axis=-1)
            dist_entropy = dist.entropy().sum(axis=-1)
            state_values = state_values.squeeze()
            
            # --- PPO LOSS CALCULATION ---
            
            # A. Probability Ratio (r_t)
            ratios = torch.exp(logprobs - old_logprobs)

            # B. Advantages
            # A_t = Returns - V(s)
            advantages = rewards - state_values.detach()
            # Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            # C. Clipped Surrogate Objective (L_CLIP)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_range, 1+self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # D. Value Loss (L_VF)
            value_loss = nn.MSELoss()(state_values, rewards)

            # E. Total Loss
            loss = policy_loss + (self.c1 * value_loss) - (self.c2 * dist_entropy.mean())
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear buffer after update
        buffer.clear()
