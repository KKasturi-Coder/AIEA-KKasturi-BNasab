import torch
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        """Resets the buffer for the next iteration."""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, state, action, action_logprob, reward, is_terminal, state_value):
        """Adds a single step of experience to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.state_values.append(state_value)

    def compute_returns_and_advantages(self, gamma=0.99):
        """
        Calculates the discounted rewards (returns) for PPO.
        This is a simplified version of GAE (Generalized Advantage Estimation).
        """
        rewards = []
        discounted_reward = 0
        
        # Iterate backwards through the rewards to calculate discounted return
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards helps training stability
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        return rewards
