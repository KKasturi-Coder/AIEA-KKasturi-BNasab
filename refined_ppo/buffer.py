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

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        """
        Computes Generalized Advantage Estimation (GAE) returns.
        """
        rewards = self.rewards
        values = self.state_values + [0]
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step+1] * (1 - self.is_terminals[step]) - values[step]
            gae = delta + gamma * lam * (1 - self.is_terminals[step]) * gae
            returns.insert(0, gae + values[step])
        
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        return returns
