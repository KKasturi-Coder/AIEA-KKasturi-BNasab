import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────────
device = torch.device("cpu")
torch.set_num_threads(4)

REPLAY_SIZE = 100000
MIN_REPLAY = 5000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # entropy temperature

# TensorBoard writer
writer = SummaryWriter("runs/sac_logs")

# ────────────────────────────────────────────────────────────────────────────────
# REPLAY BUFFER
# ────────────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen=size)

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.as_tensor(np.array(obs), dtype=torch.float32, device=device) / 255.0
        next_obs = torch.as_tensor(np.array(next_obs), dtype=torch.float32, device=device) / 255.0
        actions = torch.as_tensor(np.array(actions), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.as_tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1)

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)

# ────────────────────────────────────────────────────────────────────────────────
# CNN ENCODER
# ────────────────────────────────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            dummy = self.conv(dummy)
            self.flat_size = dummy.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

# ────────────────────────────────────────────────────────────────────────────────
# ACTOR
# ────────────────────────────────────────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = CNNEncoder()
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

    def forward(self, obs):
        h = self.encoder(obs)
        mean = self.fc_mean(h)
        logstd = torch.clamp(self.fc_logstd(h), -5, 2)
        std = torch.exp(logstd)

        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        action = torch.tanh(action)
        return action, log_prob, mean

# ────────────────────────────────────────────────────────────────────────────────
# CRITIC
# ────────────────────────────────────────────────────────────────────────────────
class Critic(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = CNNEncoder()
        self.q1 = nn.Sequential(nn.Linear(256 + action_dim, 256), nn.ReLU(),
                                nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(256 + action_dim, 256), nn.ReLU(),
                                nn.Linear(256, 1))

    def forward(self, obs, action):
        h = self.encoder(obs)
        x = torch.cat([h, action], dim=-1)
        return self.q1(x), self.q2(x)

# ────────────────────────────────────────────────────────────────────────────────
# SAC AGENT
# ────────────────────────────────────────────────────────────────────────────────
class SAC:
    def __init__(self, action_dim):
        self.actor = Actor(action_dim).to(device)
        self.critic = Critic(action_dim).to(device)
        self.critic_target = Critic(action_dim).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.replay = ReplayBuffer(REPLAY_SIZE)

    def soft_update(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

    def update(self, step):
        if len(self.replay) < MIN_REPLAY:
            return None

        obs, actions, rewards, next_obs, dones = self.replay.sample(BATCH_SIZE)

        # ----- Critic update -----
        with torch.no_grad():
            next_action, next_logp, _ = self.actor(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_action)
            q_target = torch.min(q1_target, q2_target) - ALPHA * next_logp
            target_value = rewards + GAMMA * (1 - dones) * q_target

        q1, q2 = self.critic(obs, actions)
        critic_loss = ((q1 - target_value) ** 2 + (q2 - target_value) ** 2).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ----- Actor update -----
        new_action, logp, _ = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (ALPHA * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update()

        # ---- TensorBoard logging ----
        writer.add_scalar("Loss/Critic", critic_loss.item(), step)
        writer.add_scalar("Loss/Actor", actor_loss.item(), step)
        writer.add_scalar("Replay/Size", len(self.replay), step)

        return float(critic_loss.item()), float(actor_loss.item())

# ────────────────────────────────────────────────────────────────────────────────
# TRAIN LOOP
# ────────────────────────────────────────────────────────────────────────────────
def train_sac():
    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)
    agent = SAC(action_dim=3)

    obs, _ = env.reset()

    for step in range(150000):

        if step < MIN_REPLAY:
            action = np.random.uniform(-1, 1, size=3)
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _ = agent.actor(obs_t)
            action = action.cpu().detach().numpy()[0]

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay.add(obs, action, reward, next_obs, done)
        obs = next_obs

        if done:
            obs, _ = env.reset()

        result = agent.update(step)

        if (step + 1) % 1000 == 0:
            print(f"[step {step+1}] replay={len(agent.replay)}")

        if result:
            critic_l, actor_l = result

    env.close()
    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    train_sac()
