import gymnasium as gym
import numpy as np
from ppo_agent import PPOAgent
from buffer import RolloutBuffer
from torch.utils.tensorboard import SummaryWriter

def main():
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    obs, _ = env.reset()
    obs_dim = obs.shape
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(action_dim=action_dim, lr=3e-4, clip_range=0.25, n_epochs=10)
    buffer = RolloutBuffer()
    writer = SummaryWriter("logs/PPO_CarRacing_Optimized")

    update_timestep = 4000
    time_step = 0
    episode_num = 0

    print("---------------------------------------")
    print("Starting Optimized PPO Training...")
    print("---------------------------------------")

    while time_step < 100000:
        state, _ = env.reset()
        score = 0
        done = False

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward = np.clip(reward, -1, 1)
            buffer.add(state, action, log_prob, reward, done, value)

            state = next_state
            score += reward
            time_step += 1

            if time_step % update_timestep == 0:
                agent.update(buffer)
                print(f"Updated PPO at timestep {time_step}")

        episode_num += 1
        writer.add_scalar("Reward/Episode", score, episode_num)
        print(f"Episode {episode_num} | Score: {score:.2f} | Total Steps: {time_step}")

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
