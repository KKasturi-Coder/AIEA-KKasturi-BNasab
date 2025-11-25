import gymnasium as gym
import torch
import numpy as np
from ppo_agent import PPOAgent
from buffer import RolloutBuffer
from torch.utils.tensorboard import SummaryWriter

def main():
    # 1. Setup Environment
    # CarRacing-v2 is standard; render_mode="rgb_array" runs headless
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    
    # Hyperparameters
    action_dim = env.action_space.shape[0]
    max_training_timesteps = 100000
    update_timestep = 2000 # Update policy every 2000 steps
    
    # 2. Initialize Agent and Buffer
    agent = PPOAgent(action_dim)
    buffer = RolloutBuffer()
    writer = SummaryWriter("logs/PPO_CarRacing_Scratch") # TensorBoard logging

    print("---------------------------------------")
    print("Starting Training from Scratch...")
    print("---------------------------------------")

    time_step = 0
    episode_num = 0
    
    # 3. Main Training Loop
    while time_step < max_training_timesteps:
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            # Select action
            action, log_prob, val = agent.select_action(state)
            
            # Step env
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Save data to buffer
            buffer.add(state, action, log_prob, reward, done, val)
            
            state = next_state
            score += reward
            time_step += 1
            
            # Update PPO agent
            if time_step % update_timestep == 0:
                agent.update(buffer)
                print(f"Updated PPO at timestep {time_step}")

        # Log to TensorBoard
        episode_num += 1
        writer.add_scalar("Reward/Mean_Episode_Reward", score, time_step)
        print(f"Episode {episode_num} | Score: {score:.2f} | Total Steps: {time_step}")

    env.close()
    writer.close()

if __name__ == '__main__':
    main()
