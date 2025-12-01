import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from ppo_agent import PPOAgent
from buffer import RolloutBuffer
from torch.utils.tensorboard import SummaryWriter


# Different PPO settings to compare
configs = {
    "PPO_lr3e-4_clip0.25": {"lr": 3e-4, "clip": 0.25, "epochs": 10},
    "PPO_lr1e-4_clip0.20": {"lr": 1e-4, "clip": 0.20, "epochs": 10},
    "PPO_lr3e-4_clip0.10": {"lr": 3e-4, "clip": 0.10, "epochs": 15},
}


def run_experiment(name, cfg, total_steps=20000):
    print(f"\nRunning {name}")

    results_dir = f"results/{name}"
    tb_dir = f"results/tb/{name}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)

    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    obs, _ = env.reset()

    action_dim = env.action_space.shape[0]

    agent = PPOAgent(
        action_dim=action_dim,
        lr=cfg["lr"],
        clip_range=cfg["clip"],
        n_epochs=cfg["epochs"]
    )
    agent.device = "cpu"  # avoid GPU mismatch issues

    buffer = RolloutBuffer()

    update_timestep = 4000
    timestep = 0
    episode = 0

    reward_log = []
    update_steps = []

    while timestep < total_steps:
        state, _ = env.reset()
        done = False
        score = 0

        while not done:
            action, logprob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            reward = np.clip(reward, -1, 1)

            buffer.add(state, action, logprob, reward, done, value)
            state = next_state

            score += reward
            timestep += 1

            if timestep % update_timestep == 0:
                agent.update(buffer)
                update_steps.append(timestep)
                writer.add_scalar("Update/Step", timestep, timestep)
                print(f"{name}: update at step {timestep}")

        episode += 1
        reward_log.append(score)
        writer.add_scalar("Reward/Episode", score, episode)
        print(f"{name}: episode {episode}, reward {score:.2f}")

    np.save(f"{results_dir}/rewards.npy", np.array(reward_log))
    np.save(f"{results_dir}/update_steps.npy", np.array(update_steps))

    writer.close()
    env.close()

    return reward_log


def plot_all_results(all_rewards):
    plt.figure(figsize=(10, 5))

    for name, rewards in all_rewards.items():
        plt.plot(rewards, label=name)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Comparison")
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/compare_rewards.png")
    plt.close()


def main():
    all_rewards = {}

    for name, cfg in configs.items():
        rewards = run_experiment(name, cfg)
        all_rewards[name] = rewards

    plot_all_results(all_rewards)

    print("\nFinished. Plots saved in results/")
    print("TensorBoard logs in results/tb/")
    print("Run: tensorboard --logdir results/tb --port 6006")


if __name__ == "__main__":
    main()

