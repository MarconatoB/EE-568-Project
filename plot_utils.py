import numpy as np
import matplotlib.pyplot as plt

def aggregate_over_seeds(results):
    max_timesteps = np.max([len(seed_data[0]) for seed_data in results])
    max_episodes = np.max([len(seed_data[1]) for seed_data in results])
    max_grad_steps = np.max([len(seed_data[2]) for seed_data in results])
    
    reward_agg = np.empty((max_timesteps, len(results)))    # total reward
    reward_agg[:] = np.nan

    critic_agg = np.empty((max_timesteps, len(results)))    # critic loss
    critic_agg[:] = np.nan

    actor_agg = np.empty((max_timesteps, len(results)))     # actor loss
    actor_agg[:] = np.nan

    episodes_agg = np.empty((max_episodes, len(results)))   # episode reward
    episodes_agg[:] = np.nan


    for seed, seed_data in enumerate(results):
        for t in range(max_timesteps): 
            try:
                reward_agg[t, seed] = seed_data[0][t]
            except IndexError:
                pass    # just leave it as NaN

        for episode in range(max_episodes):
            try:
                episodes_agg[episode, seed] = seed_data[1][episode]
            except IndexError:
                pass

        for grad_step in range(max_grad_steps):
            try:
                critic_agg[grad_step, seed] = seed_data[2][grad_step]
                actor_agg[grad_step, seed] = seed_data[3][grad_step]
            except IndexError:
                pass

    mean_rewards_t = np.nanmean(reward_agg, axis=1)
    mean_rewards_ep = np.nanmean(episodes_agg, axis=1)
    mean_critic_loss = np.nanmean(critic_agg, axis=1)
    mean_actor_loss = np.nanmean(actor_agg, axis=1)

    return mean_rewards_t, mean_rewards_ep, mean_critic_loss, mean_actor_loss

def plot_results(results, env_name: str):
    env_result = results[env_name]
    
    if isinstance(env_result, list):
        rewards_t, episodes_reward, critic_losses, actor_losses = aggregate_over_seeds(env_result)
    else:
        rewards_t, episodes_reward, critic_losses, actor_losses = results[env_name]

    fig, ax1 = plt.subplots()
    critic_plot = ax1.plot(critic_losses, label='critic', color='C0')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Critic loss')
    ax2 = ax1.twinx()
    actor_plot = ax2.plot(actor_losses, label='actor', color='C1')
    ax2.set_ylabel('Actor loss (= -Q1)')
    fig.legend()
    fig.suptitle(f"Losses, {env_name}")
    plt.show()

    plt.figure(figsize=(20,6))
    episodes = np.arange(0, len(episodes_reward))
    plt.plot(episodes_reward, '.-', label=env_name, color='C0')
    plt.fill_between(episodes, episodes_reward, color='C0', alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Total episode reward")
    plt.legend(loc="lower right")
    plt.show()
