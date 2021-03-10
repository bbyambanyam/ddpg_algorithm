import gym
import numpy as np
import matplotlib.pyplot as plt
import gc
from torch.autograd import Variable
import torch
import torch.nn.functional as F

import memory
import models
import utilities


if __name__ == "__main__":
    # Orchin beldeh
    env = gym.make('BipedalWalker-v3')

    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.shape[0]
    action_max = env.action_space.high[0]

    print("State dimension: {}" .format(state_dimension))
    print("Action dimension: {}" .format(action_dimension))
    print("Action max: {}" .format(action_max))

    # Actor network, critic network uusgeh

    actor = models.Actor(state_dimension, action_dimension, action_max)
    target_actor = models.Actor(state_dimension, action_dimension, action_max)
    actor_optimizer = torch.optim.Adam(actor.parameters(), 0.001)

    critic = models.Critic(state_dimension, action_dimension)
    target_critic = models.Critic(state_dimension, action_dimension)
    critic_optimizer = torch.optim.Adam(critic.parameters(), 0.001)

    # Target network-g huulah

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)

    # Replay buffer uusgeh

    ram = memory.ReplayBuffer(1000000)

    # Reward-iig hadgalah list

    reward_list = []
    average_reward_list = []

    for ep in range(100):

        # Anhnii state-g awah

        observation = env.reset()

        ep_reward = 0

        for step in range(1000):
            env.render()
            state = np.float32(observation)

            # Action-g songoh

            tmp_state = Variable(torch.from_numpy(state))
            action_without_noise = actor.forward(tmp_state).detach()

            # Noise uusgeh

            noise = utilities.OrnsteinUhlenbeckActionNoise(action_dimension)

            # Action-d noise nemeh

            action_with_noise = action_without_noise.data.numpy() + (noise.sample() * action_max)

            # Action-g hiij shine state, reward awah

            new_observation, reward, done, info = env.step(action_with_noise)

            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation)

                # Replay buffer-d state, action, reward, new_state -g hadgalah

                ram.add(state, action_with_noise, reward, new_state)

            observation = new_observation

            ep_reward += reward

            # Replay buffer-aas 128 bagts turshalagiig random-oor awna

            state, action_with_noise, reward, next_state = ram.sample_exp(128)

            state = Variable(torch.from_numpy(state))
            action_with_noise = Variable(torch.from_numpy(action_with_noise))
            reward = Variable(torch.from_numpy(reward))
            next_state = Variable(torch.from_numpy(next_state))

            # Critic network-g surgah

            predicted_action = target_actor.forward(next_state).detach()
            next_val = torch.squeeze(target_critic.forward(next_state, predicted_action)).detach()
            y_expected = reward + 0.99*next_val
            y_predicted = torch.squeeze(critic.forward(state, action_with_noise))

            # Critic network-g shinechleh, critic loss-g tootsooloh

            critic_loss = F.smooth_l1_loss(y_predicted, y_expected)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor network-g surgah

            predicted_action = actor.forward(state)
            actor_loss = -1*torch.sum(critic.forward(state, predicted_action))
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Target network-g shinechleh

            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - 0.001) + param.data * 0.001)

            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - 0.001) + param.data * 0.001)
            
            if done:
                break
        
        # reward-g hadgalj awn

        reward_list.append(ep_reward)
        average_reward = np.mean(reward_list[-40:])
        print("Episode: {} Average Reward: {}" .format(ep, average_reward))
        average_reward_list.append(average_reward)
        
        gc.collect()
        
    # Reward-g durslen haruulah

    plt.plot(average_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Reward")
    plt.show()


