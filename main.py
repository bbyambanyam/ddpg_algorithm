import gc
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn.functional as F


import memory
import models
import utilities


if __name__ == "__main__":

    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_LEARNING_RATE = 0.001
    MAX_EPISODE = 1200
    NOISE_TYPE = "Parameter" #OU, OUBaseline, Parameter, Uncorrelated
    # Orchin beldeh
    env = gym.make('BipedalWalker-v3')

    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.shape[0]
    action_max = env.action_space.high[0]

    print("State dimension: {}" .format(state_dimension))
    print("Action dimension: {}" .format(action_dimension))
    print("Action max: {}" .format(action_max))

    load_models = False

    # Actor network, critic network uusgeh

    actor = models.Actor(state_dimension, action_dimension, action_max)
    target_actor = models.Actor(state_dimension, action_dimension, action_max)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE)

    critic = models.Critic(state_dimension, action_dimension)
    target_critic = models.Critic(state_dimension, action_dimension)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE)

    # Target network-g huulah

    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(param.data)

    # Hadgalsan modeliig ashiglah

    if load_models:
        actor.load_state_dict(torch.load('./Models/' + str(0) + '_actor.pt'))
        critic.load_state_dict(torch.load('./Models/' + str(0) + '_critic.pt'))

        for target_param, param in zip(target_actor.parameters(), actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(param.data)

        print("Models loaded!")

    # Replay buffer uusgeh

    ram = memory.ReplayBuffer(1000000)

    # Reward-iig hadgalah list

    reward_list = []
    average_reward_list = []

    # Noise uusgeh

    if NOISE_TYPE == "OU":
        noise = utilities.OrnsteinUhlenbeckActionNoise(action_dimension)
    elif NOISE_TYPE == "OUBaseline":
        noise = utilities.OrnsteinUhlenbeckActionNoiseBaseline(mu=np.zeros(action_dimension), sigma=float(0.2) * np.ones(action_dimension))
    elif NOISE_TYPE == "Parameter":
        #Parameter noise-d zoriulsan actor
        actor_copy = models.Actor(state_dimension, action_dimension, action_max)

        #Parameter noise
        parameter_noise = utilities.AdaptiveParamNoiseSpec(initial_stddev=0.05,desired_action_stddev=0.3, adaptation_coefficient=1.05)
        noise = utilities.OrnsteinUhlenbeckActionNoise(action_dimension)
        
    print("Noise type: ", NOISE_TYPE)

    for ep in range(MAX_EPISODE):

        # Anhnii state-g awah

        observation = env.reset()

        ep_reward = 0
        step_cntr = 0

        if NOISE_TYPE == "Parameter":
            #Actor-g actor_copy-d huulah

            for target_param, param in zip(actor_copy.parameters(), actor.parameters()):
                target_param.data.copy_(param.data)

            # Parameter noise-iig neural suljeen deer nemeh

            parameters = actor_copy.state_dict()
            for name in parameters:
                parameter = parameters[name]
                rand_number = torch.randn(parameter.shape)
                parameter = parameter + rand_number * parameter_noise.current_stddev

        for step in range(env._max_episode_steps):
            env.render()
            state = np.float32(observation)

            # Action-g songoh

            tmp_state = Variable(torch.from_numpy(state))
            action_without_noise = actor.forward(tmp_state).detach()

            if NOISE_TYPE == "OU":
                #OU processiin noisetoi action
                action = action_without_noise.data.numpy() + (noise.sample() * action_max)
            elif NOISE_TYPE == "OUBaseline":
                #OU Baseline noisetoi action
                action = action_without_noise.data.numpy() + noise()
            elif NOISE_TYPE == "Parameter":
                action_with_parameter_noise = actor_copy.forward(tmp_state).detach()
                #Parameter noisetoi action
                action = action_with_parameter_noise.numpy() + (noise.sample() * action_max)
            elif NOISE_TYPE == "Uncorrelated":
                #[-0.2, 0.2] random noisetoi action
                action = action_without_noise.data.numpy() + (np.random.uniform(-0.2,0.2) * action_max)
            else:
                raise RuntimeError('Buruu turliin noise: "{}"'.format(NOISE_TYPE)) 
                
            # Action-g hiij shine state, reward awah

            new_observation, reward, done, info = env.step(action)

            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation)

                # Replay buffer-d state, action, reward, new_state -g hadgalah

                ram.add(state, action, reward, new_state)
                ep_reward += reward

            observation = new_observation

            # Replay buffer-aas 128 bagts turshalagiig random-oor awna

            states, actions, rewards, next_states = ram.sample_exp(128)

            states = Variable(torch.from_numpy(states))
            actions = Variable(torch.from_numpy(actions))
            rewards = Variable(torch.from_numpy(rewards))
            next_states = Variable(torch.from_numpy(next_states))

            # Critic network-g surgah

            predicted_action = target_actor.forward(next_states).detach()
            next_val = torch.squeeze(target_critic.forward(next_states, predicted_action).detach())
            y_expected = rewards + 0.99*next_val
            y_predicted = torch.squeeze(critic.forward(states, actions))

            # Critic network-g shinechleh, critic loss-g tootsooloh
            
            critic_loss = F.smooth_l1_loss(y_predicted, y_expected)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor network-g surgah

            predicted_action = actor.forward(states)
            actor_loss = -1*torch.sum(critic.forward(states, predicted_action))
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

            step_cntr += 1

        if NOISE_TYPE == "Parameter":
            #Noisetoi actor deer hiigdsen data-g list-d hadgalj awaad suuliin episode-d hiigdsen stepiin toogoor datagaa awna
            
            noise_data_list = list(ram.buffer)
            noise_data_list = np.array(noise_data_list[-step_cntr:])

            actor_copy_state, actor_copy_action, _, _ = zip(*noise_data_list)

            #Noisetoi actoriin action
            actor_copy_actions = np.array(actor_copy_action)

            #Engiin actoriin action

            actor_actions = []
            for state in np.array(actor_copy_state):
                state = Variable(torch.from_numpy(state))
                action = actor.forward(state).detach().numpy()
                actor_actions.append(action)

            #Distance tootsoh
            diff_actions = actor_copy_actions - actor_actions
            mean_diff_actions = np.mean(np.square(diff_actions),axis=0)
            distance = math.sqrt(np.mean(mean_diff_actions))

            #Sigma-g update hiih
            parameter_noise.adapt(distance)

        #Model-iig hadgalah
        if ep % 100 == 0:
            torch.save(target_actor.state_dict(), './Models/' + str(ep) + '_actor.pt')
            torch.save(target_critic.state_dict(), './Models/' + str(ep) + '_critic.pt')
            print("Target actor, critic models saved")
        
        # reward-g hadgalj awna

        reward_list.append(ep_reward)
        average_reward = np.mean(reward_list[-40:])
        print("Episode: {} Average Reward: {}" .format(ep, average_reward))
        average_reward_list.append(average_reward)
        
        gc.collect()
        
    # Reward-g durslen haruulah

    print("Reward max: ", max(average_reward_list))

    plt.plot(average_reward_list, label = NOISE_TYPE)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Average Episode Reward")
    plt.show()
