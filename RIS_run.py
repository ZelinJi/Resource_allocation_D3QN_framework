from RIS_env import Maze
from RIS_env import Training
from RL_brain2 import DeepQNetwork
from load_model import Model
import numpy as np
import time
import os
Is_double_q = True

os.environ[("CUDA_VISIBLE_DEVICE")] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start = time.time()

n_elements = 8 # number of elements

width = 100
height = 100

duser_center = np.array([[55,5], [50,20], [10,80], [20,90]])
cuser_center = np.array([[25,30], [95,65], [50,85], [75,30]])
BS_center = [50, 50]



if Is_double_q:
    label = 'marl_model_double'
else:
    label = 'marl_model'

def run_maze(n_elements):

    iter_average = []
    loss_epoch = []
    loss_cumulative = []

    C_power_dB = 23  # dBm
    D_power_dB_List = [24, 21, 18, 15, 12, 9, 6, 3, 0]   # the power levels
    n_episode = 3000
    epsi_anneal_length = int(0.8 * n_episode)
    epsi_final = 0.02
    agent_index = {0:0, 2:1}

    observation = env.reset_ris(n_elements, agent_index)

    model = Model(BS_center, cuser_center, duser_center, C_power_dB, D_power_dB_List, label, n_elements)
    model.load_saved_models(agent_index) # load trained resource allocation models

    for i_episode in range(n_episode):

        if i_episode < epsi_anneal_length:
            # epsi = 1 - np.log(i_episode * (1 - epsi_final)) / np.log(epsi_anneal_length - 1)  # epsilon decreases over each episode for fast convergence
            epsi = 1 - (i_episode * (1 - epsi_final)) / (epsi_anneal_length - 1)  # linear decreases over each episode for better online performance
        else:
            # epsi =  1 - np.log(epsi_anneal_length * (1 - epsi_final)) / np.log(epsi_anneal_length - 1) # epsilon decreases over each episode for fast convergence
            epsi = 1 - (epsi_anneal_length * (1 - epsi_final)) / (epsi_anneal_length - 1)  # linear decreases over each episode for better online performance
        print('--------------------------------------------')
        print('Episode:', i_episode, 'Epsi', epsi)
        cumulative_reward = 0
        cumulative_loss = 0
        # env.reset_ris(n_elements, agent_index)
        for i_step in range (i_step_per_episode):
            time_step = i_episode * i_step_per_episode + i_step
            RIS_action, flag = RL.choose_action(observation, epsi)
            resource_allocation_action = model.get_resource_allocation(observation, BS_center, cuser_center, duser_center, C_power_dB, D_power_dB_List, RIS_action, n_elements, agent_index) # get the resource allocation results
            observation_, reward, done, thr = env.step(observation, RIS_action, resource_allocation_action, C_power_dB, D_power_dB_List, n_elements) # interact with the env

            cumulative_reward += reward

            RL.store_transition(observation, RIS_action, reward, observation_) # store the experience
            observation = observation_ # next state

            if time_step % mini_batch_step == mini_batch_step - 1:
                loss, Is_replacement = RL.learn() # learn at each mini_batch_step
            if i_step % i_step_per_episode == i_step_per_episode - 1:
                average_reward = cumulative_reward / (i_step + 1)
                loss_epoch.append(loss)
                loss_cumulative.append(cumulative_loss)
                iter_average.append(average_reward)
                print('Step:', time_step, 'Loss:', loss, 'Reward: ', average_reward)


    if Is_double_q:
        np.savetxt('./Train_data/Average_return_double', iter_average)
        np.savetxt('./Train_data/Loss_double', loss_epoch)
    else:
        np.savetxt('./Train_data/Average_return', iter_average)
        np.savetxt('./Train_data/Loss', loss_epoch)
    end = time.time()

    print("game over!")
    print('Running time:', end - start)

if __name__ == "__main__":
        n_phases = np.power(3, n_elements) # RIS phase adjustment choice
        n_positions = 16 # RIS location choice
        env = Maze(duser_center, cuser_center, BS_center, n_elements)
        i_step_per_episode = 100
        mini_batch_step = i_step_per_episode
        RL = DeepQNetwork(n_positions, env.n_features,
                          Is_double_q = Is_double_q,
                          learning_rate=0.001,
                          # learning_rate = 0.0005,
                          reward_decay= 0.95,
                          replace_target_iter= int(i_step_per_episode / mini_batch_step) * 10,
                          memory_size=100000,
                          batch_size=2000,
                          output_graph=True
                          ) # initialize centralized DDQN
        run_maze(n_elements)
        # RL.plot_cost()
