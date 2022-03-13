from __future__ import division, print_function
import scipy.io
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import Environment_marl
import os
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_ddqnAGS'] = '--tf_xla_enable_xla_devices'
my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth=True
plt.rcParams['figure.dpi'] = 300

class Agent_DQN(object):
    def __init__(self, memory_entry_size):
        self.discount = 1
        self.double_q = True
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)


class Agent_DDQN(object):
    def __init__(self, memory_entry_size_DDQN):
        self.discount_DDQN = 1
        self.double_q_DDQN = True
        self.memory_entry_size_DDQN = memory_entry_size_DDQN
        self.memory_DDQN = ReplayMemory(self.memory_entry_size_DDQN)


# ################## SETTINGS ######################
duser_center = np.array([[55,5], [50,20], [10,80], [20,90]])
cuser_center = np.array([[25,30], [95,65], [50,85], [75,30]])
BS_center = [50, 50]

# This main file is for testing only
IS_TRAIN = 0 # hard-coded to 0
IS_TEST = 1-IS_TRAIN

label_DQN = 'marl_model'
label_DDQN = 'marl_model_double'

n_D2D = 4
n_neighbor = 1
n_RB = n_D2D

C_power_dB = 23  # dBm
D_power_dB_List = [24, 21, 18, 15, 12, 9, 6, 3, 0]   # the power levels

env = Environment_marl.Environ(BS_center, cuser_center, duser_center, C_power_dB, D_power_dB_List) # initialize parameters in env
n_episode = 5000
#n_step_per_episode = int(env.time_slow/env.time_fast)
n_step_per_episode = 1 # using random testing samples
epsi_final = 0.02
epsi_anneal_length = int(0.8*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*10

n_episode_test = 30  # test episodes

######################################################


def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """
    C_fast = (env.C_channels_with_fastfading[idx[0], :] - env.C_channels_abs[idx[0]] + 10)/35

    D_fast = (env.D_channels_with_fastfading[:, env.D2D_users[idx[0]].destinations[idx[1]], :] - env.D_channels_abs[:, env.D2D_users[idx[0]].destinations[idx[1]]] + 10)/35
    D_interference = (-env.D2D_Interference_all[idx[0], idx[1], :] - 60) / 60

    C_abs = (env.C_channels_abs[idx[0]] - 80) / 60.0
    D_abs = (env.D_channels_abs[:, env.D2D_users[idx[0]].destinations[idx[1]]] - 80)/60.0
    return np.concatenate((np.reshape(C_fast, -1), np.reshape(D_fast, -1), D_interference, np.reshape(C_abs, -1), D_abs, np.asarray([ind_episode, epsi])))

# -----------------------------------------------------------
n_hidden_1 = 500
n_hidden_2 = 250
n_hidden_3 = 120
n_input = len(get_state(env=env))
n_output = n_RB * len(env.D_power_dB_List)

g = tf.Graph()
with g.as_default():
    # ============== Training network ========================
    x = tf.placeholder(tf.float32, [None, n_input])

    w_1 = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3 = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4 = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3 = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4 = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w_1), b_1))
    layer_1_b = tf.layers.batch_normalization(layer_1)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, w_2), b_2))
    layer_2_b = tf.layers.batch_normalization(layer_2)
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, w_3), b_3))
    layer_3_b = tf.layers.batch_normalization(layer_3)
    y = tf.nn.relu(tf.add(tf.matmul(layer_3, w_4), b_4))
    g_q_action = tf.argmax(y, axis=1)

    # compute loss
    g_target_q_t = tf.placeholder(tf.float32, None, name="target_value")
    g_action = tf.placeholder(tf.int32, None, name='g_action')
    action_one_hot = tf.one_hot(g_action, n_output, 1.0, 0.0, name='action_one_hot')
    q_acted = tf.reduce_sum(y * action_one_hot, reduction_indices=1, name='q_acted')

    g_loss = tf.reduce_mean(tf.square(g_target_q_t - q_acted), name='g_loss')
    optim = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(g_loss)
    # ==================== Prediction network ========================
    x_p = tf.placeholder(tf.float32, [None, n_input])

    w_1_p = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
    w_2_p = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
    w_3_p = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
    w_4_p = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

    b_1_p = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
    b_2_p = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
    b_3_p = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
    b_4_p = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

    layer_1_p = tf.nn.relu(tf.add(tf.matmul(x_p, w_1_p), b_1_p))
    layer_1_p_b = tf.layers.batch_normalization(layer_1_p)

    layer_2_p = tf.nn.relu(tf.add(tf.matmul(layer_1_p_b, w_2_p), b_2_p))
    layer_2_p_b = tf.layers.batch_normalization(layer_2_p)

    layer_3_p = tf.nn.relu(tf.add(tf.matmul(layer_2_p_b, w_3_p), b_3_p))
    layer_3_p_b = tf.layers.batch_normalization(layer_3_p)

    y_p = tf.nn.relu(tf.add(tf.matmul(layer_3_p_b, w_4_p), b_4_p))

    g_target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')
    target_q_with_idx = tf.gather_nd(y_p, g_target_q_idx)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


def predict(sess, s_t, ep, test_ep = False):

    n_power_levels = len(env.D_power_dB_List)
    if np.random.rand() < ep and not test_ep:
        pred_action = np.random.randint(n_RB*n_power_levels)
    else:
        pred_action = sess.run(g_q_action, feed_dict={x: [s_t]})[0]
    return pred_action


def q_learning_mini_batch(current_agent, current_sess):
    """ Training a sampled mini-batch """

    batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = current_agent.memory.sample()

    if current_agent.double_q:  # double q-learning
        pred_action = current_sess.run(g_q_action, feed_dict={x: batch_s_t_plus_1})
        q_t_plus_1 = current_sess.run(target_q_with_idx, {x_p: batch_s_t_plus_1, g_target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})
        batch_target_q_t = current_agent.discount * q_t_plus_1 + batch_reward
    else:
        q_t_plus_1 = current_sess.run(y_p, {x_p: batch_s_t_plus_1})
        max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
        batch_target_q_t = current_agent.discount * max_q_t_plus_1 + batch_reward

    _, loss_val = current_sess.run([optim, g_loss], {g_target_q_t: batch_target_q_t, g_action: batch_action, x: batch_s_t})
    return loss_val


def update_target_q_network(sess):
    """ Update target q network once in a while """

    sess.run(w_1_p.assign(sess.run(w_1)))
    sess.run(w_2_p.assign(sess.run(w_2)))
    sess.run(w_3_p.assign(sess.run(w_3)))
    sess.run(w_4_p.assign(sess.run(w_4)))

    sess.run(b_1_p.assign(sess.run(b_1)))
    sess.run(b_2_p.assign(sess.run(b_2)))
    sess.run(b_3_p.assign(sess.run(b_3)))
    sess.run(b_4_p.assign(sess.run(b_4)))


def save_models(sess, model_path):
    """ Save models to the current directory with the name filename """

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "model/" + model_path)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    saver.save(sess, model_path, write_meta_graph=False)


def load_models(sess, model_path):
    """ Restore models from the current directory with the name filename """

    dir_ = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_, "model/" + model_path)
    saver.restore(sess, model_path)


def print_weight(sess, target=False):
    """ debug """

    if not target:
        print(sess.run(w_1[0, 0:4]))
    else:
        print(sess.run(w_1_p[0, 0:4]))


# --------------------------------------------------------------
agents_dqn = []
sesses_dqn = []
agents_ddqn = []
sesses_ddqn = []

for ind_agent in range(n_D2D * n_neighbor):  # initialize agents for DQN
    print("Initializing agent", ind_agent)
    agent_dqn = Agent_DQN(memory_entry_size=len(get_state(env)))
    agents_dqn.append(agent_dqn)

    sess_dqn = tf.Session(graph=g,config=my_config)
    sess_dqn.run(init)
    sesses_dqn.append(sess_dqn)
    
for ind_agent in range(n_D2D * n_neighbor):  # initialize agents for DDQN
    print("Initializing agent", ind_agent)
    agent_ddqn = Agent_DDQN(memory_entry_size_DDQN=len(get_state(env)))
    agents_ddqn.append(agent_ddqn)

    sess_ddqn = tf.Session(graph=g,config=my_config)
    sess_ddqn.run(init)
    sesses_ddqn.append(sess_ddqn)


# -------------- Testing --------------
if IS_TEST:
    for i in range(n_D2D):
        for j in range(n_neighbor):
            print("\nRestoring the DQN model...", i)
            model_path_dqn = label_DQN + '/agent_' + str(i * n_neighbor + j)
            load_models(sesses_dqn[i * n_neighbor + j], model_path_dqn)
            
            
    for i in range(n_D2D):
        for j in range(n_neighbor):
            print("\nRestoring the DDQN model...", i)
            model_path_ddqn = label_DDQN + '/agent_' + str(i * n_neighbor + j)
            load_models(sesses_ddqn[i * n_neighbor + j], model_path_ddqn)
            

    Sum_rate_list_dqn = []
    Sum_rate_list_ddqn = []
    Sum_rate_list_rand = []
    Sum_rate_list_max = []

    rate_dqn = np.zeros([n_episode_test, n_step_per_episode, n_D2D, n_neighbor])
    rate_ddqn = np.zeros([n_episode_test, n_step_per_episode, n_D2D, n_neighbor])
    rate_rand = np.zeros([n_episode_test, n_step_per_episode, n_D2D, n_neighbor])
    
    action_all_testing_max = np.zeros([n_D2D, n_neighbor, 2], dtype='int32')
    for idx_episode in range(n_episode_test):
        print('----- Episode', idx_episode, '-----')

        env.random_RIS_implement()
        env.renew_neighbor()
        env.renew_channels_fastfading()

        Sum_rate_per_episode_dqn = []
        Sum_rate_per_episode_ddqn = []
        Sum_rate_per_episode_rand = []
        Sum_rate_per_episode_max = []

        for test_step in range(n_step_per_episode):
            env.renew_neighbor()
            env.renew_channels_fastfading()
            # DQN models
            action_all_testing_dqn = np.zeros([n_D2D, n_neighbor, 2], dtype='int32')
            for i in range(n_D2D):
                for j in range(n_neighbor):
                    state_old = get_state(env, [i, j], 1, epsi_final)
                    action_dqn = predict(sesses_dqn[i*n_neighbor+j], state_old, epsi_final, True)
                    action_all_testing_dqn[i, j, 0] = action_dqn % n_RB  # chosen RB
                    action_all_testing_dqn[i, j, 1] = int(np.floor(action_dqn / n_RB))  # power level

            action_temp_dqn = action_all_testing_dqn.copy()
            C_rate_dqn, D_rate_dqn, reward_dqn = env.act_for_testing(action_temp_dqn)
            Sum_rate_per_episode_dqn.append(np.sum(C_rate_dqn) + np.sum(D_rate_dqn))  # sum rate in bps

            rate_dqn[idx_episode, test_step,:,:] = D_rate_dqn

        
            # DDQN models
            action_all_testing_ddqn = np.zeros([n_D2D, n_neighbor, 2], dtype='int32')
            for i in range(n_D2D):
                for j in range(n_neighbor):
                    state_old = get_state(env, [i, j], 1, epsi_final)
                    action_ddqn = predict(sesses_ddqn[i*n_neighbor+j], state_old, epsi_final, True)
                    action_all_testing_ddqn[i, j, 0] = action_ddqn % n_RB  # chosen RB
                    action_all_testing_ddqn[i, j, 1] = int(np.floor(action_ddqn / n_RB))  # power level

            action_temp_ddqn = action_all_testing_ddqn.copy()
            C_rate_ddqn, D_rate_ddqn, reward_ddqn = env.act_for_testing(action_temp_ddqn)
            Sum_rate_per_episode_ddqn.append(np.sum(C_rate_ddqn) + np.sum(D_rate_ddqn))  # sum rate in bps

            rate_ddqn[idx_episode, test_step,:,:] = D_rate_ddqn

            # random baseline
            action_rand = np.zeros([n_D2D, n_neighbor, 2], dtype='int32')
            action_rand[:, :, 0] = np.random.randint(0, n_RB, [n_D2D, n_neighbor]) # band
            action_rand[:, :, 1] = np.random.randint(0, len(env.D_power_dB_List), [n_D2D, n_neighbor]) # power
            action_temp_rand = action_rand.copy()
            C_rate_rand, D_rate_rand, reward_rand = env.act_for_testing(action_temp_rand)
            Sum_rate_per_episode_rand.append(np.sum(C_rate_rand) + np.sum(D_rate_rand))  # sum rate in bps

            rate_rand[idx_episode, test_step, :, :] = D_rate_rand

            # The following applies to n_D2D = 4 only
            action_all_testing_max = np.zeros([n_D2D, n_neighbor, 2], dtype='int32')
            n_power_level = len(env.D_power_dB_List)
            store_action = np.zeros([(n_RB*n_power_level)**n_D2D, 4])
            reward_all_max = []
            t = 0
            episode_exhaustive = 1
            for i in range(n_RB*len(env.D_power_dB_List)):
                print('Episode for exhaustive search', episode_exhaustive)
                episode_exhaustive = episode_exhaustive+1
                for j in range(n_RB*len(env.D_power_dB_List)):
                    for m in range(n_RB*len(env.D_power_dB_List)):
                        for n in range(n_RB*len(env.D_power_dB_List)):
                            action_all_testing_max[0, 0, 0] = i % n_RB
                            action_all_testing_max[0, 0, 1] = int(np.floor(i / n_RB))  # power level

                            action_all_testing_max[1, 0, 0] = j % n_RB
                            action_all_testing_max[1, 0, 1] = int(np.floor(j / n_RB))  # power level

                            action_all_testing_max[2, 0, 0] = m % n_RB
                            action_all_testing_max[2, 0, 1] = int(np.floor(m / n_RB))  # power level

                            action_all_testing_max[3, 0, 0] = n % n_RB
                            action_all_testing_max[3, 0, 1] = int(np.floor(n / n_RB))  # power level

                            action_temp_findMax = action_all_testing_max.copy()
                            reward_findMax = env.act_for_training(action_temp_findMax)
                            reward_all_max.append(reward_findMax)

                            store_action[t, :] = [i,j,m,n]
                            t += 1

            i = store_action[np.argmax(reward_all_max), 0]
            j = store_action[np.argmax(reward_all_max), 1]
            m = store_action[np.argmax(reward_all_max), 2]
            n = store_action[np.argmax(reward_all_max), 3]

            action_testing_max = np.zeros([n_D2D, n_neighbor, 2], dtype='int32')

            action_testing_max[0, 0, 0] = i % n_RB
            action_testing_max[0, 0, 1] = int(np.floor(i / n_RB))  # power level

            action_testing_max[1, 0, 0] = j % n_RB
            action_testing_max[1, 0, 1] = int(np.floor(j / n_RB))  # power level

            action_testing_max[2, 0, 0] = m % n_RB
            action_testing_max[2, 0, 1] = int(np.floor(m / n_RB))  # power level

            action_testing_max[3, 0, 0] = n % n_RB
            action_testing_max[3, 0, 1] = int(np.floor(n / n_RB))  # power level


            action_temp_max = action_testing_max.copy()
            C_rate_max, D_rate_max, reward_max = env.act_for_testing(action_temp_max)
            Sum_rate_per_episode_max.append(np.sum(C_rate_max) + np.sum(D_rate_max))  # sum V2I rate in bps

            # update the environment and compute interference
            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp_dqn)
            env.Compute_Interference(action_temp_ddqn)
            env.Compute_Interference(action_temp_max)


        Sum_rate_list_dqn.append(np.mean(Sum_rate_per_episode_dqn))
        Sum_rate_list_ddqn.append(np.mean(Sum_rate_per_episode_ddqn))
        Sum_rate_list_rand.append(np.mean(Sum_rate_per_episode_rand))
        Sum_rate_list_max.append(np.mean(Sum_rate_per_episode_max))

        print('DQN', round(np.average(Sum_rate_per_episode_dqn), 2), 'DDQN', round(np.average(Sum_rate_per_episode_ddqn), 2), 'rand', round(np.average(Sum_rate_per_episode_rand), 2), 'max', round(np.average(Sum_rate_per_episode_max), 2))

    print('-------- DQN -------------')
    print('n_D2D:', n_D2D, ', n_neighbor:', n_neighbor)
    print('Average sum rate:', round(np.average(Sum_rate_list_dqn), 2), 'Mbps')

    print('-------- DDQN -------------')
    print('n_D2D:', n_D2D, ', n_neighbor:', n_neighbor)
    print('Average sum rate:', round(np.average(Sum_rate_list_ddqn), 2), 'Mbps')

    print('-------- random -------------')
    print('n_D2D:', n_D2D, ', n_neighbor:', n_neighbor)
    print('Average sum rate:', round(np.average(Sum_rate_list_rand), 2), 'Mbps')

    print('-------- max -------------')
    print('n_D2D:', n_D2D, ', n_neighbor:', n_neighbor)
    print('Average sum rate:', round(np.average(Sum_rate_list_max), 2), 'Mbps')

    np.savetxt('Test_rate_DQN', Sum_rate_list_dqn)
    np.savetxt('Test_rate_DDQN', Sum_rate_list_ddqn)
    np.savetxt('Test_rate_random', Sum_rate_list_rand)
    np.savetxt('Test_rate_max', Sum_rate_list_max)
    
    plt.figure(1)
    plt.grid()
    plt.plot(Sum_rate_list_max, label = 'Upperbound')
    plt.plot(Sum_rate_list_ddqn, label = 'DDQN')
    plt.plot(Sum_rate_list_dqn, label = 'DQN')
    plt.plot(Sum_rate_list_rand, label = 'Rand')
    plt.ylabel('Rate')
    plt.xlabel('Testing epoch')
    plt.legend()
    plt.show()

    with open("Data.txt", "a") as f:
        f.write('-------- DQN, ' + label_DQN + '------\n')
        f.write('n_D2D: ' + str(n_D2D) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum rate: ' + str(round(np.average(Sum_rate_list_dqn), 5)) + ' Mbps\n')
        f.write('-------- DDQN, ' + label_DDQN + '------\n')
        f.write('n_D2D: ' + str(n_D2D) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum rate: ' + str(round(np.average(Sum_rate_list_ddqn), 5)) + ' Mbps\n')
        f.write('--------random ------------\n')
        f.write('n_D2D: ' + str(n_D2D) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum rate: ' + str(round(np.average(Sum_rate_list_rand), 5)) + ' Mbps\n')
        f.write('--------max ------------\n')
        f.write('n_D2D: ' + str(n_D2D) + ', n_neighbor: ' + str(n_neighbor) + '\n')
        f.write('Sum rate: ' + str(round(np.average(Sum_rate_list_max), 5)) + ' Mbps\n')

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dqn_path = os.path.join(current_dir, "model/" + label_DQN + '/rate_dqn.mat')
    scipy.io.savemat(dqn_path, {'rate_dqn': rate_dqn})
    ddqn_path = os.path.join(current_dir, "model/" + label_DDQN + '/rate_ddqn.mat')
    scipy.io.savemat(ddqn_path, {'rate_ddqn': rate_ddqn})
    rand_path = os.path.join(current_dir, "model/" + '/rate_rand.mat')
    scipy.io.savemat(rand_path, {'rate_rand': rate_rand})

# close sessions
for sess in sesses_dqn:
    sess.close()

for sess in sesses_ddqn:
    sess.close()

