from __future__ import division, print_function
import scipy.io
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import Environment_marl
import os
from replay_memory import ReplayMemory
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth = True
IS_double_q = True
sigma = 0.0001


class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = 1
        self.double_q = IS_double_q
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)


# ################## SETTINGS ######################

width = 100
height = 100

duser_center = np.array([[55,5], [50,20], [10,80], [20,90]])
cuser_center = np.array([[25,30], [95,65], [50,85], [75,30]])
BS_center = [50, 50]

IS_TRAIN = True


if IS_double_q:
    label = 'marl_model_double'
else:
    label = 'marl_model'

n_veh = 4
n_neighbor = 1
n_RB = n_veh

C_power_dB = 23  # dBm
D_power_dB_List = [24, 21, 18, 15, 12, 9, 6, 3, 0]  # the power levels

env = Environment_marl.Environ(BS_center, cuser_center, duser_center, C_power_dB, D_power_dB_List) # initialize parameters in env
# env.new_random_game()

n_episode = 5000
n_step_per_episode = int(env.time_slow/env.time_fast)
epsi_final = 0.02
epsi_anneal_length = int(0.9*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode * 10

n_episode_test = 100  # test episodes

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
    y = tf.nn.relu(tf.add(tf.matmul(layer_3_b, w_4), b_4))
    g_q_action = tf.argmax(y, axis=1)

    # compute loss
    g_target_q_t = tf.placeholder(tf.float32, None, name="target_value")

    g_action = tf.placeholder(tf.int32, None, name='g_action')
    action_one_hot = tf.one_hot(g_action, n_output, 1.0, 0.0, name='action_one_hot')
    q_acted = tf.reduce_sum(y * action_one_hot, reduction_indices=1, name='q_acted')

    g_loss = tf.reduce_mean(tf.square(g_target_q_t - q_acted), name='g_loss')
    # to update
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
    """ Determine the action """
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
agents = []
sesses = []
for ind_agent in range(n_veh * n_neighbor):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)

    sess = tf.Session(graph=g,config=my_config)
    sess.run(init)
    sesses.append(sess)



# ------------------------- Training -----------------------------
record_reward = np.zeros([n_episode*n_step_per_episode, 1])
record_loss = []

if IS_TRAIN:
    average_return = []
    average_rate = []
    for i_episode in range(n_episode):
        print("-------------------------")
        if i_episode < epsi_anneal_length:
            epsi = 1 - (i_episode * (1 - epsi_final)) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
        else:
            epsi = 1 - (epsi_anneal_length * (1 - epsi_final)) / (epsi_anneal_length - 1)
        print('Episode:', i_episode, 'Epsi', epsi)
        if i_episode%100 == 0:
            env.renew_neighbor()
            env.random_RIS_implement()
            env.overall_channel() # update channel slow fading
            env.renew_channels_fastfading() # update channel fast fading

        cumulative_reward = 0
        cumulative_rate = np.zeros([n_veh, 1])
        step_rate = []
        for i_step in range(n_step_per_episode):
            time_step = i_episode*n_step_per_episode + i_step
            state_old_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
            for i in range(n_veh):
                for j in range(n_neighbor):
                    state = get_state(env, [i, j], i_episode/(n_episode-1), epsi)
                    state_old_all.append(state)
                    action = predict(sesses[i*n_neighbor+j], state, epsi)
                    action_all.append(action)

                    action_all_training[i, j, 0] = action % n_RB  # chosen RB
                    action_all_training[i, j, 1] = int(np.floor(action / n_RB)) # power level

            # All agents take actions simultaneously, obtain shared reward, and update the environment.
            action_temp = action_all_training.copy()
            train_reward = env.act_for_training(action_temp)
            C_rate, D_rate, _ = env.act_for_testing(action_temp)
            V2V_success = 1
            record_reward[time_step] = train_reward
            
            cumulative_reward += train_reward
            average_reward = cumulative_reward / (i_step+1)
            step_rate.append(sum(C_rate)+sum(D_rate))
            cumulative_rate = sum(step_rate) / (i_step+1)

            env.random_RIS_implement()
            env.overall_channel()  # update channel slow fading
            env.renew_channels_fastfading()  # update channel fast fading
            env.Compute_Interference(action_temp)

            for i in range(n_veh):
                for j in range(n_neighbor):
                    state_old = state_old_all[n_neighbor * i + j]
                    action = action_all[n_neighbor * i + j]
                    state_new = get_state(env, [i, j], i_episode/(n_episode-1), epsi)
                    agents[i * n_neighbor + j].memory.add(state_old, state_new, train_reward, action)  # add entry to this agent's memory

                    # training this agent
                    if time_step % mini_batch_step == mini_batch_step-1:
                        loss_val_batch = q_learning_mini_batch(agents[i*n_neighbor+j], sesses[i*n_neighbor+j])
                        record_loss.append(loss_val_batch)
                        if i == 0 and j == 0:
                            print('step:', time_step, 'agent',i*n_neighbor+j, 'loss', sum(record_loss[-4:]))
                    if time_step % target_update_step == target_update_step-1:
                        update_target_q_network(sesses[i*n_neighbor+j])
                        if i == 0 and j == 0:
                            print('Update target Q network...')
                            print('reward', average_reward)
                            average_return.append(average_reward)
                            average_rate.append(cumulative_rate)
    print('Training Done. Saving models...')
    for i in range(n_veh):
        for j in range(n_neighbor):
            model_path = label + '/agent_' + str(i * n_neighbor + j)
            save_models(sesses[i * n_neighbor + j], model_path)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    reward_path = os.path.join(current_dir, "model/" + label + '/reward.mat')
    scipy.io.savemat(reward_path, {'reward': record_reward})


    record_loss = np.asarray(record_loss).reshape((-1, n_veh*n_neighbor))
    loss_path = os.path.join(current_dir, "model/" + label + '/train_loss.mat')
    scipy.io.savemat(loss_path, {'train_loss': record_loss})


# close sessions
for sess in sesses:
    sess.close()
