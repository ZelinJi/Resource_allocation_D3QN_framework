from __future__ import division, print_function
import random
import scipy
import scipy.io
import numpy as np
import os
from RIS_env import Training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

from replay_memory import ReplayMemory
import sys


my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth=True

def get_state(env, idx=(0, 0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """
    C_fast = (env.C_channels_with_fastfading[idx[0], :] - env.C_channels_abs[idx[0]] + 10) / 35

    D_fast = (env.D_channels_with_fastfading[:, env.D2D_users[idx[0]].destinations[idx[1]], :] - env.D_channels_abs[
                                                                                                 :, env.D2D_users[idx[0]].destinations[idx[1]]] + 10) / 35
    D_interference = (-env.D2D_Interference_all[idx[0], idx[1], :] - 60) / 60

    C_abs = (env.C_channels_abs[idx[0]] - 80) / 60.0
    D_abs = (env.D_channels_abs[:, env.D2D_users[idx[0]].destinations[idx[1]]] - 80) / 60.0
    return np.concatenate((np.reshape(C_fast, -1), np.reshape(D_fast, -1), D_interference, np.reshape(C_abs, -1),
                           D_abs, np.asarray([ind_episode, epsi])))


class Agent(object):
    def __init__(self, memory_entry_size):
        self.discount = 1
        self.double_q = True
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)
        
#settings

class Model:
    def __init__(self, BS_center, cuser_center, duser_center, p_k_dB, p_t_dB_List, label):
        self.epsi_final = 0.02
        self.BS_center = BS_center
        self.cuser_center = cuser_center
        self.duser_center = duser_center
        self.p_k_dB = p_k_dB
        self.p_t_dB_List = p_t_dB_List
        self.n_D2D = len(duser_center)
        self.n_RB = len(cuser_center)
        self.n_neighbor = 1
        self.label = label
        self.environment = Training(self.BS_center, self.cuser_center, self.duser_center, p_k_dB, p_t_dB_List)

    def build_net(self):
        n_hidden_1 = 500
        n_hidden_2 = 250
        n_hidden_3 = 120
        n_input = 31 #len(get_state(self.environment))
        n_output = 36 #n_RB * len(env.D_power_dB_List)

        self.g = tf.Graph()
        with self.g.as_default():
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
            optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(g_loss)

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
            return x, g_q_action, saver, init

    def predict(self, sess, s_t, x, g_q_action):
        pred_action = sess.run(g_q_action, feed_dict={x: [s_t]})[0]
        return pred_action

    def load_models(self, sess, model_path, saver):
        """ Restore models from the current directory with the name filename """
        dir_ = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_, "model/" + model_path)
        saver.restore(sess, model_path)


    def load_saved_models(self):
        """ Restore all models """
        self.x, self.g_q_action, saver, init = self.build_net()
        # --------------------------------------------------------------
        agents = []
        self.sesses = []
        for ind_agent in range(4):  # initialize agents
            print("Initializing agent", ind_agent)
            agent = Agent(memory_entry_size = 31)#len(get_state(self.environment)))
            agents.append(agent)
            sess = tf.Session(graph=self.g,config=my_config)
            sess.run(init)
            self.sesses.append(sess)
        print("\nRestoring the model...")
        
        for i in range(self.n_D2D):
            for j in range(self.n_neighbor):
                model_path = self.label + '/agent_' + str(i *self. n_neighbor + j)
                self.load_models(self.sesses[i * self.n_neighbor + j], model_path, saver)

    def get_resource_allocation(self, observation, BS_center, cuser_center, duser_center, p_k_dB, p_t_dB_List, action, n_elements):
        """Aquire the resource allocation policy"""
        train = Training(self.BS_center, self.cuser_center, self.duser_center, self.p_k_dB, self.p_t_dB_List)
        action_temp = action
        next_coords = [observation[0] * (100), observation[1] * (100)]


        RIS_next_coords, theta, next_theta_number = train.get_next_state(observation, action_temp, next_coords)
        self.environment.overall_channel(RIS_next_coords, theta, n_elements)
        self.environment.renew_channels_fastfading()

        action_all_testing = np.zeros([len(duser_center), 1, 2], dtype='int32')
        resource_allocation_action = np.zeros(len(duser_center))
        n_resource_allocation_action = (self.n_RB * len(self.p_t_dB_List))

        for i in range(len(self.duser_center)):
            for j in range(self.n_neighbor):
                state_old = get_state(self.environment, [i, j], 1, self.epsi_final)
                resource_allocation_action[i] = self.predict(self.sesses[i*self.n_neighbor+j], state_old, self.x, self.g_q_action)
                action_all_testing[i, j, 0] = resource_allocation_action[i] % self.n_RB  # chosen RB
                action_all_testing[i, j, 1] = int(np.floor(resource_allocation_action[i] / self.n_RB))  # power level

        return resource_allocation_action