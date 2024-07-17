#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:01:49 2019

@author: zj303
"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)
my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth = True

class DeepQNetwork:
    def __init__(
            self,
            n_positions,
            n_features,
            Is_double_q,
            learning_rate,
            reward_decay,
            replace_target_iter,
            memory_size,
            batch_size,
            output_graph=False,
            n_elements = 4
    ):
        self.n_positions = n_positions  # number of location choice for RIS
        self.n_features = n_features # length of the state
        self.n_actions = np.power(3, n_elements) * self.n_positions # number of actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter # target network update period
        self.memory_size = memory_size # total memory size of the replay memory
        self.batch_size = batch_size # batch sample size
        self.double_q = Is_double_q
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        # memory: current_state (n_features) + action (1) + reward (1) + next_state (n_features) = n_features * 2 + 2
        self._build_net_hidden()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session(config=my_config)

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []


    def _build_net_hidden(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        # -----------------------------------------------------------
        n_hidden_1 = 256
        n_hidden_2 = 128
        n_hidden_3 = 256
        n_input = self.n_features
        n_output = self.n_actions

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):

            w_1 = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
            w_2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
            w_3 = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
            w_4 = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

            b_1 = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
            b_2 = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
            b_3 = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
            b_4 = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

            layer_1 = tf.nn.relu(tf.add(tf.matmul(self.s, w_1), b_1))
            layer_1_b = tf.layers.batch_normalization(layer_1)
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1_b, w_2), b_2))
            layer_2_b = tf.layers.batch_normalization(layer_2)
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2_b, w_3), b_3))
            layer_3_b = tf.layers.batch_normalization(layer_3)

            self.q_eval = tf.nn.relu(tf.add(tf.matmul(layer_3_b, w_4), b_4)) # y
            self.g_q_action = tf.argmax(self.q_eval, axis=1)

            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval, name='TD_error'))

            # self.optim = tf.train.RMSPropOptimizer(learning_rate=self.lr, momentum=0.95, epsilon=0.01).minimize(self.loss)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss) # Adam optimizer
        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):

            w_1_p = tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1))
            w_2_p = tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1))
            w_3_p = tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1))
            w_4_p = tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1))

            b_1_p = tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1))
            b_2_p = tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1))
            b_3_p = tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1))
            b_4_p = tf.Variable(tf.truncated_normal([n_output], stddev=0.1))

            layer_1_p = tf.nn.relu(tf.add(tf.matmul(self.s_, w_1_p), b_1_p))
            layer_1_p_b = tf.layers.batch_normalization(layer_1_p)

            layer_2_p = tf.nn.relu(tf.add(tf.matmul(layer_1_p_b, w_2_p), b_2_p))
            layer_2_p_b = tf.layers.batch_normalization(layer_2_p)

            layer_3_p = tf.nn.relu(tf.add(tf.matmul(layer_2_p_b, w_3_p), b_3_p))
            layer_3_p_b = tf.layers.batch_normalization(layer_3_p)

            self.q_next = tf.nn.relu(tf.add(tf.matmul(layer_3_p_b, w_4_p), b_4_p)) # y_p
            self.g_target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')
            self.target_q_with_idx = tf.gather_nd(self.q_next, self.g_target_q_idx)


        
        
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, epsilon):
        # Unify the shape of observation (1, size_of_observation)
        observation = observation[np.newaxis, :]
        if np.random.rand() > epsilon:
            # forward feed the observation and get q value for every actions
            action = self.sess.run(self.g_q_action, feed_dict={self.s: observation})[0]
            flag = 1
        else:
            action = np.random.randint(0, self.n_actions)
            flag = 0
        return action, flag

    def learn(self):
        Is_replacement = False
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')
            print('learning rate:', self.lr)
            Is_replacement = True

        """sample batch memory from all memory"""
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        """trainning the DDQN using batch memory"""
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, self.n_features + 2: 2 * self.n_features + 2],    # next observation
                       self.s: batch_memory[:, self.n_features + 2: 2 * self.n_features + 2]})    # current observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        batch_reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            pred_action = self.sess.run(self.g_q_action, feed_dict={self.s: batch_memory[:, self.n_features + 2: 2 * self.n_features + 2]})
            q_t_plus_1 = self.sess.run(self.target_q_with_idx, {self.s_: batch_memory[:, self.n_features + 2: 2 * self.n_features + 2],
                                                              self.g_target_q_idx: [[idx, pred_a] for idx, pred_a in
                                                                               enumerate(pred_action)]})
            # max_act4next = np.argmax(q_eval4next, axis=1)        # choose the action evaluated by q_eval network
            # selected_q_next = q_next[batch_index, max_act4next]  # Double DQN calculate q_next according to the action chosen by q_eval
            selected_q_next = q_t_plus_1
        else:       # Natural DQN
            selected_q_next = np.max(q_next, axis=1)    # May lead to overestimate issue

        q_target[batch_index, eval_act_index] = batch_reward + self.gamma * selected_q_next
        
        # Training eval_net
        _, cost = self.sess.run(
            [self.optim, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.q_target: q_target
            })
        if Is_replacement:
            self.cost_his.append(cost)
        self.learn_step_counter += 1
        return cost, Is_replacement

    # def plot_cost(self):
    #     import matplotlib.pyplot as plt
    #     plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    #     plt.ylabel('Cost')
    #     plt.xlabel('training steps')
    #     plt.show()


# if __name__ == '__main__':
#     DQN = DeepQNetwork(3, 4, output_graph=False)
