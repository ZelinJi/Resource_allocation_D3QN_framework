from __future__ import division
import numpy as np
import time
import random
import math


np.random.seed(1234)

class D2D_user:
    """D2D simulator: include all the information for a D2D user"""

    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []

class Environ:
    def __init__(self, BS_position, cuser_center, duser_center, C_power_dB, D_power_dB_List):
        self.n_positions = 16  # number of positions
        self.N = 8  # number of RIS elements

        self.BS_position = BS_position
        self.cuser_center = cuser_center
        self.duser_center = duser_center
        self.C_power_dB = C_power_dB  # dBm
        self.D_power_dB_List = D_power_dB_List

        self.h_bs = 25
        self.h_ms = 1.5
        self.sig2_dB = -114
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.fc = 2

        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/vehicle position every 100 ms

        self.n_D2D = len(duser_center)
        self.n_RB = len(cuser_center)
        self.n_neighbor = 1
        self.D2D_users = []
        self.add_D2Ds(self.n_D2D)
        self.renew_neighbor()

        self.RIS_position = np.zeros(2)
        self.theta = np.zeros(self.N, dtype=complex)
        self.random_RIS_implement()

        self.overall_channel()
        self.renew_channels_fastfading()
        self.active_links = np.ones((self.n_D2D, 1), dtype='bool')

        self.D2D_Interference_all = np.zeros((self.n_D2D, self.n_neighbor, self.n_RB)) + self.sig2

    def add_new_D2D(self, start_position, start_direction, start_velocity):
        self.D2D_users.append(D2D_user(start_position, start_direction, start_velocity)) #add D2D users

    def add_D2Ds(self, N):
        for j in range(N):
            self.add_new_D2D(self.duser_center[j], 'd', 0)

    def renew_neighbor(self):
        """ Determine the neighbors of each D2D """

        for i in range(len(self.D2D_users)):
            self.D2D_users[i].neighbors = []
            self.D2D_users[i].actions = []
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.D2D_users]])
        Distance = abs(z.T - z)

        for i in range(len(self.D2D_users)):
            sort_idx = np.argsort(Distance[:, i])
            for j in range(1):
                self.D2D_users[i].neighbors.append(sort_idx[j + 1])
            destination = self.D2D_users[i].neighbors

            self.D2D_users[i].destinations = destination

    def get_D_path_loss(self, position_A, position_B):
        """Calculate pathloss between D2D pairs"""
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        distance = math.hypot(d1, d2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(
                        self.h_ms) + 2.7 * np.log10(self.fc / 5)
        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        PL = 32.4 + 35.3 * np.log10(distance) + 21.3 * np.log10(self.fc)
        return PL  # + self.shadow_std * np.random.normal()

    def get_D_RIS_path_loss(self, position_A, position_B, position_RIS, theta):
        """Calculate RIS pathloss between D2D pairs"""
        theta_diag = np.diag(theta)
        ds = 0.02
        dA1 = abs(position_A[0] - position_RIS[0])
        dA2 = abs(position_A[1] - position_RIS[1])
        dB1 = abs(position_B[0] - position_RIS[0])
        dB2 = abs(position_B[1] - position_RIS[1])
        dA = math.hypot(dA1, dA2) + 0.001
        dB = math.hypot(dB1, dB2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)
        a_aoa = np.zeros(self.N, dtype=complex)
        a_aod = np.zeros(self.N, dtype=complex)
        theta_aoa = np.arctan((position_A[1] - position_RIS[1])/(position_A[0] - position_RIS[0]))
        theta_aod = np.arctan((position_B[1] - position_RIS[1])/(position_B[0] - position_RIS[0]))

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(
                        self.h_ms) + 2.7 * np.log10(self.fc / 5)

        PLA = 24 + 20 * np.log10(dA) + 20 * np.log10(self.fc / 5)
        PLB = 24 + 20 * np.log10(dB) + 20 * np.log10(self.fc / 5)

        for n in range (self.N):
            a_aod[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aod))
            a_aoa[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aoa))

        ChannelA = 1/np.power(10, PLA/10) * np.exp(-1.0j*(2*np.pi)*dA*(self.fc/0.3))*a_aoa
        ChannelB = 1/np.power(10, PLB/10) * np.exp(-1.0j*(2*np.pi)*dB*(self.fc/0.3))*a_aod.conj().T

        PL_RIS_sig = np.dot(np.dot(ChannelA,  theta_diag), ChannelB)
        PL_RIS = np.log10((1 / (PL_RIS_sig))) * 10

        return PL_RIS  # + self.shadow_std * np.random.normal()

    def get_C_path_loss(self, position_A):
        """Calculate pathloss between cellular users and BS"""
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2)
        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(
                        self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        PL = 32.4 + 35.3 * np.log10(distance) + 21.3 * np.log10(self.fc)
        return PL

    def get_C_RIS_path_loss(self, position_A, position_RIS, theta):
        """Calculate RIS pathloss between cellular users and BS"""
        theta_diag = np.diag(theta)
        ds = 0.02 # The spacing between elements
        dA1 = abs(position_A[0] - position_RIS[0])
        dA2 = abs(position_A[1] - position_RIS[1])
        dB1 = abs(self.BS_position[0] - position_RIS[0])
        dB2 = abs(self.BS_position[1] - position_RIS[1])
        dA = math.hypot(dA1, dA2)
        dB = math.hypot(dB1, dB2)
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)
        a_aoa = np.zeros(self.N, dtype=complex)
        a_aod = np.zeros(self.N, dtype=complex)
        theta_aoa = np.arctan((position_A[1] - position_RIS[1])/(position_A[0] - position_RIS[0]))
        theta_aod = np.arctan((self.BS_position[1] - position_RIS[1])/(self.BS_position[0] - position_RIS[0]))

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(
                        self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        PLA = 24 + 20 * np.log10(dA) + 20 * np.log10(self.fc / 5)
        PLB = 24 + 20 * np.log10(dB) + 20 * np.log10(self.fc / 5)

        for n in range (self.N):
            a_aod[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aod))
            a_aoa[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aoa))

        ChannelA = 1/np.power(10, PLA/10) * np.exp(-1.0j*(2*np.pi)*dA*(self.fc/0.3))*a_aoa
        ChannelB = 1/np.power(10, PLB/10) * np.exp(-1.0j*(2*np.pi)*dB*(self.fc/0.3))*a_aod.conj().T

        PL_RIS_sig = np.dot(np.dot(ChannelA,  theta_diag), ChannelB)
        PL_RIS = np.log10((1 / (PL_RIS_sig))) * 10
        return PL_RIS  # + self.shadow_std * np.random.normal()

    def random_RIS_implement(self):
        """RIS are randomly deployed during the training of resource allocation model"""
        position_number = np.random.normal(self.n_positions/2, int(np.sqrt(self.n_positions)), 1)
        self.RIS_position [0] = 12.5 + 25*(int(position_number)%int(np.sqrt(self.n_positions)))
        self.RIS_position [1] = 12.5 + 25*(np.floor(int(position_number)/int(np.sqrt(self.n_positions))))
        phase_number = np.random.randint(0, 8, self.N)
        for n in range (self.N):
            self.theta[n] = np.exp(1.0j*(np.pi/4)*phase_number[n])

    def overall_channel(self):
        """ The combined channel"""
        self.D_pathloss = np.zeros((4, 4)) + 50 * np.identity(4)
        self.C_pathloss = np.zeros((4))
        self.D_RIS_pathloss = np.zeros((4, 4), dtype=complex)
        self.C_RIS_pathloss = np.zeros((4), dtype=complex)

        self.D_channels_abs = np.zeros((4, 4))
        self.C_channels_abs = np.zeros((4))


        self.D_to_BS_pathloss = np.zeros((4)) # D2D to BS interference channel
        self.C_to_D_pathloss = np.zeros((4, 4)) # Cellular user to D2D user interference channel
        self.D_to_BS_RIS_pathloss = np.zeros((4), dtype=complex)
        self.C_to_D_RIS_pathloss = np.zeros((4, 4), dtype=complex)

        self.C_to_D_channels_abs = np.zeros((4, 4))
        self.D_to_BS_channels_abs = np.zeros((4))


        for i in range(4):
            for j in range(i + 1, 4):
                self.D_pathloss[j, i] = self.D_pathloss[i][j] = self.get_D_path_loss(self.duser_center[i],
                                                                                     self.duser_center[j])
                self.D_RIS_pathloss[j, i] = self.D_RIS_pathloss[i][j] = self.get_D_RIS_path_loss(self.duser_center[i],
                                                                                                 self.duser_center[j],
                                                                                                 self.RIS_position,
                                                                                                 self.theta)
        self.D_overall = 1 / np.abs(1 / np.power(10, self.D_pathloss / 10) + 1 / np.power(10, self.D_RIS_pathloss/10))
        self.D_channels_abs = 10 * np.log10(self.D_overall)

        for i in range(4):
            self.C_pathloss[i] = self.get_C_path_loss(self.cuser_center[i])
            self.C_RIS_pathloss[i] = self.get_C_RIS_path_loss(self.cuser_center[i], self.RIS_position, self.theta)
        self.C_overall = 1 / np.abs(1 / np.power(10, self.C_pathloss / 10) + 1 / np.power(10, self.C_RIS_pathloss/10))
        self.C_channels_abs = 10 * np.log10(self.C_overall)

        for i in range (4):
            self.D_to_BS_pathloss[i] = self.get_C_path_loss(self.duser_center[i])
            self.D_to_BS_RIS_pathloss[i] = self.get_C_RIS_path_loss(self.duser_center[i], self.RIS_position, self.theta)
            for j in range (4):
                self.C_to_D_pathloss[i, j] = self.get_D_path_loss(self.cuser_center[i], self.duser_center[j]) # i-th cellular user to j-th D2D user
                self.C_to_D_RIS_pathloss[i, j] = self.get_D_RIS_path_loss(self.cuser_center[i], self.duser_center[j], self.RIS_position, self.theta)
        self.C_to_D_overall = 1 / np.abs(1 / np.power(10, self.C_to_D_pathloss / 10) + 1 / np.power(10, self.C_to_D_RIS_pathloss))
        self.C_to_D_channels_abs = 10 * np.log10(self.C_to_D_overall)

        self.D_to_BS_overall = 1 / np.abs(1 / np.power(10, self.D_to_BS_pathloss / 10) + 1 / np.power(10, self.D_to_BS_RIS_pathloss))
        self.D_to_BS_channels_abs = 10 * np.log10(self.D_to_BS_overall)


    def renew_channels_fastfading(self):
        """ Renew fast fading combined channel """
        D_pathloss_with_fastfading = np.repeat(self.D_pathloss[:, :, np.newaxis], self.n_RB, axis=2)
        self.D_pathloss_with_fastfading = D_pathloss_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 0.5, D_pathloss_with_fastfading.shape) + 1j * np.random.normal(0, 0.5,
                                                                                                      D_pathloss_with_fastfading.shape)))
        D_RIS_pathloss = np.repeat(self.D_RIS_pathloss[:, :, np.newaxis], self.n_RB, axis=2)

        D_overall_with_fastfading = 1 / np.abs(1 / np.power(10, self.D_pathloss_with_fastfading / 10) + 1 / np.power(10, D_RIS_pathloss / 10))
        self.D_channels_with_fastfading = 10 * np.log10(D_overall_with_fastfading)

        C_pathloss_with_fastfading = np.repeat(self.C_pathloss[:, np.newaxis], self.n_RB, axis=1)
        self.C_pathloss_with_fastfading = C_pathloss_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 0.5, C_pathloss_with_fastfading.shape) + 1j * np.random.normal(0, 0.5,
                                                                                                      C_pathloss_with_fastfading.shape)))
        C_RIS_pathloss = np.repeat(self.C_RIS_pathloss[:, np.newaxis], self.n_RB, axis=1)
        C_overall_with_fastfading = 1 / np.abs(1 / np.power(10, self.C_pathloss_with_fastfading / 10) + 1 / np.power(10, C_RIS_pathloss / 10))
        self.C_channels_with_fastfading = 10 * np.log10(C_overall_with_fastfading)

    def compute_reward(self, action_all):
        """Calculate the data rate and the reward for reinforcement learning"""
        actions = action_all[:, :, 0]  # the channel_selection_part
        power_selection = action_all[:, :, 1]  # power selection

        coefficient = actions.copy()

        # ------------ Compute Cellular rate --------------------
        C_Rate = np.zeros(self.n_RB)
        C_Interference = np.zeros(self.n_RB)  # V2I interference
        real_power = np.zeros(self.n_RB)
        for index in range(self.n_RB):
            if (index % 2 == 0):
                real_power[index] = self.D_power_dB_List[power_selection[index, 0]]
            else:
                real_power[index] = float("-inf")
        for i in range(self.n_RB):
            for j in range (self.n_neighbor):
                C_Interference[coefficient[i][j]] += 10 ** ((real_power[i] - self.D_to_BS_channels_abs[i]) / 10)
        self.C_Interference = C_Interference + self.sig2
        C_Signals = 10 ** ((self.C_power_dB - self.C_channels_abs) / 10)
        C_Rate = np.log2(1 + np.divide(C_Signals, self.C_Interference))

        # ------------ Compute D2D rate -------------------------
        D_Interference = np.zeros((self.n_D2D, 1))
        D_Signal = np.zeros((self.n_D2D, 1))

        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(coefficient == i)  # find spectrum-sharing D2Ds
            if (len(indexes) != 0):
                for j in range (len(indexes)):
                    receiver_j = self.D2D_users[indexes[j, 0]].destinations[indexes[j, 1]]
                    D_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((real_power[indexes[j, 0]] -
                                                                     self.D_channels_with_fastfading[
                                                                         indexes[j][0], receiver_j, i]) / 10)
                    # Cellular links interference to D2D links
                    D_Interference[indexes[j, 0], indexes[j, 1]] = 10 ** (
                                (self.C_power_dB - self.C_to_D_channels_abs[i, receiver_j]) / 10)

                    #  D2D interference
                    for k in range(j + 1, len(indexes)):  # spectrum-sharing D2Ds
                        receiver_k = self.D2D_users[indexes[k][0]].destinations[indexes[k][1]]
                        D_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** (
                                    (real_power[indexes[k, 0]]
                                     - self.D_channels_with_fastfading[indexes[k][0]][receiver_j][i]) / 10)
                        D_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** (
                                    (real_power[indexes[j, 0]]
                                     - self.D_channels_with_fastfading[indexes[j][0]][receiver_k][i]) / 10)
        self.D_Interference = D_Interference + self.sig2
        D_Rate = np.log2(1 + np.divide(D_Signal, self.D_Interference))
        return C_Rate, D_Rate

    def Compute_Interference(self, actions):
        """Calculate Interference"""
        D2D_Interference = np.zeros((len(self.D2D_users), self.n_neighbor, self.n_RB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        channel_selection[np.logical_not(self.active_links)] = -1

        # interference from cellular links
        for i in range(self.n_RB):
            for k in range(len(self.D2D_users)):
                for m in range(len(channel_selection[k, :])):
                    D2D_Interference[k, m, i] += 10 ** ((self.C_power_dB - self.D_channels_with_fastfading[i][
                        self.D2D_users[k].destinations[m]][i]) / 10)

        # interference from peer D2D links
        for i in range(len(self.D2D_users)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.D2D_users)):
                    for m in range(len(channel_selection[k, :])):
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        D2D_Interference[k, m, channel_selection[i, j]] += 10 ** (
                                    (self.D_power_dB_List[power_selection[i, j]]
                                     - self.D_channels_with_fastfading[i][self.D2D_users[k].destinations[m]][
                                         channel_selection[i, j]]) / 10)
        self.D2D_Interference_all = 10 * np.log10(D2D_Interference)

    def act_for_training(self, actions):
        """Determine the real reward based on the SINR constraints"""
        done = True
        reward_C = 0
        reward_D = 0
        action_temp = actions.copy()
        C_Rate, D_Rate = self.compute_reward(action_temp)
        for i in range (len(C_Rate)):
            if (C_Rate[i] > 3.16):
                reward_C += C_Rate[i] / (self.n_RB * 10)
            else:
                done = False
        for i in range(len(D_Rate)):
            if (i % 2 == 0):
                if (D_Rate[i]> 2):
                    reward_D += D_Rate[i] / (self.n_RB * 10)
                else:
                    done = False
        if done:
            reward = reward_C + reward_D
        else:
            reward = 0
        return reward

    def act_for_testing(self, actions):
        """Determine the real reward based on the SINR constraints (Strict for testing)"""
        done = True
        reward_C = 0
        reward_D = 0
        action_temp = actions.copy()
        C_Rate, D_Rate = self.compute_reward(action_temp)
        for i in range (len(C_Rate)):
            if (C_Rate[i] < 3.16):
                done = False
            reward_C += C_Rate[i] / (self.n_RB * 10)
        for i in range(len(D_Rate)):
            if (i % 2 == 0):
                if (D_Rate[i] < 2):
                    done = False
                reward_D += D_Rate[i] / (self.n_RB * 10)
        if done:
            reward = reward_C + reward_D
        else:
            reward = 0
            for i in range(len(C_Rate)):
                C_Rate[i] = C_Rate[i] / 2
        return C_Rate, D_Rate, reward
