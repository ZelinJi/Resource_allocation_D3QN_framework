import numpy as np
# import tkinter as tk
import time
import math

width = 100
height = 100

n_phase_adjust_number = 8 # {0, 45, 90, 135, 180, 225, 270, 315}

x = 0
np.random.seed(1)

n_D2D = 4
n_neighbor = 1
n_RB = n_D2D

class D2D_user:
    # D2D simulator: include all the information for a D2D user

    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []
        
class Training:
    def __init__(self, BS_position, cuser_center, duser_center, C_power_dB, D_power_dB_List):
        self.BS_position = BS_position
        self.cuser_center = cuser_center
        self.duser_center = duser_center
        self.C_power_dB = C_power_dB
        self.D_power_dB_List = D_power_dB_List
        
        self.h_bs = 25
        self.h_ms = 1.5
        self.sig2_dB = -115
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.fc = 2
        
        self.n_D2D = len(duser_center)
        self.n_RB = self.n_D2D
        self.D2D_users = []
        self.add_D2Ds(self.n_D2D)
        self.renew_neighbor()
        self.D2D_Interference_all = np.zeros((self.n_D2D, 1, self.n_RB)) + self.sig2


    def add_new_D2D(self, start_position, start_direction, start_velocity):
        self.D2D_users.append(D2D_user(start_position, start_direction, start_velocity))
    
    def add_D2Ds(self, n_D2D):
        for j in range(n_D2D):
            self.add_new_D2D(self.duser_center[j],'d',0)

        
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

        #PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        PL = 32.4 + 35.3 * np.log10(distance) + 21.3 * np.log10(self.fc)
        return PL  # + self.shadow_std * np.random.normal()

    def get_D_RIS_path_loss(self, position_A, position_B, position_RIS, theta, n_elements_total):

        theta_all = np.zeros(n_elements_total, dtype=complex)
        a_aoa_all = np.zeros(n_elements_total, dtype=complex)
        a_aod_all = np.zeros(n_elements_total, dtype=complex)
        n_elements_per_row = len(theta)
        number_of_row = np.floor(n_elements_total / n_elements_per_row)

        ds = 0.02
        dA1 = abs(position_A[0] - position_RIS[0])
        dA2 = abs(position_A[1] - position_RIS[1])
        dB1 = abs(position_B[0] - position_RIS[0])
        dB2 = abs(position_B[1] - position_RIS[1])
        dA = math.hypot(dA1, dA2) + 0.001
        dB = math.hypot(dB1, dB2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)
        a_aoa = np.zeros(n_elements_per_row, dtype=complex)
        a_aod = np.zeros(n_elements_per_row, dtype=complex)
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

        # PLA = PL_Los(dA)
        # PLB = PL_Los(dB)
        PLA = 24 + 20 * np.log10(dA) + 20 * np.log10(self.fc / 5)
        PLB = 24 + 20 * np.log10(dB) + 20 * np.log10(self.fc / 5)

        for n in range (n_elements_per_row):
            a_aod[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aod))
            a_aoa[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aoa))

        for i in range (n_elements_total):
            index = i % n_elements_per_row
            theta_all [i] = theta[index]
            a_aoa_all [i] = a_aoa [index]
            a_aod_all[i] = a_aod[index]
        theta_diag = np.diag(theta_all)

        ChannelA = 1/np.power(10, PLA/10) * np.exp(-1.0j*(2*np.pi)*dA*(self.fc/0.3))*a_aoa_all
        ChannelB = 1/np.power(10, PLB/10) * np.exp(-1.0j*(2*np.pi)*dB*(self.fc/0.3))*a_aod_all.conj().T

        PL_RIS_sig = np.dot(np.dot(ChannelA,  theta_diag), ChannelB)
        PL_RIS = np.log10((1 / (PL_RIS_sig))) * 10

        return PL_RIS  # + self.shadow_std * np.random.normal()

    def get_C_path_loss(self, position_A):
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

        #PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        PL = 32.4 + 35.3 * np.log10(distance) + 21.3 * np.log10(self.fc)
        return PL

    def get_C_RIS_path_loss(self, position_A, position_RIS, theta, n_elements_total):
        theta_all = np.zeros(n_elements_total, dtype=complex)
        a_aoa_all = np.zeros(n_elements_total, dtype=complex)
        a_aod_all = np.zeros(n_elements_total, dtype=complex)
        n_elements_per_row = len(theta)
        number_of_row = np.floor(n_elements_total / n_elements_per_row)
        ds = 0.02 # The spacing between elements
        dA1 = abs(position_A[0] - position_RIS[0])
        dA2 = abs(position_A[1] - position_RIS[1])
        dB1 = abs(self.BS_position[0] - position_RIS[0])
        dB2 = abs(self.BS_position[1] - position_RIS[1])
        dA = math.hypot(dA1, dA2)
        dB = math.hypot(dB1, dB2)
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)
        a_aoa = np.zeros(n_elements_per_row, dtype=complex)
        a_aod = np.zeros(n_elements_per_row, dtype=complex)
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


        # PLA = PL_Los(dA)
        # PLB = PL_Los(dB)
        PLA = 24 + 20 * np.log10(dA) + 20 * np.log10(self.fc / 5)
        PLB = 24 + 20 * np.log10(dB) + 20 * np.log10(self.fc / 5)

        for n in range (n_elements_per_row):
            a_aod[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aod))
            a_aoa[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aoa))

        for i in range (n_elements_total):
            index = i % n_elements_per_row
            theta_all [i] = theta[index]
            a_aoa_all [i] = a_aoa [index]
            a_aod_all[i] = a_aod[index]
        theta_diag = np.diag(theta_all)

        ChannelA = 1/np.power(10, PLA/10) * np.exp(-1.0j*(2*np.pi)*dA*(self.fc/0.3))*a_aoa_all
        ChannelB = 1/np.power(10, PLB/10) * np.exp(-1.0j*(2*np.pi)*dB*(self.fc/0.3))*a_aod_all.conj().T

        PL_RIS_sig = np.dot(np.dot(ChannelA,  theta_diag), ChannelB)
        PL_RIS = np.log10((1 / (PL_RIS_sig))) * 10
        return PL_RIS  # + self.shadow_std * np.random.normal()

    def overall_channel(self, RIS_position, theta, n_elements):
        """ Renew slow fading channel """
        self.D_pathloss = np.zeros((4, 4)) + 50 * np.identity(4)
        self.C_pathloss = np.zeros((4))
        self.D_RIS_pathloss = np.zeros((4, 4), dtype=complex)
        self.C_RIS_pathloss = np.zeros((4), dtype=complex)

        self.D_channels_abs = np.zeros((4, 4))
        self.C_channels_abs = np.zeros((4))

        self.RIS_position = RIS_position
        self.theta = theta
        self.n_elements = n_elements

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
                                                                                                 self.theta,
                                                                                                 self.n_elements)
        self.D_overall = 1 / np.abs(1 / np.power(10, self.D_pathloss / 10) + 1 / np.power(10, self.D_RIS_pathloss/10))
        self.D_channels_abs = 10 * np.log10(self.D_overall)

        for i in range(4):
            self.C_pathloss[i] = self.get_C_path_loss(self.cuser_center[i])
            self.C_RIS_pathloss[i] = self.get_C_RIS_path_loss(self.cuser_center[i], self.RIS_position, self.theta, self.n_elements)
        self.C_overall = 1 / np.abs(1 / np.power(10, self.C_pathloss / 10) + 1 / np.power(10, self.C_RIS_pathloss/10))
        self.C_channels_abs = 10 * np.log10(self.C_overall)

        for i in range (4):
            self.D_to_BS_pathloss[i] = self.get_C_path_loss(self.duser_center[i])
            self.D_to_BS_RIS_pathloss[i] = self.get_C_RIS_path_loss(self.duser_center[i], self.RIS_position, self.theta,
                                                                                                 self.n_elements)
            for j in range (4):
                self.C_to_D_pathloss[i, j] = self.get_D_path_loss(self.cuser_center[i], self.duser_center[j]) # i-th cellular user to j-th D2D user
                self.C_to_D_RIS_pathloss[i, j] = self.get_D_RIS_path_loss(self.cuser_center[i], self.duser_center[j], self.RIS_position, self.theta,
                                                                                                 self.n_elements)
        self.C_to_D_overall = 1 / np.abs(1 / np.power(10, self.C_to_D_pathloss / 10) + 1 / np.power(10, self.C_to_D_RIS_pathloss))
        self.D_to_BS_overall = 1 / np.abs(1 / np.power(10, self.D_to_BS_pathloss / 10) + 1 / np.power(10, self.D_to_BS_RIS_pathloss ))

        self.C_to_D_channels_abs = 10 * np.log10(self.C_to_D_overall)
        self.D_to_BS_channels_abs = 10 * np.log10(self.D_to_BS_overall)

    def renew_channels_fastfading(self):
        """ Renew fast fading channel """

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

    def compute_reward(self, resource_allocation_action):
        self.resource_allocation_action = resource_allocation_action
        action_all_testing = np.zeros([self.n_D2D, 1, 2], dtype='int32')
        for i in range(len(self.duser_center)):
            for j in range(1):
                action_all_testing[i, j, 0] = self.resource_allocation_action[i] % self.n_RB  # chosen RB
                action_all_testing[i, j, 1] = int(np.floor(self.resource_allocation_action[i] / self.n_RB))  # power level

        resource_allocation_action_temp = action_all_testing.copy()
        coefficient = resource_allocation_action_temp[:, :, 0]  # the channel_selection_part
        power_selection = resource_allocation_action_temp[:, :, 1]  # power selection


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
            indexes = np.argwhere(coefficient == i)
            if (len(indexes) != 0):
                for j in range(len(indexes)):
                    C_Interference[coefficient[i][j]] += 10 ** ((real_power[indexes[j, 0]]  -
                                                                     self.D_to_BS_channels_abs[indexes[j, 0]]) / 10)
                    break
        self.C_Interference = C_Interference + self.sig2
        C_Signals = 10 ** ((self.C_power_dB - self.C_channels_abs)/ 10)
        C_Rate = np.log2(1 + np.divide(C_Signals, self.C_Interference))

        # ------------ Compute V2V rate -------------------------
        D_Interference = np.zeros((4,1))
        D_Signal = np.zeros((4,1))
        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(coefficient == i)  # find spectrum-sharing V2Vs
            for j in range(len(indexes)):
                receiver_j = self.D2D_users[indexes[j, 0]].destinations[indexes[j, 1]]
                # print(self.D_channels_with_fastfading[indexes[j][0], receiver_j, i])
                D_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((real_power[indexes[j, 0]]- self.D_channels_with_fastfading[indexes[j][0], receiver_j, i]) / 10)
                # Cellular links interference to D2D links
                D_Interference[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.C_power_dB - self.C_to_D_channels_abs[i, receiver_j]) / 10)

                #  D2D interference
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.D2D_users[indexes[k][0]].destinations[indexes[k][1]]
                    D_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((real_power[indexes[k, 0]]
                                                                              - self.D_channels_with_fastfading[indexes[k][0]][receiver_j][i]) / 10)
                    D_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((real_power[indexes[j, 0]]
                                                                              - self.D_channels_with_fastfading[indexes[j][0]][receiver_k][i]) / 10)
        self.D_Interference = D_Interference + self.sig2
        D_Rate = np.log2(1 + np.divide(D_Signal, self.D_Interference))
        return C_Rate, D_Rate

    def Compute_Interference(self, actions):
        D2D_Interference = np.zeros((len(self.D2D_users), 1, self.n_RB)) + self.sig2

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
                        # if i == k or channel_selection[i,j] >= 0:
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        D2D_Interference[k, m, channel_selection[i, j]] += 10 ** (
                                    (self.D_power_dB_List[power_selection[i, j]]
                                     - self.D_channels_with_fastfading[i][self.D2D_users[k].destinations[m]][
                                         channel_selection[i, j]]) / 10)
        self.D2D_Interference_all = 10 * np.log10(D2D_Interference)

    def act_for_training(self, resource_allocation_action):
        reward = 0
        reward_C = 0
        reward_D = 0
        done = True
        resource_allocation_action_temp = resource_allocation_action
        C_Rate, D_Rate = self.compute_reward(resource_allocation_action_temp)
        for i in range (len(C_Rate)):
            if (C_Rate[i] > 3.16):
                reward_C += C_Rate[i] / (self.n_RB * 10)
            else:
                reward_C += 0
                done = False
        for i in range(len(D_Rate)):
            if (i % 2 == 0):
                if (D_Rate[i]> 2):
                    reward_D += D_Rate[i] / (self.n_RB * 10)
                else:
                    reward_D += 0
                    done = False
        lambdda = 1.
        if done:
            reward = reward_C + reward_D
        else:
            reward = 0


        return reward

    def get_next_state(self, observation, action_temp, next_coords):
        M = 3 # number of phase adjustment, i.e., +1, 0, -1
        n_elements = 4
        self.observation = observation
        self.n_phases = np.power(3, n_elements)
        RIS_next_coords = [next_coords[0], next_coords[1]]
        theta_number = self.observation[2 : 2+n_elements] * n_phase_adjust_number
        element_phase_action = np.zeros(n_elements)
        phase_action = action_temp % self.n_phases
        for n in range (n_elements):
            element_phase_action[n] = int(np.floor(phase_action % np.power(M, n + 1) / np.power(M, n)))
            if element_phase_action[n] == 0:
                theta_number[n] = theta_number[n]
            elif element_phase_action[n] == 1:
                theta_number[n] = (theta_number[n] + 1) % n_phase_adjust_number
            elif element_phase_action[n] == 2:
                theta_number[n] = (theta_number[n] - 1) % n_phase_adjust_number
            else:
                print("Something goes wrong!")

        next_theta_number = theta_number / n_phase_adjust_number

        theta = np.zeros(n_elements, dtype=complex)
        for n in range (n_elements):
            theta [n] = np.exp(1j * theta_number[n] * (2 * np.pi / n_phase_adjust_number))
        return RIS_next_coords, theta, next_theta_number


# class Maze(tk.Tk, object):
class Maze():
    def __init__(self, duser_center, n_elements):
        super(Maze, self).__init__()
        self.n_positions = 16
        self.duser_center = duser_center
        self.n_features = 6 + 2 * len(duser_center)

        self._build_maze()

    def _build_maze(self):

        self.oval_center = np.zeros([self.n_positions,2],dtype = float)

        for i in range (int(self.n_positions)):
            self.oval_center [i, 0] = 12.5 + 25*(i%int(np.sqrt(self.n_positions)))
            self.oval_center [i, 1] = 12.5 + 25*(np.floor(i/int(np.sqrt(self.n_positions))))

        self.cuser_center = np.array([[25,30], [95,65], [50,85], [75,30]])

        # --------------------------BS--------------------------------
        self.BS_center = [50,50]

        
    # --------------------------Reset position-----------------------------
    def reset_ris(self, n_elements_total):
        n_elements = 4
        self.theta_number = np.zeros(n_elements)
        action_all = np.zeros([len(self.duser_center), 1, 2], dtype='int32')
        return np.hstack((np.array([0, 0]), self.theta_number, np.reshape(action_all, -1)))
        
    # ------------------------math--RIS moves each step-----------------------------
    def step(self, observation, action, resource_allocation_action, C_power_dB, D_power_dB_List, n_elements):
        """ Interaction step """
        M = 3 # number of phase adjustment, i.e., +1, 0, -1
        self.observation = observation
        self.n_phases = np.power(M, 4)
        self.n_actions = int (self.n_positions * self.n_phases)
        self.action_space = []
        action_temp = action

        for i in range (int(self.n_actions)):
            self.action_space.append(str(i+1))
        
        for i in range(self.n_positions): # find position
            if int (np.floor(action_temp/self.n_phases)) == i:
                point = self.oval_center[i, :]
                next_coords = [point[0], point[1]]
                break

        train = Training(self.BS_center, self.cuser_center, self.duser_center, C_power_dB, D_power_dB_List)
        RIS_next_coords, theta, next_theta_number = train.get_next_state(self.observation, action_temp, next_coords)
        train.overall_channel(RIS_next_coords, theta, n_elements)
        train.renew_channels_fastfading()

        resource_allocation_action_temp = resource_allocation_action


        C_Rate, D_Rate = Training.compute_reward(train, resource_allocation_action_temp)
        done = True
        for i in range (len (C_Rate)):
            if (C_Rate[i] > 3.16):
                done = True
            else:
                done = False
                break
        for i in range(len(D_Rate)):
            if (i % 2 == 0):
                if (D_Rate[i]> 2):
                    done = True
                else:
                    done = False
                    break
        if done:
            reward = Training.act_for_training(train, resource_allocation_action_temp)
        else:
            reward = Training.act_for_training(train, resource_allocation_action_temp)
        throughput = C_Rate + sum(D_Rate)
        reward_max = reward
        throughput_max = throughput


        action_all = np.zeros([len(self.duser_center), 1, 2], dtype='float')
        for i in range(len(self.duser_center)):
            for j in range(1):
                action_all[i, j, 0] = resource_allocation_action_temp[i] % len(self.cuser_center) / len(self.duser_center)  # chosen RB
                action_all[i, j, 1] = int(np.floor(resource_allocation_action_temp[i] / len(self.cuser_center)))/ len(D_power_dB_List) # power level

        s_ = np.hstack((np.array([next_coords[0] / (height), next_coords[1] / (width)]), next_theta_number, np.reshape(action_all, -1)))
        return s_, reward, done, throughput

# if __name__ == "__main__":
#     env = Maze()
#     env.mainloop()

