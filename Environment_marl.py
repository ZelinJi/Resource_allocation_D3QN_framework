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
        self.h_ris = 10
        self.sig2_dB = -135
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.bsAntGain = 1
        self.bsNoiseFigure = 8
        self.vehAntGain = 1
        self.vehNoiseFigure = 9
        self.RISNoiseFigure = 3
        self.fc = 2
        self.temp = 1 / pow(2, 1 / 2)
        self.ric_fac = 10
        self.ric_LoS = np.sqrt(self.ric_fac / (self.ric_fac + 1))
        self.ric_NLoS = np.sqrt(1 / (self.ric_fac + 1))


        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/vehicle position every 100 ms

        self.n_D2D = len(duser_center)
        self.n_RB = len(cuser_center)
        self.n_neighbor = 1
        self.n_pairs = int(self.n_D2D / (self.n_neighbor + 1))
        self.D2D_users = []
        self.add_D2Ds(self.n_D2D)
        self.renew_neighbor()

        self.RIS_position = np.zeros(2)
        self.theta = np.zeros(self.N, dtype=complex)
        self.random_RIS_implement()

        self.overall_channel()
        # self.renew_channels_fastfading()
        self.active_links = np.ones((self.n_D2D, 1), dtype='bool')

        self.D2D_Interference_all = np.zeros((self.n_D2D, self.n_RB)) + self.sig2

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
            for j in range(self.n_neighbor):
                self.D2D_users[i].neighbors.append(sort_idx[j + 1])
            destination = self.D2D_users[i].neighbors

            self.D2D_users[i].destinations = destination

    def get_D_coefficient(self, position_A, position_B):
        """Calculate direct coefficient between D2D pairs"""
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        distance = math.hypot(d1, d2)

        PL = 32.4 + 38.3 * np.log10(distance) + 21.3 * np.log10(self.fc)        # - 2 * self.vehAntGain + self.vehNoiseFigure

        gain_coefficient = np.sqrt(1 / 10 ** (PL / 10)) # channel coefficient

        real = np.random.normal(0, 1)  # 瑞利分布实数
        imag = np.random.normal(0, 1)  # 瑞利分布虚数部分

        real = self.temp * real
        imag = self.temp * imag

        rayleign = complex(real, imag)  # 复高斯瑞利分布

        CL = rayleign * gain_coefficient

        return CL  # + self.shadow_std * np.random.normal()

    def get_D_channels(self, position_A, position_B, position_RIS, theta):
        """Calculate channel coefficient between D2D pairs with RIS"""
        theta_diag = np.diag(theta)
        ds = 0.038
        dA1 = abs(position_A[0] - position_RIS[0])
        dA2 = abs(position_A[1] - position_RIS[1])
        dA3 = abs(self.h_ris - self.h_ms)
        dB1 = abs(position_B[0] - position_RIS[0])
        dB2 = abs(position_B[1] - position_RIS[1])
        dB3 = abs(self.h_ris - self.h_ms)
        dA = math.hypot(dA1, dA2, dA3)
        dB = math.hypot(dB1, dB2, dB3)
        a_aoa = np.zeros(self.N, dtype=complex)
        a_aod = np.zeros(self.N, dtype=complex)
        CL_A = np.zeros(self.N, dtype=complex)
        CL_B = np.zeros(self.N, dtype=complex)
        theta_aoa = np.arctan((math.hypot(dA1, dA2))/(dA3))
        theta_aod = np.arctan((math.hypot(dB1, dB2))/(dB3))

        PLA = 10 + 30.3 * np.log10(dA) + 10 * np.log10(self.fc) # - self.vehAntGain + self.RISNoiseFigure
        PLB = 10 + 30.3 * np.log10(dB) + 10 * np.log10(self.fc)  # - self.vehAntGain + self.RISNoiseFigure

        gain_coefficient_A = np.sqrt(1 / 10 ** (PLA / 10))
        gain_coefficient_B = np.sqrt(1 / 10 ** (PLB / 10))

        real = np.mat(np.random.normal(0, 1, self.N))
        imag = np.mat(np.random.normal(0, 1, self.N))
        for n in range (self.N):
            a_aod[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aod)) # h_LoS
            a_aoa[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aoa))
            rayleign = complex(real[0, n], imag[0, n])  # 复高斯瑞利分布
            CL_A[n] = rayleign * gain_coefficient_A * self.ric_NLoS + gain_coefficient_A * self.ric_LoS * a_aoa[n] # 这是RICIAN中的瑞利分布，V*NLOS部分
            CL_B[n] = rayleign * gain_coefficient_B * self.ric_NLoS + gain_coefficient_B * self.ric_LoS * a_aod[n] # 这是RICIAN中的瑞利分布，V*NLOS部分

        CL_RIS = np.dot(np.dot(CL_A,  theta_diag), CL_B.conj().T)
        CL_direct = self.get_D_coefficient(position_A, position_B)

        CL = CL_RIS + CL_direct

        return CL  # + self.shadow_std * np.random.normal()

    def get_C_coefficient(self, position_A):
        """Calculate channel coefficient between cellular users and BS"""
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        d3 = abs(self.h_bs - self.h_ms)
        distance = math.hypot(d1, d2, d3)

        PL = 32.4 + 38.3 * np.log10(distance) + 21.3 * np.log10(self.fc)        # - self.vehAntGain - self.bsAntGain + self.bsNoiseFigure
        gain_coefficient = np.sqrt(1 / 10 ** (PL / 10)) # channel coefficient

        real = np.random.normal(0, 1)  # 瑞利分布实数
        imag = np.random.normal(0, 1)  # 瑞利分布虚数部分

        real = self.temp * real
        imag = self.temp * imag

        rayleign = complex(real, imag)  # 复高斯瑞利分布

        CL = rayleign * gain_coefficient
        return CL

    def get_C_channels (self, position_A, position_RIS, theta):
        """Calculate channel coefficient between cellular users and BS with RIS"""
        theta_diag = np.diag(theta)
        ds = 0.038 # The spacing between elements
        dA1 = abs(position_A[0] - position_RIS[0])
        dA2 = abs(position_A[1] - position_RIS[1])
        dB1 = abs(self.BS_position[0] - position_RIS[0])
        dB2 = abs(self.BS_position[1] - position_RIS[1])
        dA = math.hypot(dA1, dA2)
        dB = math.hypot(dB1, dB2)
        a_aoa = np.zeros(self.N, dtype=complex)
        a_aod = np.zeros(self.N, dtype=complex)
        CL_A = np.zeros(self.N, dtype=complex)
        CL_B = np.zeros(self.N, dtype=complex)
        theta_aoa = np.arctan((position_A[1] - position_RIS[1])/(position_A[0] - position_RIS[0]))
        theta_aod = np.arctan((self.BS_position[1] - position_RIS[1])/(self.BS_position[0] - position_RIS[0]))

        PLA = 10 + 30.3 * np.log10(dA) + 10 * np.log10(self.fc) # - self.vehAntGain + self.RISNoiseFigure
        PLB = 10 + 30.3 * np.log10(dA) + 10 * np.log10(self.fc) # - self.bsAntGain + self.RISNoiseFigure

        gain_coefficient_A = np.sqrt(1 / 10 ** (PLA / 10))
        gain_coefficient_B = np.sqrt(1 / 10 ** (PLB / 10))

        real = np.mat(np.random.normal(0, 1, self.N))
        imag = np.mat(np.random.normal(0, 1, self.N))
        for n in range (self.N):
            a_aod[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aod)) # h_LoS
            a_aoa[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aoa))
            rayleign = complex(real[0, n], imag[0, n])  # 复高斯瑞利分布
            CL_A[n] = rayleign * gain_coefficient_A * self.ric_NLoS + gain_coefficient_A * self.ric_LoS * a_aoa[n]
            CL_B[n] = rayleign * gain_coefficient_B * self.ric_NLoS + gain_coefficient_B * self.ric_LoS * a_aod[n]

        CL_RIS = np.dot(np.dot(CL_A,  theta_diag), CL_B.conj().T)
        CL_direct = self.get_C_coefficient(position_A)

        CL = CL_RIS + CL_direct

        return CL  # + self.shadow_std * np.random.normal()

    def random_RIS_implement(self):
        """RIS are randomly deployed during the training of resource allocation model"""
        position_number = np.random.normal(self.n_positions / 2, int(np.sqrt(self.n_positions)), 1)
        self.RIS_position [0] = 12.5 + 25 * (int(position_number) % int(np.sqrt(self.n_positions)))
        self.RIS_position [1] = 12.5 + 25 * (np.floor(int(position_number) / int(np.sqrt(self.n_positions))))
        phase_number = np.random.randint(0, 8, self.N)
        for n in range (self.N):
            self.theta[n] = np.exp(1.0j*(np.pi/4)*phase_number[n])

    def overall_channel(self):
        """ The combined channel"""
        self.D_channels = np.zeros((4, 4),dtype=complex)
        self.C_channels = np.zeros((4), dtype=complex)

        self.D_channels_abs = np.zeros((4, 4))
        self.C_channels_abs = np.zeros((4))


        self.D_to_BS_channels = np.zeros((4), dtype=complex) # D2D to BS interference channel
        self.C_to_D_channels = np.zeros((4, 4), dtype=complex) # Cellular user to D2D user interference channel
        self.D_to_BS_channels_abs = np.zeros((4))
        self.C_to_D_channels_abs = np.zeros((4, 4))



        for i in range(4):
            for j in range(i + 1, 4):
                self.D_channels[j, i] = self.D_channels[i][j] = self.get_D_channels(self.duser_center[i],
                                                                                                 self.duser_center[j],
                                                                                                 self.RIS_position,
                                                                                                 self.theta)
        self.D_channels_abs = abs(self.D_channels) ** 2
        self.D_pathloss = np.log10((1 / (self.D_channels_abs))) * 10

        for i in range(4):
            self.C_channels[i] = self.get_C_channels(self.cuser_center[i], self.RIS_position, self.theta)
        self.C_channels_abs = abs(self.C_channels) ** 2
        self.C_pathloss = 10 * np.log10(1 / self.C_channels_abs)

        for i in range (4):
            self.D_to_BS_channels[i] = self.get_C_channels(self.duser_center[i], self.RIS_position, self.theta)
            for j in range (4):
                self.C_to_D_channels[i, j] = self.get_D_channels(self.cuser_center[i], self.duser_center[j], self.RIS_position, self.theta)

        self.D_to_BS_channels_abs = abs(self.D_to_BS_channels) ** 2
        self.C_to_D_channels = abs(self.C_to_D_channels) ** 2

        self.D_to_BS_pathloss = np.log10((1 / (self.D_to_BS_channels_abs))) * 10
        self.C_to_D_pathloss = np.log10((1 / (self.C_to_D_channels))) * 10

    def compute_reward(self, agent_index, action_all):
        """Calculate the data rate and the reward for reinforcement learning"""
        channel_selection = action_all[:, :, 0]  # the channel_selection_part
        power_selection = action_all[:, :, 1]  # power selection

        # ------------ Compute Cellular rate --------------------
        C_Rate = np.zeros(self.n_RB)
        C_Interference = np.zeros(self.n_RB)  # V2I interference
        transmitter_power = np.zeros(self.n_RB)

        coefficient = np.zeros(self.n_RB, dtype = int)
        for i in range(self.n_RB):
            if i in agent_index.keys():
                coefficient[i] = channel_selection[agent_index[i], 0]
                transmitter_power[i] = self.D_power_dB_List[power_selection[agent_index[i], 0]]
            else:
                coefficient[i] = -1
                transmitter_power[i] = -np.inf

        for i in range(self.n_RB):
            C_Interference[coefficient[i]] += 10 ** ((transmitter_power[i] - self.D_to_BS_pathloss[i] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

        self.C_Interference = C_Interference + self.sig2
        C_Signals = 10 ** ((self.C_power_dB - self.C_pathloss + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        C_Rate = np.log2(1 + np.divide(C_Signals, self.C_Interference))

        # ------------ Compute D2D rate -------------------------
        D_Interference = np.zeros(self.n_D2D)
        D_Signal = np.zeros(self.n_D2D)
        D_Rate = np.zeros(self.n_D2D)

        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(coefficient == i)  # find spectrum-sharing D2Ds
            if (len(indexes) != 0):
                for j in range(len(indexes)):
                    transmitter_j = indexes[j, 0]
                    receiver_j = self.D2D_users[transmitter_j].destinations[0]
                    D_Signal[transmitter_j] = 10 ** ((transmitter_power[transmitter_j] - self.D_pathloss[transmitter_j, receiver_j] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    # Cellular links interference to D2D links
                    D_Interference[receiver_j] = 10 ** ((self.C_power_dB - self.C_to_D_pathloss[i, receiver_j] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                    #  D2D interference
                    for k in range(j + 1, len(indexes)):  # spectrum-sharing D2Ds
                        transmitter_k = indexes[k][0]
                        receiver_k = self.D2D_users[transmitter_k].destinations[0]
                        D_Interference[receiver_j] += 10 ** ((transmitter_power[transmitter_k] - self.D_pathloss[transmitter_k][receiver_j] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                        D_Interference[receiver_k] += 10 ** ((transmitter_power[transmitter_j] - self.D_pathloss[transmitter_j][receiver_k] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.D_Interference = D_Interference + self.sig2
        for i in range(self.n_RB):
            receiver = self.D2D_users[i].destinations[0]
            D_Rate[i] = np.log2(1 + np.divide(D_Signal[i], self.D_Interference[receiver]))
        return C_Rate, D_Rate

    def Compute_Interference(self, agent_index, actions):
        """Calculate Interference"""
        D2D_Interference = np.zeros((len(self.D2D_users), self.n_RB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        transmitter_power = np.zeros(self.n_RB)

        coefficient = np.zeros(self.n_D2D, dtype = int)
        for i in range(self.n_D2D):
            if i in agent_index.keys():
                coefficient[i] = channel_selection[agent_index[i], 0]
                transmitter_power[i] = self.D_power_dB_List[power_selection[agent_index[i], 0]]
            else:
                coefficient[i] = -1
                transmitter_power[i] = -np.inf
        # interference from cellular links
        for i in range(self.n_RB):
            for j in range(len(self.D2D_users)):
                receiver_j = self.D2D_users[j].destinations[0]
                D2D_Interference[receiver_j, i] += 10 ** ((self.C_power_dB - self.C_to_D_pathloss[i][receiver_j] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # interference from peer D2D links
        for i in range(self.n_D2D):
            for j in range(self.n_D2D):
                if coefficient[i] == coefficient[j] and i != j:
                    receiver_j = self.D2D_users[j].destinations[0]
                    D2D_Interference[receiver_j, coefficient[i]] += 10 ** ((transmitter_power[i] - self.D_pathloss[i][receiver_j] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.D2D_Interference_all = 10 * np.log10(D2D_Interference)

    def act_for_training(self, agent_index, actions):
        """Determine the real reward based on the SINR constraints"""
        done = True
        reward_C = 0
        reward_D = 0
        action_temp = actions.copy()
        C_Rate, D_Rate = self.compute_reward(agent_index, action_temp)
        for i in range (len(C_Rate)):
            if (C_Rate[i] < 3.16):
                done = False
            reward_C += C_Rate[i] / (len(agent_index) * 10)
        for i in range(len(D_Rate)):
            if (D_Rate[i] > 2):
                reward_D += D_Rate[i] / (len(agent_index) * 10)
            elif i in agent_index:
                done = False
        if done:
            reward = reward_C + reward_D
        else:
            reward = 0
        return reward

    def act_for_testing(self, agent_index, actions):
        """Determine the real reward based on the SINR constraints (Strict for testing)"""
        done = True
        reward_C = 0
        reward_D = 0
        action_temp = actions.copy()
        C_Rate, D_Rate = self.compute_reward(agent_index, action_temp)
        for i in range (len(C_Rate)):
            if (C_Rate[i] < 3.16):
                done = False
            reward_C += C_Rate[i] / (self.n_RB * 10)
        for i in range(len(D_Rate)):
            if (D_Rate[i]> 2):
                reward_D += D_Rate[i] / (self.n_RB * 10)
            elif i in agent_index:
                done = False
        if done:
            reward = reward_C + reward_D
        else:
            reward = 0
        return C_Rate, D_Rate, reward
