import numpy as np
from src.parameters import QLearningParams as qlp
from src.data_structures import Coords3d
from src.math_tools import decision
import itertools
import copy
import os
from src.parameters import rng, MAX_DIST_FROM_DBS
from npy_append_array import NpyAppendArray

# Actions = orientation: +x, +y, +z, 0 3d loc: +x, +y, +z, -x, -y, -z, +x +y, -x +y, -x -y, +x -y,
# +x +z, -x +z, -x -z, +x -z, +y +z, -y +z, -y -z, +y -z,
# +x +y +z, +x +y -z, +x -y +z, +x -y -z,
# -x +y +z, -x +y -z, -x -y +z, -x -y -z, 0 = 3 * 27 = 81
# States prev_v_b_az, v_b_az, d_v_elev, d_v_az, d_b_elev, d_b_az
# Reward delta_payload/total_payload


d_orient_dict = {0: 0, 1: 1, 2: -1, 3: 0.5, 4: -0.5, 5: 0.25, 6: -0.25, 7: 0.75, 8: -0.75}


class QManager:
    q_values = None
    state_idxs = np.zeros((qlp.N_Q_PAIRS * 4), dtype=int)
    selected_action = None
    last_state_idxs = None
    last_action_idx = 0

    def __init__(self, dbs_list, max_dist, boundaries, testing_flag=qlp.TESTING_FLAG,
                 save_on=qlp.SAVE_ON, load_model=qlp.LOAD_MODEL):
        self.dbs_list = dbs_list
        self.testing_flag = testing_flag
        self.save_on = save_on  # Every N cycles
        self.max_dist = max_dist
        self.boundaries = boundaries
        self.orientation_space_size = len(d_orient_dict)
        self.action_space_size = self.orientation_space_size
        self.available_action_idxs = np.ones(self.action_space_size, dtype=int)
        self.q_shape = (
            (2, qlp.AZIMUTH_CARDINALITY, qlp.AZIMUTH_CARDINALITY, qlp.ELEVATION_CARDINALITY, qlp.ELEVATION_CARDINALITY,
             self.action_space_size))
        self.initialize_model(load_model)
        self.d_orient_dict = d_orient_dict
        self.prev_d_center = None

    def initialize_model(self, load_model=qlp.LOAD_MODEL):
        if not load_model:
            self.q_values = np.zeros(self.q_shape)
            self.cycle_idx = 0
        else:
            file_name = os.path.join(qlp.CHECKPOINTS_FILE, self.__repr__() + '.npy')
            file_name_cycle_idx = os.path.join(qlp.CHECKPOINTS_FILE, self.__repr__() + '_cycle_idx' + '.npy')
            self.q_values = np.load(file_name)
            self.cycle_idx = np.load(file_name_cycle_idx)
        if self.save_on:
            self.rewards_history = np.zeros(self.save_on)
            rewards_file = os.path.join(qlp.REWARDS_FILE, self.__repr__())
            if not load_model and os.path.exists(rewards_file):
                os.remove(rewards_file)

    def get_elev_idx(self, elev_val):
        return int(np.floor(qlp.ELEVATION_CARDINALITY * (elev_val / (np.pi / 2)) ** 0.5))

    def get_az_idx(self, az_val):
        return int(np.floor((qlp.AZIMUTH_CARDINALITY - 1) * ((np.cos(az_val) + 1) / 2) ** 0.4))

    def get_elev_idx_uniform(self, elev_val):
        return int(elev_val / ((np.pi / 2) / qlp.ELEVATION_CARDINALITY))

    def get_az_idx_uniform(self, az_val):
        return int(az_val / (2 * np.pi / (qlp.AZIMUTH_CARDINALITY - 1)))

    def get_dist_idx(self, dist_val):
        return int(dist_val * qlp.DISTANCES_CARDINALITY / self.max_dist)

    def get_dist_idx_negative(self, dist_val, max_dist=MAX_DIST_FROM_DBS):
        return max(min(int((dist_val + max_dist) / (max_dist * 2) * (qlp.DISTANCES_CARDINALITY - 1)),
                       qlp.DISTANCES_CARDINALITY - 1), 0)

    def get_dist_idx_nonuniform(self, dist_val):
        return max(min(int(np.floor((dist_val / self.max_dist) ** 0.5 * (qlp.DISTANCES_CARDINALITY - 1))),
                       qlp.DISTANCES_CARDINALITY - 1), 0)

    def update_state_idxs(self, selected_pairs, dbs_coords):
        self.last_state_idxs = tuple(self.state_idxs)
        self.state_idxs = np.zeros((qlp.N_Q_PAIRS * 4), dtype=int)
        for idx, pair in enumerate(selected_pairs):
            if idx < qlp.N_Q_PAIRS:
                az_r_idx = self.get_az_idx_uniform(pair.stats.r_rx_phi % (2 * np.pi))
                az_t_idx = self.get_az_idx_uniform(pair.stats.r_tx_phi % (2 * np.pi))
                elev_r_idx = self.get_elev_idx(pair.stats.r_rx_theta)
                elev_t_idx = self.get_elev_idx(pair.stats.r_tx_theta)
                # prev_az_r_idx = self.get_az_idx(pair.prev_stats.r_rx_phi % (2 * np.pi))
                # prev_az_t_idx = self.get_az_idx(pair.prev_stats.r_tx_phi % (2 * np.pi))
                # prev_elev_r_idx = self.get_elev_idx(pair.prev_stats.r_rx_theta)
                # prev_elev_t_idx = self.get_elev_idx(pair.prev_stats.r_tx_theta)
                self.state_idxs[idx * 4: (idx + 1) * 4] = az_r_idx, az_t_idx, elev_r_idx, elev_t_idx  #,prev_az_r_idx, prev_az_t_idx, prev_elev_r_idx, prev_elev_t_idx
        # self.state_idxs[-1] = self.last_action_idx
        self.state_idxs = tuple(self.state_idxs)

    def save_checkpoint(self, folder_name=qlp.CHECKPOINTS_FILE):
        file_name = os.path.join(folder_name, self.__repr__())
        np.save(file_name, self.q_values)
        file_name_cycle_idx = os.path.join(qlp.CHECKPOINTS_FILE, self.__repr__() + '_cycle_idx' + '.npy')
        np.save(file_name_cycle_idx, self.cycle_idx)

    def save_rewards_history(self, folder_name=qlp.REWARDS_FILE):
        file_name = os.path.join(folder_name, self.__repr__())
        with NpyAppendArray(file_name) as npaa:
            npaa.append(self.rewards_history)

    @staticmethod
    def get_learning_rate(cycle_idx):
        return qlp.LEARNING_RATE(cycle_idx + 1)

    @staticmethod
    def get_exploration_probability(cycle_idx):
        return qlp.EXPLORATION_PROB(cycle_idx)

    @staticmethod
    def get_discount_ratio():
        return qlp.DISCOUNT_RATIO

    @staticmethod
    def __repr__(id=None):
        return f'QManager_Single_or_V1_{qlp.CHECKPOINT_ID if not id else id}'

    def select_action(self, t_step, q_set=None, testing=False):
        q_values = self.q_values[0][self.state_idxs] + self.q_values[1][self.state_idxs] if q_set is None else \
            self.q_values[q_set][self.state_idxs]
        self.available_action_idxs[:] = 1
        selected_action_idx = self.select_action_from_qs(q_values, testing)
        orientation_idx = selected_action_idx
        selected_action = np.array([0, 0, 0]), self.d_orient_dict[orientation_idx]
        self.selected_action = selected_action
        self.last_action_idx = orientation_idx
        return self.last_action_idx

    def update_q_values(self, reward, t_step):
        q_set_idx = rng.integers(2)
        last_action_idx = self.last_action_idx
        new_action_idx = self.select_action(t_step, testing=True, q_set=q_set_idx)
        learning_rate = QManager.get_learning_rate(self.cycle_idx)
        discount_ratio = QManager.get_discount_ratio()
        new_action_q = self.q_values[1 - q_set_idx][self.state_idxs][new_action_idx]
        old_q_value = self.q_values[q_set_idx][self.last_state_idxs][last_action_idx]
        updated_q_value = old_q_value + learning_rate * (reward + discount_ratio * new_action_q - old_q_value)
        self.q_values[q_set_idx][self.last_state_idxs][last_action_idx] = updated_q_value
        if qlp.BATCH_INITIALIZATION:
            zero_indices = np.where(self.q_values[q_set_idx][self.last_state_idxs] == 0)[0]
            if zero_indices.shape[0] > 0:
                self.q_values[q_set_idx][self.last_state_idxs][zero_indices] = updated_q_value

    def select_action_from_qs(self, q_values, testing=False):
        if not testing and decision(QManager.get_exploration_probability(self.cycle_idx)):
            if qlp.EXPLORE_UNEXPLORED:
                zero_indices = np.where(q_values)[0]
                if zero_indices.shape[0] > 0:
                    return rng.choice(zero_indices)
            idx = rng.integers(q_values.shape[0])
        else:
            idx = rng.choice(np.where(q_values == q_values.max())[0])
        return idx

    def begin_cycle(self, t_step):
        self.select_action(t_step, testing=self.testing_flag)
        return self.selected_action

    def end_cycle(self, reward, t_step):
        if self.save_on:
            self.rewards_history[self.cycle_idx % self.save_on] = reward
        if not self.testing_flag:
            self.update_q_values(reward, t_step)
        self.cycle_idx += 1
        if self.save_on and not (self.cycle_idx % self.save_on):
            self.save_checkpoint()
            self.save_rewards_history()
