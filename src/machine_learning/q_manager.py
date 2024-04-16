import numpy as np
from src.parameters import QLearningParams as qlp
from src.data_structures import Coords3d
from src.math_tools import decision
import itertools
import copy
import os
from src.parameters import rng
from npy_append_array import NpyAppendArray

# Actions = orientation: +x, +y, 0 2d loc: +x, +y, -x, -y, +x +y, -x +y, -x -y, +x -y, 0 = 3 * 9 = 27
# States elevation_r_max_payload, azimuth_r_max_payload, elevation_t_max_payload, azimuth_t_max_payload, median_azimuths, median_elevations, mean_distances
# Reward delta_payload/total_payload

d_loc_dict = {0: np.array([0, 0, 0]), 1: np.array([1, 0, 0]), 2: np.array([0, 1, 0]),
              3: np.array([0.7071067811865475, 0.7071067811865475, 0]),
              4: np.array([-0.7071067811865475, 0.7071067811865475, 0]),
              5: np.array([0.7071067811865475, -0.7071067811865475, 0]),
              6: np.array([-0.7071067811865475, -0.7071067811865475, 0]),
              7: np.array([-1, 0, 0]),
              8: np.array([0, -1, 0]),
              }
d_orient_dict = {0: 0, 1: 1, 2: -1}


class QManager:
    q_values = None
    state_idxs = None
    selected_action = None
    last_state_idxs = None
    last_action_idx = None

    def __init__(self, dbs_list, max_dist, boundaries, testing_flag=qlp.TESTING_FLAG,
                 save_on=qlp.SAVE_ON, load_model=qlp.LOAD_MODEL):
        self.dbs_list = dbs_list
        self.testing_flag = testing_flag
        self.save_on = save_on  # Every N cycles
        self.max_dist = max_dist
        self.boundaries = boundaries
        self.loc_space_size = len(d_loc_dict)
        self.orientation_space_size = len(d_orient_dict)
        self.action_space_size = self.loc_space_size * self.orientation_space_size
        self.available_action_idxs = np.ones(self.action_space_size, dtype=int)
        self.q_shape = ((2, qlp.ELEVATION_CARDINALITY, qlp.AZIMUTH_CARDINALITY,
                         qlp.ELEVATION_CARDINALITY, qlp.AZIMUTH_CARDINALITY,
                         qlp.ELEVATION_CARDINALITY, qlp.AZIMUTH_CARDINALITY,
                         qlp.DISTANCES_CARDINALITY, self.action_space_size))
        self.d_loc_dict = d_loc_dict
        self.initialize_model(load_model)

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
        return int(elev_val / ((np.pi / 2) / (qlp.ELEVATION_CARDINALITY - 1)))

    def get_az_idx(self, az_val):
        return int((az_val + np.pi) / (2 * np.pi / (qlp.AZIMUTH_CARDINALITY - 1)))

    def get_dist_idx(self, dist_val):
        return int(dist_val * qlp.DISTANCES_CARDINALITY / self.max_dist)

    def update_state_idxs(self, v_stats):
        self.last_state_idxs = self.state_idxs
        payloads, elevs_r_rx, azs_r_rx, dists_r_rx, elevs_r_tx, azs_r_tx, dists_r_tx = v_stats
        max_payload_idx = payloads.argmax()
        elev_r_max_pyl_idx = self.get_elev_idx(elevs_r_rx[max_payload_idx])
        elev_t_max_pyl_idx = self.get_elev_idx(elevs_r_tx[max_payload_idx])
        az_r_max_pyl_idx = self.get_az_idx(azs_r_rx[max_payload_idx])
        az_t_max_pyl_idx = self.get_az_idx(azs_r_tx[max_payload_idx])
        elev_median = self.get_elev_idx(np.median(np.concatenate((elevs_r_rx, elevs_r_tx))))
        az_median = self.get_az_idx(np.median(np.concatenate((azs_r_rx, azs_r_tx))))
        d_mean = self.get_dist_idx(np.mean(np.concatenate((dists_r_rx, dists_r_tx))))
        self.state_idxs = elev_r_max_pyl_idx, az_r_max_pyl_idx, elev_t_max_pyl_idx, \
                          az_t_max_pyl_idx, elev_median, az_median, d_mean

    def save_checkpoint(self, folder_name=qlp.CHECKPOINTS_FILE):
        file_name = os.path.join(folder_name, self.__repr__())
        np.save(file_name, self.q_values)
        file_name_cycle_idx = os.path.join(qlp.CHECKPOINTS_FILE, self.__repr__() + '_cycle_idx' + '.npy')
        np.save(file_name_cycle_idx, self.cycle_idx)

    def save_rewards_history(self, folder_name=qlp.REWARDS_FILE):
        file_name = os.path.join(folder_name, self.__repr__())
        with NpyAppendArray(file_name) as npaa:
            npaa.append(np.array([self.rewards_history]))

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
        return f'QManager_V1_{qlp.CHECKPOINT_ID if not id else id}'

    def select_action(self, t_step, q_set=None, testing=False):
        q_values = self.q_values[0][self.state_idxs] + self.q_values[1][self.state_idxs] if q_set is None else \
            self.q_values[q_set][self.state_idxs]
        self.available_action_idxs[:] = 1
        dbs = self.dbs_list[0]
        loc_size_shift = self.loc_space_size * np.arange(self.orientation_space_size)
        for idx, action in self.d_loc_dict.items():
            new_coord = dbs.coords + action * t_step * dbs.speed
            if not new_coord.in_boundary(*self.boundaries):
                self.available_action_idxs[idx + loc_size_shift] = 0
        selected_action_idx = self.select_action_from_qs(q_values[self.available_action_idxs == 1], testing)
        complete_action_idx = np.arange(self.action_space_size)[self.available_action_idxs == 1][selected_action_idx]
        orientation_idx = complete_action_idx // self.loc_space_size
        loc_action_idx = complete_action_idx % self.loc_space_size
        selected_action = self.d_loc_dict[loc_action_idx], d_orient_dict[orientation_idx]
        self.selected_action = selected_action
        self.last_action_idx = orientation_idx * self.loc_space_size + loc_action_idx
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

    def select_action_from_qs(self, q_values, testing=False):
        if not testing and decision(QManager.get_exploration_probability(self.cycle_idx)):
            idx = rng.integers(q_values.shape[0])
        else:
            idx = rng.choice(np.where(q_values == q_values.max())[0])
        return idx

    def begin_cycle(self, t_step):
        print("Cycle idx:", self.cycle_idx)
        self.select_action(t_step)
        return self.selected_action

    def end_cycle(self, reward, t_step):
        if self.save_on:
            self.rewards_history[self.cycle_idx % self.save_on] = reward
        self.update_q_values(reward, t_step)
        self.cycle_idx += 1
        if self.save_on and not (self.cycle_idx % self.save_on):
            self.save_checkpoint()
            self.save_rewards_history()
