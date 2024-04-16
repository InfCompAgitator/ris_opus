import numpy as np
from src.parameters import QLearningParams as qlp
from src.data_structures import Coords3d
from src.math_tools import decision
import itertools
import copy
import os
from src.parameters import rng, MAX_DIST_FROM_DBS
from npy_append_array import NpyAppendArray
from src.machine_learning.orientation_tylko.q_manager_v2x_single_orientation import QManager as QM
# Actions = orientation: +x, +y, +z, 0 3d loc: +x, +y, +z, -x, -y, -z, +x +y, -x +y, -x -y, +x -y,
# +x +z, -x +z, -x -z, +x -z, +y +z, -y +z, -y -z, +y -z,
# +x +y +z, +x +y -z, +x -y +z, +x -y -z,
# -x +y +z, -x +y -z, -x -y +z, -x -y -z, 0 = 3 * 27 = 81
# States prev_v_b_az, v_b_az, d_v_elev, d_v_az, d_b_elev, d_b_az
# Reward delta_payload/total_payload


d_orient_dict = {0: 0, 1: 1, 2: -1}


class QManager(QM):
    q_values = None
    state_idxs = np.zeros((qlp.N_Q_PAIRS * 4 + 1), dtype=int)
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
        self.q_shape = ((2, qlp.AZIMUTH_CARDINALITY, qlp.AZIMUTH_CARDINALITY, qlp.ELEVATION_CARDINALITY, qlp.ELEVATION_CARDINALITY,
                         self.action_space_size, self.action_space_size))
        self.initialize_model(load_model)
        self.d_orient_dict = d_orient_dict
        self.prev_d_center = None

    def update_state_idxs(self, selected_pairs, dbs_coords):
        self.last_state_idxs = self.state_idxs
        self.state_idxs = np.zeros((qlp.N_Q_PAIRS * 4 + 1), dtype=int)
        for idx, pair in enumerate(selected_pairs):
            if idx < qlp.N_Q_PAIRS:
                az_d_idx = self.get_az_idx((pair.stats.r_rx_phi % (2 * np.pi) - pair.stats.r_tx_phi % (2 * np.pi))% (2 * np.pi))
                prev_az_d_idx = self.get_az_idx((pair.prev_stats.r_rx_phi % (2 * np.pi) - pair.prev_stats.r_tx_phi % (2 * np.pi))% (2 * np.pi))

                elev_d_idx = self.get_elev_idx(abs(pair.stats.r_rx_theta - pair.stats.r_tx_theta))
                prev_elev_d_idx = self.get_elev_idx(abs(pair.prev_stats.r_rx_theta - pair.prev_stats.r_tx_theta))

                self.state_idxs[idx * 4: (idx+1)*4] = az_d_idx, prev_az_d_idx, elev_d_idx, prev_elev_d_idx
        self.state_idxs[-1] = self.last_action_idx
        self.state_idxs = tuple(self.state_idxs)

