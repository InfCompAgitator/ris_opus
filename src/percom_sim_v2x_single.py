from src.channel_model.ris_model import RIS, RadiationPatterns, Coords3d, RisLinkStats, lin2db, CARRIER_FREQUENCY
from src.drone_agent import DroneAgent
from src.environment.vehicular_streets import Lane, LanesController, Vehicle
from src.parameters import TX_POWER_VEHICLE, TIME_STEP, rng, MAX_DIST_FROM_DBS
import numpy as np
from src.parameters import QLearningParams as qlp
from src.channel_model.mmwave_modeling import get_throughput_5g
from src.scheduling import ActiveLink, Pfs
from src.machine_learning.q_manager import QManager as Q_1
from src.machine_learning.q_manager_no_medians import QManager_v1 as Q_1_no_median
from src.machine_learning.q_manager_z_medians import QManager_v2 as Q_2_medians_z
from src.machine_learning.q_manager_v2x_single import QManager as Q_1_single
from src.machine_learning.q_manager_v2x_single_plus_azimuth import QManager_v2_s as Q_2_single
from src.machine_learning.q_manager_v2x_single_plus_azimuth_temporal import QManager_v2_s as Q_2_single_t
from src.machine_learning.q_manager_v2x_single_plus_azimuth_distance import QManager_v3_s as Q_3_single
from src.machine_learning.q_manager_v2x_single_plus_azimuth_distance_elev import QManager_v4_s as Q_4_single
from src.machine_learning.q_manager_v2x_single_az_meanelev_d import QManager_v5_s as Q_5_single
from src.machine_learning.q_manager_v2x_single_az_meanelev_d_d_t import QManager_v5_dist_t as QManager_v5_dist_t
from src.channel_model.v2v import ThreegppModel_H, ThreegppModel_U
import os

# V2V_INIT_LAMBDA = 0.02
# V2I_INIT_LAMBDA = 0.01

V2V_INIT_LAMBDA = 0.01
V2I_INIT_LAMBDA = 0.005

Q_MANAGER_ID = qlp.Q_MANAGER_ID


def q_managers_dict(x):
    return {
        '1': Q_1_single, '2': Q_2_single, '2_t': Q_2_single_t, '3': Q_3_single, '4': Q_4_single, '5': Q_5_single,
        '6': QManager_v5_dist_t
    }.get(x, Q_1_single)


class PercomSim:
    lanes = []
    ris_list = []
    dbs_list = []
    q_manager_initialized = False
    previous_payload_ratio = 0

    def __init__(self, q_manager_id=Q_MANAGER_ID):
        y_length = 5000
        x_length = 500
        self.dbs_list.append(DroneAgent(Coords3d(x_length / 2, 500, 200)))
        self.lanes_ctrlr = LanesController()
        self.lanes_ctrlr.append_rsu(Coords3d(x_length / 2, y_length / 2, 20))
        self.lanes_ctrlr.append_lane(Coords3d(x_length, 0, 0), Coords3d(x_length, y_length, 0))
        self.lanes_ctrlr.append_lane(Coords3d(0, y_length, 0), Coords3d(0, 0, 0))
        _center = Coords3d(-y_length + x_length / np.sqrt(2), y_length / 2, 0)
        _begin = Coords3d(0, 0, 0)
        # self.lanes_ctrlr.append_curved_lane(_begin, Coords3d(10, y_length, 0), _center, np.pi / 8, -np.pi / 4)
        self.pfs = Pfs()
        max_lane_dist = max([l.distance for l in self.lanes_ctrlr.lanes])
        boundaries = Coords3d(0, 0, 100), Coords3d(x_length, y_length, 600)
        max_lane_dist = np.sqrt(boundaries[1].z ** 2 + max_lane_dist ** 2 + x_length ** 2)
        self.q_manager = q_managers_dict(q_manager_id)(self.dbs_list, max_lane_dist, boundaries)
        self.per_vehicle_reward = np.zeros(10000)
        self.per_vehicle_reward_avg = np.zeros(10000)
        self.per_vehicle_time = 0
        self.per_vehicle_reward_idx = 0
        self.n_vs = 0

    def simulate_n_of_v2x_events(self, t_step):
        mu_i = t_step * V2I_INIT_LAMBDA
        mu_v = t_step * V2V_INIT_LAMBDA
        return rng.poisson(mu_v), rng.poisson(mu_i)

    def simulate_time_step(self, t_step=TIME_STEP):
        if self.pfs.single_pair:
            self.pfs.update_links(t_step)
            # payload_ratio_1, payload_ratio_2 = self.pfs.get_stats()
            # reward = 10 * (payload_ratio_1 - payload_ratio_2)
            # reward = 10 * payload_ratio_1
        if self.pfs.single_pair:
            target = (self.pfs.single_pair.t_1.coords + self.pfs.single_pair.t_2.coords)/2
            target.z = min(np.sqrt(2) * self.pfs.single_pair.t_1.coords.get_distance_to(self.pfs.single_pair.t_2.coords), self.q_manager.boundaries[1].z)
            self.dbs_list[0].move(t_step, target, qlp.BENCHLINE, boundaries=self.q_manager.boundaries)
        else:
            self.dbs_list[0].move(t_step, boundaries=self.q_manager.boundaries)
        self.lanes_ctrlr.simulate_time_step(t_step)
        if self.pfs.single_pair:
            reward = self.update_pairs_stats()
            self.per_vehicle_reward[self.per_vehicle_time] = reward

        # Qlearning
        if self.pfs.single_pair:
            #
            #
            #
            # ("DBS coords:", self.dbs_list[0].coords)
            self.update_q_manager_states()
            if self.q_manager_initialized:
                self.q_manager.end_cycle(reward, t_step)
            self.q_manager_initialized = True
            new_action = self.q_manager.begin_cycle(t_step)
            self.update_dbs_action(0, new_action)
            # print("New action:", new_action, "Previous reward:", reward)

        # Add new pairs if any
        n_v2v, n_v2i = self.simulate_n_of_v2x_events(t_step)
        for i in range(n_v2v):
            v1, v2, blocker = self.lanes_ctrlr.select_v2v_pair()
            if not v1 or not v2:
                break
            link_stats, ris_pl = self.get_pl_ris(v1, v2, self.dbs_list[0])
            direct_power = TX_POWER_VEHICLE / ThreegppModel_H.get_path_loss_v2v(v1.coords, v2.coords,
                                                                                blocker.coords if blocker else None)
            self.pfs.add_pair(v1, v2, link_stats, ris_pl, direct_power)

        for i in range(n_v2i):
            v1, v2 = self.lanes_ctrlr.select_v2i_pair()
            if not v1 or not v2:
                break
            direct_power = TX_POWER_VEHICLE / ThreegppModel_H.get_path_loss_v2i(v1.coords, v2.coords, None)
            link_stats, ris_pl = self.get_pl_ris(v1, v2, self.dbs_list[0])
            self.pfs.add_pair(v1, v2, link_stats, ris_pl, direct_power)
        if self.pfs.active_pairs:
            first_pair = self.pfs.single_pair
            if self.pfs.single_pair is None or not self.pfs.single_pair.t_1.data_transfer_flag:
                self.select_single_pair(random=False)
                # self.select_single_pair(random=True)
            if self.pfs.single_pair is not first_pair:
                self.q_manager.prev_d_center = None
                self.q_manager_initialized = False
                if self.per_vehicle_time > self.per_vehicle_reward_idx:
                    self.per_vehicle_reward_avg[
                    self.per_vehicle_reward_idx:self.per_vehicle_time] = self.per_vehicle_reward[
                                                                         self.per_vehicle_reward_idx:self.per_vehicle_time]
                self.per_vehicle_reward_idx = max(self.per_vehicle_reward_idx, self.per_vehicle_time)
                self.per_vehicle_reward[
                self.per_vehicle_time:self.per_vehicle_reward_idx] = self.per_vehicle_reward_avg[
                                                                     self.per_vehicle_time:self.per_vehicle_reward_idx]
                if self.n_vs > 0:
                    self.per_vehicle_reward_avg = self.per_vehicle_reward_avg + (
                            self.per_vehicle_reward - self.per_vehicle_reward_avg) / self.n_vs
                self.per_vehicle_time = 0
                self.n_vs += 1
            else:
                self.per_vehicle_time += 1
        self.pfs.schedule_slots(t_step)
        if qlp.SAVE_ON is not None and self.q_manager.cycle_idx % qlp.SAVE_ON == 0:
            file_name = os.path.join(qlp.CHECKPOINTS_FILE, '_rewards_per_veh' + qlp.CHECKPOINT_ID)
            np.save(file_name, self.per_vehicle_reward_avg[:self.per_vehicle_reward_idx])

    def select_single_pair(self, random=False, dist_from_dbs=MAX_DIST_FROM_DBS):
        if dist_from_dbs and not random:
            poss_pairs = [_pair for _pair in self.pfs.active_pairs if self.dbs_list[0].coords.get_distance_to((_pair.t_1.coords + _pair.t_2.coords)/2) < dist_from_dbs]
        else:
            poss_pairs = self.pfs.active_pairs
        if len(poss_pairs) == 0:
            return
        if not random:
            if self.pfs.single_pair:
                self.pfs.single_pair.t_1.color = 'r'
                self.pfs.single_pair.t_2.color = 'r'
            if self.pfs.single_pair_2:
                self.pfs.single_pair_2.t_1.color = 'r'
                self.pfs.single_pair_2.t_2.color = 'r'
            single_pair = self.pfs.active_pairs[0]
            if len(self.pfs.active_pairs) == 1:
                single_pair.t_1.color = 'blue'
                single_pair.t_2.color = 'blue'
                self.pfs.single_pair = single_pair
                return
            single_pair_2 = self.pfs.active_pairs[1]

            min_distance = ((single_pair.t_1.coords + single_pair.t_2.coords) / 2).get_distance_to(
                self.dbs_list[0].coords)
            min_distance_2 = ((single_pair_2.t_1.coords + single_pair_2.t_2.coords) / 2).get_distance_to(
                self.dbs_list[0].coords)
            if min_distance_2 < min_distance:
                single_pair, single_pair_2 = single_pair_2, single_pair
                min_distance, min_distance_2 = min_distance_2, min_distance

            for i in range(2, len(self.pfs.active_pairs)):
                new_pair = self.pfs.active_pairs[i]
                new_distance = ((new_pair.t_1.coords + new_pair.t_2.coords) / 2).get_distance_to(
                    self.dbs_list[0].coords)
                if new_distance < min_distance:
                    single_pair, single_pair_2 = new_pair, single_pair
                    min_distance, min_distance_2 = new_distance, min_distance
                elif new_distance < min_distance_2:
                    single_pair_2 = new_pair
                    min_distance_2 = new_distance

            single_pair.t_1.color = 'blue'
            single_pair.t_2.color = 'blue'
            single_pair_2.t_1.color = 'y'
            single_pair_2.t_2.color = 'y'
            self.pfs.single_pair = single_pair
            self.pfs.single_pair_2 = single_pair_2
        else:
            if self.pfs.single_pair_2:
                active_pairs_not_selected = list(filter(lambda x: x != self.pfs.single_pair_2, self.pfs.active_pairs))
                self.pfs.single_pair = self.pfs.single_pair_2
                self.pfs.single_pair_2 = rng.choice(active_pairs_not_selected, 1, replace=False)[0] if len(active_pairs_not_selected) > 0 else None
            elif len(self.pfs.active_pairs) > 1:
                self.pfs.single_pair, self.pfs.single_pair_2 = rng.choice(self.pfs.active_pairs, 2, replace=False)
            else:
                self.pfs.single_pair = self.pfs.active_pairs[0]
            if self.pfs.single_pair is None:
                return
            self.pfs.single_pair.t_1.color = 'b'
            self.pfs.single_pair.t_2.color = 'b'

    def update_pairs_stats(self):
        link_stats, ris_pl = self.get_pl_ris(self.pfs.single_pair.t_1, self.pfs.single_pair.t_2, self.dbs_list[0])
        self.pfs.single_pair.prev_stats = self.pfs.single_pair.stats
        self.pfs.single_pair.stats = link_stats
        self.pfs.update_link_rate(self.pfs.single_pair, ris_pl)
        reward = -10 * lin2db(ris_pl)
        if self.pfs.single_pair_2:
            link_stats, ris_pl = self.get_pl_ris(self.pfs.single_pair_2.t_1, self.pfs.single_pair_2.t_2,
                                                 self.dbs_list[0])
            self.pfs.single_pair_2.prev_stats = self.pfs.single_pair_2.stats
            self.pfs.single_pair_2.stats = link_stats
            self.pfs.update_link_rate(self.pfs.single_pair, ris_pl)

        return reward

    def update_dbs_action(self, dbs_idx, action):
        self.dbs_list[dbs_idx].d_orientation = action[1]
        self.dbs_list[dbs_idx].d_location = action[0]

    @staticmethod
    def get_pl_ris(v1: Vehicle, v2: Vehicle, dbs: DroneAgent):
        stats, pl = dbs.ris.get_path_loss_beamforming(v1.transceiver, v2.transceiver, get_stats=True)
        # print("Path loss:", lin2db(pl))
        return stats, pl

    def update_q_manager_states(self):
        self.q_manager.update_state_idxs(self.pfs.single_pair.stats, self.pfs.single_pair.prev_stats,
                                         ((
                                                  self.pfs.single_pair.t_1.coords + self.pfs.single_pair.t_2.coords) / 2).get_distance_to(
                                             self.dbs_list[0].coords))


if __name__ == '__main__':
    per = PercomSim()
    average_rate_improvement = 0
    per.lanes_ctrlr.generate_plot(per.dbs_list)
    per.simulate_time_step(5)
    pair_idx = 0
    for i in range(1, int(6e6)):
        per.simulate_time_step(1)
        if per.pfs.single_pair:
            pair_idx += 1
            average_rate_improvement += (per.pfs.single_pair.rate_improvement - average_rate_improvement)/pair_idx
            print(average_rate_improvement)
