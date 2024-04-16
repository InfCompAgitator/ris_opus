from src.channel_model.ris_model import RIS, RadiationPatterns, Coords3d, RisLinkStats, lin2db, CARRIER_FREQUENCY
from src.drone_agent import DroneAgent
from src.environment.vehicular_streets import Lane, LanesController, Vehicle
from src.parameters import TX_POWER_VEHICLE, TIME_STEP, rng, MAX_DIST_FROM_DBS
import numpy as np
from src.parameters import QLearningParams as qlp
from src.channel_model.mmwave_modeling import get_throughput_5g
from src.scheduling_orientation import ActiveLink, Pfs
from src.machine_learning.orientation_tylko.q_manager_v2x_single_orientation import QManager as QManager_1
from src.machine_learning.orientation_tylko.q_manager_v2x_single_orientation_d_orientations import QManager as QManager_2
from src.channel_model.v2v import ThreegppModel_H, ThreegppModel_U
import os
from numpy import arctan, cos
from scipy.optimize import minimize


def get_optimal_height(d, H_initial, H_bounds):
    def f(H, d):
        return np.float64((d ** 2 + H ** 2) ** 2 / (cos(arctan(d / H,dtype=np.float64),dtype=np.float64) ** 6))

    result = minimize(lambda H: np.float64(f(H, d)), H_initial, method='L-BFGS-B', bounds=[H_bounds])
    return result.x


# V2V_INIT_LAMBDA = 0.02
# V2I_INIT_LAMBDA = 0.01

V2V_INIT_LAMBDA = 0.06
V2I_INIT_LAMBDA = 0.01

Q_MANAGER_ID = qlp.Q_MANAGER_ID


def q_managers_dict(x):
    return {
        '1': QManager_1, '2': QManager_2
    }.get(x, QManager_1)


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
        self.lanes_ctrlr.append_rsu(Coords3d(x_length / 2, y_length / 2, 25))
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
        self.pfs.selected_pairs = [None] * qlp.N_SELECTED_PAIRS
        self.start_cycle_idx = int(self.q_manager.cycle_idx)

    def simulate_n_of_v2x_events(self, t_step):
        mu_i = t_step * V2I_INIT_LAMBDA
        mu_v = t_step * V2V_INIT_LAMBDA
        return rng.poisson(mu_v), rng.poisson(mu_i)

    def add_new_v2x_pairs(self, t_step):
        # Add new pairs if any
        n_v2v, n_v2i = self.simulate_n_of_v2x_events(t_step)
        for i in range(n_v2v):
            v1, v2, blocker = self.lanes_ctrlr.select_v2v_pair()
            if not v1 or not v2:
                break
            link_stats, ris_pl = self.get_pl_ris(v1, v2, self.dbs_list[0])
            direct_power = TX_POWER_VEHICLE / ThreegppModel_H.get_path_loss_v2v(v1.coords, v2.coords,
                                                                                blocker.coords if blocker else None)
            link = self.pfs.add_pair(v1, v2, link_stats, ris_pl, direct_power, path_loss_function=ThreegppModel_H.get_path_loss_v2v, blocker=blocker)

        for i in range(n_v2i):
            v1, v2 = self.lanes_ctrlr.select_v2i_pair()
            if not v1 or not v2:
                break
            direct_power = TX_POWER_VEHICLE / ThreegppModel_H.get_path_loss_v2i(v1.coords, v2.coords, None)
            link_stats, ris_pl = self.get_pl_ris(v1, v2, self.dbs_list[0])
            link = self.pfs.add_pair(v1, v2, link_stats, ris_pl, direct_power, path_loss_function=ThreegppModel_H.get_path_loss_v2i,  blocker=None)

    def move_dbs(self, t_step, orientation=None):
        if orientation is None:
            self.dbs_list[0].d_orientation = 0
            if all(v is not None for v in self.pfs.selected_pairs):
                target = (self.pfs.selected_pairs[0].t_1.coords + self.pfs.selected_pairs[0].t_2.coords) / 2
                if qlp.BENCHLINE:
                    dist = 0
                    for _pair in self.pfs.selected_pairs:
                        dist += _pair.t_1.coords.get_distance_to(_pair.t_2.coords)
                    target.z = get_optimal_height(dist / len(self.pfs.selected_pairs), self.dbs_list[0].coords.z,
                                                  (self.q_manager.boundaries[0].z, self.q_manager.boundaries[1].z))[0]

                else:
                    dist = 0
                    for _pair in self.pfs.selected_pairs:
                        dist += _pair.t_1.coords.get_distance_to(_pair.t_2.coords)
                    target.z = get_optimal_height(dist / len(self.pfs.selected_pairs), self.dbs_list[0].coords.z,
                                                  (self.q_manager.boundaries[0].z, self.q_manager.boundaries[1].z))[0]
                self.dbs_list[0].move(t_step, target, True, boundaries=self.q_manager.boundaries)
        else:
            self.dbs_list[0].d_orientation = orientation
            self.dbs_list[0].move(t_step, None, False, boundaries=self.q_manager.boundaries)

    def simulate_time_step(self, t_step=TIME_STEP):
        self.move_dbs(t_step, None)
        reward_0 = self.update_pairs_stats()
        if all(v is not None for v in self.pfs.selected_pairs):
            print("Cycle idx:", self.q_manager.cycle_idx - self.start_cycle_idx)
            if not qlp.BENCHLINE:
                self.update_q_manager_states()
                self.q_manager_initialized = True
                new_action = self.q_manager.begin_cycle(t_step)
                d_orientation = new_action[1]
                self.dbs_list[0].d_location = new_action[0]
                self.move_dbs(t_step, d_orientation)
            else:
                self.q_manager.cycle_idx += 1

        if self.pfs.selected_pairs[0] is not None:
            reward = self.update_pairs_stats()
            self.update_q_manager_states()
            self.per_vehicle_reward[self.per_vehicle_time] = reward
            if not qlp.BENCHLINE:
                # Qlearning
                if self.q_manager_initialized:
                    self.q_manager.end_cycle(reward - reward_0, t_step)
        self.add_new_v2x_pairs(t_step)

        if self.pfs.active_pairs:
            self.update_selected_pairs(t_step)

        if self.pfs.selected_pairs[0] is not None:
            self.pfs.schedule_slots(t_step)
            self.pfs.update_links(t_step)

        self.lanes_ctrlr.simulate_time_step(t_step)

        if qlp.SAVE_ON is not None and self.q_manager.cycle_idx % qlp.SAVE_ON == 0:
            file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + qlp.CHECKPOINT_ID)
            np.save(file_name, self.per_vehicle_reward_avg[:self.per_vehicle_reward_idx])

    def update_selected_pairs(self, t_step):
        first_pair = self.pfs.selected_pairs[0]
        if any(v is None for v in self.pfs.selected_pairs):
            self.select_single_pair(random=False)

        if self.pfs.selected_pairs[0] is not first_pair:
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
            self.per_vehicle_reward = np.zeros(10000)
        else:
            self.per_vehicle_time += 1
    def select_single_pair(self, random=False, dist_from_dbs=MAX_DIST_FROM_DBS):
        if dist_from_dbs and not random:
            pairs_dist = np.array(
                [self.dbs_list[0].coords.get_distance_to((_pair.t_1.coords + _pair.t_2.coords) / 2) for _pair in
                 self.pfs.active_pairs])
            poss_pairs = [item for item, flag in zip(self.pfs.active_pairs, pairs_dist < dist_from_dbs) if flag]
            pairs_dist = np.array([item for item in pairs_dist if item < dist_from_dbs])
        else:
            pairs_dist = np.array(
                [self.dbs_list[0].coords.get_distance_to((_pair.t_1.coords + _pair.t_2.coords) / 2) for _pair in
                 self.pfs.active_pairs])
            poss_pairs = self.pfs.active_pairs
        if len(poss_pairs) == 0:
            self.pfs.selected_pairs = [None] * qlp.N_SELECTED_PAIRS
            return
        if not random:
            min_pair_idx = pairs_dist.argmin()
            self.pfs.selected_pairs[0] = poss_pairs[min_pair_idx]
            other_pairs_dists = np.array([((poss_pairs[min_pair_idx].t_1.coords + poss_pairs[
                min_pair_idx].t_2.coords) / 2).get_distance_to((_pair.t_1.coords + _pair.t_2.coords) / 2) for _pair in
                                          self.pfs.active_pairs])
            sorted_indices = np.argsort(other_pairs_dists)
            for pair in self.pfs.selected_pairs:
                if pair is not None:
                    pair.t_1.color = 'r'
                    pair.t_2.color = 'r'
            last_pair_idx = 0
            for i in sorted_indices[1:qlp.N_SELECTED_PAIRS + 1]:
                last_pair_idx += 1
                if last_pair_idx >= qlp.N_SELECTED_PAIRS:
                    break
                self.pfs.selected_pairs[last_pair_idx] = self.pfs.active_pairs[i]

            self.pfs.selected_pairs[last_pair_idx:] = [poss_pairs[min_pair_idx]] * (
                    qlp.N_SELECTED_PAIRS - last_pair_idx)
        else:
            self.pfs.selected_pairs = rng.choice(self.pfs.active_pairs, qlp.N_SELECTED_PAIRS, replace=False)
        for idx, pair in enumerate(self.pfs.selected_pairs):
            if pair is not None:
                if idx == 0:
                    pair.t_1.color = 'b'
                    pair.t_2.color = 'b'
                else:
                    pair.t_1.color = 'y'
                    pair.t_2.color = 'y'

    def update_pairs_stats(self):
        reward = 0
        for idx, _pair in enumerate(self.pfs.selected_pairs):
            if _pair is None:
                continue
            link_stats, ris_pl = self.get_pl_ris(_pair.t_1, _pair.t_2, self.dbs_list[0])
            _pair.prev_stats = _pair.stats
            _pair.stats = link_stats
            self.pfs.update_link_rate(_pair, ris_pl)
            if idx < qlp.N_Q_PAIRS:
                reward += -10 * lin2db(ris_pl)
        return reward

    @staticmethod
    def get_pl_ris(v1: Vehicle, v2: Vehicle, dbs: DroneAgent):
        stats, pl = dbs.ris.get_path_loss_beamforming(v1.transceiver, v2.transceiver, get_stats=True)
        # print("Path loss:", lin2db(pl))
        return stats, pl

    def update_q_manager_states(self):
        self.q_manager.update_state_idxs(self.pfs.selected_pairs, self.dbs_list[0].coords)


if __name__ == '__main__':
    per = PercomSim()
    average_rate_improvement = 0
    per.lanes_ctrlr.generate_plot(per.dbs_list)
    per.simulate_time_step(2)
    pair_idx = 0
    for i in range(1, int(6e6)):
        per.simulate_time_step(0.5)
        if per.pfs.selected_pairs[0] is not None:
            pair_idx += 1
            average_rate_improvement += (per.pfs.selected_pairs[
                                             0].rate_improvement - average_rate_improvement) / pair_idx
            print(average_rate_improvement)
