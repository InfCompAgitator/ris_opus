from src.channel_model.ris_model import RIS, RadiationPatterns, Coords3d, RisLinkStats, lin2db, CARRIER_FREQUENCY
from src.drone_agent import DroneAgent
from src.environment.vehicular_streets import Lane, LanesController, Vehicle
from src.parameters import TX_POWER_VEHICLE, TIME_STEP, rng
import numpy as np
from src.parameters import QLearningParams as qlp
from src.channel_model.mmwave_modeling import get_throughput_5g
from src.scheduling import ActiveLink, Pfs
from src.machine_learning.q_manager import QManager as Q_1
from src.machine_learning.q_manager_no_medians import QManager_v1 as Q_1_no_median
from src.machine_learning.q_manager_z_medians import QManager_v2 as Q_2_medians_z
from src.channel_model.v2v import ThreegppModel_H, ThreegppModel_U

V2V_INIT_LAMBDA = 0.05
V2I_INIT_LAMBDA = 0.04

Q_MANAGER_ID = qlp.Q_MANAGER_ID


def q_managers_dict(x):
    return {
        '1': Q_1, '2': Q_1_no_median, '3': Q_2_medians_z
    }.get(x, Q_1)


class PercomSim:
    lanes = []
    ris_list = []
    dbs_list = []
    q_manager_initialized = False
    previous_payload_ratio = 0
    def __init__(self, q_manager_id=Q_MANAGER_ID):
        y_length = 1000
        x_length = 250
        self.dbs_list.append(DroneAgent(Coords3d(x_length / 2, y_length / 2, 100)))
        self.lanes_ctrlr = LanesController()
        self.lanes_ctrlr.append_rsu(Coords3d(x_length / 2, y_length / 2, 20))
        self.lanes_ctrlr.append_lane(Coords3d(x_length, 0, 0), Coords3d(x_length, y_length, 0))
        self.lanes_ctrlr.append_lane(Coords3d(0, 0, 0), Coords3d(0, y_length, 0))
        _center = Coords3d(-y_length + x_length / np.sqrt(2), y_length / 2, 0)
        _begin = Coords3d(0, 0, 0)
        self.lanes_ctrlr.append_curved_lane(_begin, Coords3d(10, y_length, 0), _center, np.pi / 8, -np.pi / 4)
        self.pfs = Pfs()
        max_lane_dist = max([l.distance for l in self.lanes_ctrlr.lanes])
        boundaries = Coords3d(0, 0, 50), Coords3d(x_length, y_length, 600)
        max_lane_dist = np.sqrt(boundaries[1].z**2 + max_lane_dist**2 + x_length**2)
        self.q_manager = q_managers_dict(q_manager_id)(self.dbs_list, max_lane_dist, boundaries)

    def simulate_n_of_v2x_events(self, t_step):
        mu_i = t_step * V2I_INIT_LAMBDA
        mu_v = t_step * V2V_INIT_LAMBDA
        return rng.poisson(mu_v), rng.poisson(mu_i)

    def simulate_time_step(self, t_step=TIME_STEP):
        if self.pfs.active_pairs:
            self.pfs.update_links(t_step)
        if self.pfs.active_pairs:
            payload_ratio_1, payload_ratio_2 = self.pfs.get_stats()
            reward = 10 * (payload_ratio_1 - payload_ratio_2)
            # reward = 10 * payload_ratio_1
        self.lanes_ctrlr.simulate_time_step(t_step)
        self.dbs_list[0].move(t_step)
        # One DBS for now
        for _pair in self.pfs.active_pairs:
            link_stats, ris_pl = self.get_pl_ris(_pair.t_1, _pair.t_2, self.dbs_list[0])
            self.pfs.update_link_rate(_pair, ris_pl)
            _pair.stats = link_stats
        # Qlearning
        if self.pfs.active_pairs:
            print("DBS coords:", self.dbs_list[0].coords)
            self.update_q_manager_states()
            if self.q_manager_initialized:
                self.q_manager.end_cycle(reward, t_step)
            self.q_manager_initialized = True
            new_action = self.q_manager.begin_cycle(t_step)
            self.update_dbs_action(0, new_action)
            print("New action:", new_action, "Previous reward:", reward)

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
        self.pfs.schedule_slots(t_step)

    def update_dbs_action(self, dbs_idx, action):
        self.dbs_list[dbs_idx].d_orientation = action[1]
        self.dbs_list[dbs_idx].d_location = action[0]

    @staticmethod
    def get_pl_ris(v1: Vehicle, v2: Vehicle, dbs: DroneAgent):
        stats, pl = dbs.ris.get_path_loss_beamforming(v1.transceiver, v2.transceiver, get_stats=True)
        # print("Path loss:", lin2db(pl))
        return stats, pl

    def get_v2x_pairs_stats(self):
        payloads = np.array([p.payload for p in self.pfs.active_pairs])
        dists_r_tx, elevs_r_rx, elevs_r_tx, dists_r_rx, azs_r_rx, azs_r_tx = np.zeros_like(payloads), np.zeros_like(
            payloads), np.zeros_like(payloads), np.zeros_like(payloads), np.zeros_like(payloads), np.zeros_like(
            payloads)
        total_stats = [p.stats for p in self.pfs.active_pairs]

        for idx, _stat in enumerate(total_stats):
            elevs_r_rx[idx], azs_r_rx[idx], dists_r_rx[idx], elevs_r_tx[idx], azs_r_tx[idx], dists_r_tx[idx] = _stat
        return payloads, elevs_r_rx, azs_r_rx, dists_r_rx, elevs_r_tx, azs_r_tx, dists_r_tx

    def update_q_manager_states(self):
        self.q_manager.update_state_idxs(self.get_v2x_pairs_stats())


if __name__ == '__main__':
    per = PercomSim()

    per.lanes_ctrlr.generate_plot(per.dbs_list)
    per.simulate_time_step(1)
    while (1):
        per.simulate_time_step(1)
        payloads = per.get_v2x_pairs_stats()[0]
        if payloads.size > 0:
            print(payloads / 1e6)
