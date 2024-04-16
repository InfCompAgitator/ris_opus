from src.parameters import NUM_OF_USERS, TIME_STEP, N_MBS, USER_MOBILITY_SAVE_NAME, MBS_HEIGHT, MBS_LOCATIONS
from src.environment.obstacle_mobility_model import ObstaclesMobilityModel
from src.base_station import BaseStation
import numpy as np
from src.users import User
from src.data_structures import Coords3d
from src.types_constants import StationType


class SimulationController:
    area_boundaries = None

    def __init__(self):
        self.n_bs = N_MBS
        self.n_users = NUM_OF_USERS
        self.users = []
        self.base_stations = []
        self.user_model = None
        self.init_environment()

    def reset_users_model(self, load=True, n_users=NUM_OF_USERS):
        self.user_model = ObstaclesMobilityModel()
        self.area_boundaries = self.user_model.obstacles_objects.get_margin_boundary(False)

    def init_environment(self, generate_random_bs=True):
        self.reset_users_model()
        if generate_random_bs:
            self.generate_random_base_stations()
        elif MBS_LOCATIONS:
            for _loc in MBS_LOCATIONS:
                self.add_bs_station(_loc)

    def set_ues_base_stations(self, exclude_mbs=True):
        """Different list for different carrier frequencies"""
        # self.bs_rf_list = [_bs.rf_transceiver for _bs in self.base_stations[self.n_bs if exclude_mbs else 0:]]
        # all_freqs = [_bs.carrier_frequency for _bs in self.bs_rf_list]
        # available_freqs = set(all_freqs)
        # self.stations_list = []
        # for _freq in available_freqs:
        #     bs_list = []
        #     for idx, _freq_bs in enumerate(all_freqs):
        #         if _freq_bs == _freq:
        #             bs_list.append(self.bs_rf_list[idx])
        #     self.stations_list.append(bs_list)
        bs_rf_list = [_bs.rf_transceiver for _bs in self.base_stations]
        [_user.rf_transceiver.set_available_base_stations(bs_rf_list) for _user in self.users]

    def simulate_time_step(self, time_step=None):
        if not self.user_model.generate_model_step(time_step):
            print("No more simulation data")
            return

    def generate_random_base_stations(self):
        for base_id in range(self.n_bs):
            initial_coords = np.random.uniform(low=(self.area_boundaries[0][0], self.area_boundaries[1][0]),
                                               high=(self.area_boundaries[0][1], self.area_boundaries[1][1]), size=2)
            initial_coords = np.append(initial_coords, MBS_HEIGHT)
            self.base_stations.append(BaseStation(coords=Coords3d.from_array(initial_coords)))
            # while self.base_stations[-1].is_within_obstacle(self.user_model.get_obstacles()):
            #     initial_coords = np.random.uniform(low=self.min_xy, high=self.max_xy, size=2)
            #     initial_coords = np.append(initial_coords, MBS_HEIGHT)
            #     self.base_stations[-1].coords = Coords3d(initial_coords[0], initial_coords[1], initial_coords[2])

    def clear_bs(self):
        self.base_stations = []
        self.set_ues_base_stations()
        BaseStation.reset_ids()

    def add_bs_station(self, coords: Coords3d, station_type: StationType):
        base_station = BaseStation(coords=coords, station_type=station_type)
        self.base_stations.append(base_station)
        return base_station.id


if __name__ == "__main__":
    a = SimulationController()