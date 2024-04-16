from src.parameters import *
import numpy as np
# from src.channel_model.rf_a2g import PlosModel
# from src.channel_model.rf_g2g import UmaCellular, UmiCellular
from src.data_structures import Coords3d
from src.types_constants import StationType
import types
from scipy.ndimage.interpolation import shift
from itertools import compress
from collections import namedtuple
from src.math_tools import get_rotation_matrix


class Orientation:
    def __init__(self, rotation_matrix, rotation_vector):
        self.rotation_matrix = rotation_matrix
        self.rotation_vector = rotation_vector

    def rotate(self, theta_rot):
        self.rotation_matrix = np.matmul(self.rotation_matrix, np.array(
            [[np.cos(theta_rot), -np.sin(theta_rot), 0], [np.sin(theta_rot), np.cos(theta_rot), 0], [0, 0, 1]]))

    def get_azimuth(self):
        new_xy = np.dot(self.rotation_matrix, np.array([1, 0, 0]))
        return np.arctan2(new_xy[0], new_xy[1])


RadiationPattern = namedtuple('radiation_pattern', ['name', 'gain'])


class RadiationPattern:
    cos_cube_elevation_only = RadiationPattern("cosine cube-elevation only", db2lin(9.03))  # From RIS modeling paper
    x_band_horn_elevation_only = RadiationPattern("cosine 62-elevation only", db2lin(21))  # From RIS modeling paper
    isotropic = RadiationPattern("isotropic", db2lin(0))  # From RIS modeling paper


class RfTransceiver:
    received_powers = []
    bs_sinr_cochannel_mask = None
    bs_cochannel_mask = None
    bs_list = []
    cochannel_bs_list = []
    sinr_cochannel_bs_list = []
    received_sinrs = []
    received_snrs = []
    received_powers = []
    serving_bss = []
    max_sinr_idx = None
    serving_bs = None
    serving_sinr = None
    serving_snr = None
    serving_rx_power = None
    orientation = Orientation(get_rotation_matrix(np.array([-1, 0, 0]), np.array([0, 0, -1])), np.array([-1, 0, 0]))

    def __init__(self, coords: Coords3d, t_power=DRONE_TX_POWER_RF, bandwidth=USER_BANDWIDTH, bs_id=None,
                 user_id=None, carrier_frequency=DEFAULT_CARRIER_FREQ_MBS,
                 radiation_pattern=None, vehicular=False):

        self.tx_power = t_power
        self.coords = coords
        self.noise_power = NOISE_POWER_RF
        self.bandwidth = bandwidth
        self.user_id = user_id
        self.bs_id = bs_id
        self.carrier_frequency = carrier_frequency
        self.get_path_loss = get_path_loss_function(StationType.UE)
        self.available_bandwidth = 0
        self.radiation_pattern = radiation_pattern
        if vehicular:
            self.orientation = Orientation(get_rotation_matrix(np.array([1, 0, 0]), np.array([0, 0, 1])),
                                           np.array([1, 0, 0]))

    def get_received_powers(self):
        for bs_idx, bs in enumerate(self.sinr_cochannel_bs_list):
            path_loss = bs.get_path_loss(bs.coords, self.coords, bs.carrier_frequency)
            self.received_powers[bs_idx] = (self.bandwidth / bs.bandwidth * bs.tx_power / path_loss)

    def get_received_sinrs(self):
        self.get_received_powers()
        for bs_idx, bs in enumerate(self.sinr_cochannel_bs_list):
            rx_powers = self.received_powers
            interferences = np.sum(rx_powers) - rx_powers[bs_idx]
            self.received_sinrs[bs_idx] = rx_powers[bs_idx] / (interferences + self.noise_power)
            self.received_snrs[bs_idx] = rx_powers[bs_idx] / self.noise_power

    def get_received_snrs(self):
        self.get_received_powers()
        for bs_idx, bs in enumerate(self.sinr_cochannel_bs_list):
            self.received_snrs[bs_idx] = self.received_powers[bs_idx] / self.noise_power

    def update_cochannel_bs_mask(self, frequency_condition=None):
        # Creates mask of stations list that keeps only stations with carrier frequency specified
        if frequency_condition is None:
            self.bs_cochannel_mask = np.ones(len(self.stations_list), dtype=bool)
        else:
            for idx, _station in enumerate(self.stations_list):
                if _station.carrier_frequency == frequency_condition:
                    self.bs_cochannel_mask[idx] = 1
        self.cochannel_bs_list = list(compress(self.stations_list, self.bs_cochannel_mask))

    def update_sufficient_sinr_cochannel_bs_mask(self, sinr_condition=DEFAULT_SINR_SENSITIVITY_LEVEL):
        # Creates mask of stations list that keeps only stations with carrier frequency specified and sufficient SINR
        self.bs_sinr_cochannel_mask = np.copy(self.bs_cochannel_mask)
        self.sinr_cochannel_bs_list = list(compress(self.stations_list, self.bs_cochannel_mask))

        rx_powers = np.zeros(len(self.stations_list))
        for bs_idx, bs in enumerate(self.stations_list):
            if not self.bs_cochannel_mask[bs_idx]:
                rx_powers[bs_idx] = 0
            path_loss = bs.get_path_loss(bs.coords, self.coords, bs.carrier_frequency)
            rx_powers[bs_idx] = (self.bandwidth / bs.bandwidth * bs.tx_power / path_loss)

        rx_sinr = np.zeros(len(self.stations_list))

        for bs_idx, bs in enumerate(self.stations_list):
            if not self.bs_cochannel_mask[bs_idx]:
                rx_powers[bs_idx] = 0
            interferences = np.sum(rx_powers) - rx_powers[bs_idx]
            rx_sinr[bs_idx] = rx_powers[bs_idx] / (interferences + self.noise_power)

        for idx, sinr in enumerate(rx_sinr):
            if sinr < sinr_condition:
                self.bs_sinr_cochannel_mask[idx] = False

        self.sinr_cochannel_bs_list = list(compress(self.stations_list, self.bs_sinr_cochannel_mask))
        self.received_sinrs = rx_sinr[self.bs_sinr_cochannel_mask]
        self.received_powers = rx_powers[self.bs_sinr_cochannel_mask]

    def set_available_base_stations(self, stations_list, sinr_condition=None, frequency_condition=None,
                                    skip_mask_updates=True):
        self.stations_list = stations_list
        if not skip_mask_updates:
            self.update_cochannel_bs_mask(frequency_condition)
            self.update_sufficient_sinr_cochannel_bs_mask(sinr_condition)

    def calculate_serving_bs(self, recalculate=False):
        """Return base_station ID, SINR, SNR, Received Power, Capacity"""
        if recalculate:
            self.get_received_sinrs()

        self.max_sinr_idx = self.received_sinrs.argmax()
        self.serving_bs = [self.sinr_cochannel_bs_list[self.max_sinr_idx]]
        self.serving_sinr = self.received_sinrs[self.max_sinr_idx]
        self.serving_snr = self.received_powers[self.max_sinr_idx] / self.noise_power
        self.serving_rx_power = self.received_powers[self.max_sinr_idx]

        return self.serving_bs.bs_id, self.received_sinr, self.received_snr, self.received_power

    def is_sinr_satisfied(self):
        return self.received_sinr >= DEFAULT_SINR_THRESHOLD

    def is_snr_satisfied(self):
        return self.received_snr >= DEFAULT_SNR_THRESHOLD


class UserRfTransceiver(RfTransceiver):
    def __init__(self, coords: Coords3d, t_power=DRONE_TX_POWER_RF, bandwidth=USER_BANDWIDTH, bs_id=None,
                 user_id=None, carrier_frequency=DEFAULT_CARRIER_FREQ_MBS):
        super().__init__(coords, t_power=t_power, bandwidth=bandwidth, bs_id=bs_id, user_id=user_id,
                         carrier_frequency=carrier_frequency)
        self.get_path_loss = None


class MacroRfTransceiver(RfTransceiver):
    def __init__(self, coords: Coords3d, t_power=DRONE_TX_POWER_RF, bandwidth=USER_BANDWIDTH, bs_id=None,
                 user_id=None, carrier_frequency=DEFAULT_CARRIER_FREQ_MBS, station_type=StationType.UMa,
                 macro_type=StationType.UMa):
        super().__init__(coords, t_power=t_power, bandwidth=bandwidth, bs_id=bs_id, user_id=user_id,
                         carrier_frequency=carrier_frequency)
        self.get_path_loss = None


def get_path_loss_function(station_type):
        return None


if __name__ == "__main__":
    rf1 = RfTransceiver(Coords3d(0, 1, 2))
    rf2 = RfTransceiver(Coords3d(5, 6, 7))
    rf3 = RfTransceiver(rf1.coords)
    rf1.orientation.get_azimuth()
