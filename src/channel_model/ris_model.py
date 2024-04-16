import numpy as np
from src.math_tools import get_azimuth, get_elevation, sinc_fun, get_elevation_azimuth_antennas
from src.data_structures import Coords3d
from src.apparatus.rf_transciever import RfTransceiver, RadiationPattern, Orientation
from scipy.constants import c as speed_of_light
from src.math_tools import lin2db, db2lin, get_rotation_matrix
from src.environment.vehicular_streets import Lane
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple

RisLinkStats = namedtuple('ris_link_stats', ['r_rx_theta', 'r_rx_phi', 'd_rx', 'r_tx_theta', 'r_tx_phi', 'd_tx', 'center_dif'])
# Default RIS parameters LARGE RIS 1
N_ROWS = 100
N_COLUMNS = 102
DEFAULT_REFLECTION_COEF_A = 0.9
DEFAULT_REFLECTION_COEF_PHI = 0
DX = 0.01
DY = 0.01
CARRIER_FREQUENCY = 5e9  # Hz
WAVELENGTH = speed_of_light / CARRIER_FREQUENCY


# # Default RIS parameters SMALL RIS 1
# N_ROWS = 8
# N_COLUMNS = 32
# DEFAULT_REFLECTION_COEF_A = 0.7
# DEFAULT_REFLECTION_COEF_PHI = 0
# DX = 0.012
# DY = 0.012
# CARRIER_FREQUENCY = 60e9  # Hz
# WAVELENGTH = speed_of_light / CARRIER_FREQUENCY

class RIS:
    def __init__(self, location: Coords3d = Coords3d(0, 0, 0)):
        self.coords = location
        self.n_rows = N_ROWS
        self.n_columns = N_COLUMNS
        self.reflection_coefs = np.zeros((self.n_rows, self.n_columns, 2))
        self.reflection_coefs[:, :, 0] = DEFAULT_REFLECTION_COEF_A
        self.reflection_coefs[:, :, 1] = DEFAULT_REFLECTION_COEF_PHI
        self.identical_reflection_coefs = True
        self.near_field_distance = 2 * self.n_rows * self.n_columns * DX * DY / WAVELENGTH
        self.orientation = Orientation(get_rotation_matrix(np.array([0, 0, 1]), np.array([1, 0, 0])),
                                       np.array([0, 1, 0]))
        self.radiation_pattern = RadiationPattern.cos_cube_elevation_only
        xs_elem, ys_elem = np.meshgrid(
            np.arange(- DX * (self.n_columns // 2), DX * (self.n_columns // 2), DX),
            np.arange(- DY * (self.n_rows // 2), DY * (self.n_rows // 2), DY))
        self.elems_origin_coords = np.array(
            [(x, y, z) for x, y, z in zip(xs_elem.ravel(), ys_elem.ravel(), np.zeros(xs_elem.size))])
        self.elem_coords = self.elems_origin_coords.copy()
        self.update_elems_coords()

    def update_elems_coords(self):
        self.elem_coords = np.dot(self.elems_origin_coords, self.orientation.rotation_matrix.T) + self.coords
        return

    def get_path_loss_beamforming(self, tx_transceiver: RfTransceiver, rx_transceiver: RfTransceiver, get_stats=False):
        tx_ris_elevation, tx_ris_azimuth, ris_tx_elevation, ris_tx_azimuth = get_elevation_azimuth_antennas(
            *tx_transceiver.coords.np_array(),
            tx_transceiver.orientation.rotation_matrix,
            *self.coords.np_array(), self.orientation.rotation_matrix)
        rx_ris_elevation, rx_ris_azimuth, ris_rx_elevation, ris_rx_azimuth = get_elevation_azimuth_antennas(
            *rx_transceiver.coords.np_array(),
            rx_transceiver.orientation.rotation_matrix,
            *self.coords.np_array(), self.orientation.rotation_matrix)
        tx_antenna_gain = tx_transceiver.radiation_pattern.gain
        rx_antenna_gain = rx_transceiver.radiation_pattern.gain
        ris_antenna_gain = RadiationPattern.cos_cube_elevation_only.gain

        d_tx = tx_transceiver.coords.get_distance_to(self.coords)
        d_rx = rx_transceiver.coords.get_distance_to(self.coords)

        if min(d_rx, d_tx) < self.near_field_distance:
            # Near-field case
            path_loss_den = tx_antenna_gain * rx_antenna_gain * ris_antenna_gain * DX * DY * \
                            WAVELENGTH ** 2 * DEFAULT_REFLECTION_COEF_A ** 2 / (64 * np.pi ** 3)
            def get_value(loc):
                ris_loc = Coords3d.from_array(loc)
                tx_ris_elem_elevation, tx_ris_elem_azimuth, ris_tx_elem_elevation, ris_tx_elem_azimuth = get_elevation_azimuth_antennas(
                    *tx_transceiver.coords.np_array(),
                    tx_transceiver.orientation.rotation_matrix,
                    *ris_loc.np_array(),
                    self.orientation.rotation_matrix)
                rx_ris_elem_elevation, rx_ris_elem_azimuth, ris_rx_elem_elevation, ris_rx_elem_azimuth = get_elevation_azimuth_antennas(
                    *rx_transceiver.coords.np_array(),
                    rx_transceiver.orientation.rotation_matrix,
                    *ris_loc.np_array(),
                    self.orientation.rotation_matrix)
                ris_tx_elem_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=ris_tx_elem_elevation,
                                                                              radiation_pattern=self.radiation_pattern)
                ris_rx_elem_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=ris_rx_elem_elevation,
                                                                              radiation_pattern=self.radiation_pattern)
                tx_ris_elem_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=tx_ris_elem_elevation,
                                                                              radiation_pattern=tx_transceiver.radiation_pattern)
                rx_ris_elem_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=rx_ris_elem_elevation,
                                                                              radiation_pattern=rx_transceiver.radiation_pattern)
                d_elem_tx = ris_loc.get_distance_to(tx_transceiver.coords)
                d_elem_rx = ris_loc.get_distance_to(rx_transceiver.coords)
                return np.sqrt(
                    ris_tx_elem_pattern_gain * ris_rx_elem_pattern_gain * tx_ris_elem_pattern_gain * rx_ris_elem_pattern_gain) / (
                               d_elem_tx * d_elem_rx)
            def get_value_n(loc):
                ris_loc = Coords3d.from_array(loc)
                tx_ris_elem_elevation, tx_ris_elem_azimuth, ris_tx_elem_elevation, ris_tx_elem_azimuth = get_elevation_azimuth_antennas(
                    *tx_transceiver.coords.np_array(),
                    tx_transceiver.orientation.rotation_matrix,
                    *ris_loc.np_array(),
                    self.orientation.rotation_matrix)
                rx_ris_elem_elevation, rx_ris_elem_azimuth, ris_rx_elem_elevation, ris_rx_elem_azimuth = get_elevation_azimuth_antennas(
                    *rx_transceiver.coords.np_array(),
                    rx_transceiver.orientation.rotation_matrix,
                    *ris_loc.np_array(),
                    self.orientation.rotation_matrix)
                ris_tx_elem_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=ris_tx_elem_elevation,
                                                                              radiation_pattern=self.radiation_pattern)
                ris_rx_elem_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=ris_rx_elem_elevation,
                                                                              radiation_pattern=self.radiation_pattern)
                tx_ris_elem_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=tx_ris_elem_elevation,
                                                                              radiation_pattern=tx_transceiver.radiation_pattern)
                rx_ris_elem_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=rx_ris_elem_elevation,
                                                                              radiation_pattern=rx_transceiver.radiation_pattern)
                d_elem_tx = ris_loc.get_distance_to(tx_transceiver.coords)
                d_elem_rx = ris_loc.get_distance_to(rx_transceiver.coords)
                return np.sqrt(
                    ris_tx_elem_pattern_gain * ris_rx_elem_pattern_gain * tx_ris_elem_pattern_gain * rx_ris_elem_pattern_gain) * np.exp(-1j*(2*np.pi*(d_elem_tx + d_elem_rx))/WAVELENGTH) / (
                               d_elem_tx * d_elem_rx)

            vfunc = np.vectorize(get_value, signature='(3)->()')
            results = vfunc(self.elem_coords)
            path_loss_den = path_loss_den * np.abs(results.sum()) ** 2

        else:


            ris_tx_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=ris_tx_elevation,
                                                                     radiation_pattern=self.radiation_pattern)
            ris_rx_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=ris_rx_elevation,
                                                                     radiation_pattern=self.radiation_pattern)
            tx_ris_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=tx_ris_elevation,
                                                                     radiation_pattern=tx_transceiver.radiation_pattern)
            rx_ris_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=rx_ris_elevation,
                                                                     radiation_pattern=rx_transceiver.radiation_pattern)
            path_loss_den = tx_antenna_gain * rx_antenna_gain * ris_antenna_gain * self.n_columns ** 2 * self.n_rows ** 2 * DX * DY * \
                            WAVELENGTH ** 2 * ris_tx_pattern_gain * ris_rx_pattern_gain * DEFAULT_REFLECTION_COEF_A ** 2 / \
                            (64 * np.pi ** 3 * d_tx ** 2 * d_rx ** 2) * \
                            abs(sinc_fun(self.n_columns * np.pi / WAVELENGTH * (
                                    np.sin(ris_tx_elevation) * np.cos(ris_tx_azimuth) + np.sin(
                                ris_rx_elevation) * np.cos(
                                ris_rx_azimuth)) * DX) / \
                                sinc_fun(np.pi / WAVELENGTH * (
                                        np.sin(ris_tx_elevation) * np.cos(ris_tx_azimuth) + np.sin(
                                    ris_rx_elevation) * np.cos(
                                    ris_rx_azimuth)) * DX) * \
                                sinc_fun(self.n_rows * np.pi / WAVELENGTH * (
                                        np.sin(ris_tx_elevation) * np.sin(ris_tx_azimuth) + np.sin(
                                    ris_rx_elevation) * np.sin(
                                    ris_rx_azimuth)) * DY) / \
                                sinc_fun(np.pi / WAVELENGTH * (
                                        np.sin(ris_tx_elevation) * np.sin(ris_tx_azimuth) + np.sin(
                                    ris_rx_elevation) * np.sin(
                                    ris_rx_azimuth)) * DY)) ** 2
        if get_stats:
            return RisLinkStats(ris_rx_elevation, ris_rx_azimuth, d_rx, ris_tx_elevation, ris_tx_azimuth,
                                d_tx, ((tx_transceiver.coords+rx_transceiver.coords)/2) - self.coords), 1 / path_loss_den
        return 1 / path_loss_den


class RadiationPatterns:
    @staticmethod
    def get_pattern_gain(coords_src: Coords3d = None, coords_dest: Coords3d = None,
                         elevation_angle=None,
                         radiation_pattern: RadiationPattern = RadiationPattern.cos_cube_elevation_only):
        elevation_angle = elevation_angle if elevation_angle is not None else get_elevation(*coords_src.np_array(),
                                                                                            *coords_dest.np_array())
        if radiation_pattern == RadiationPattern.isotropic:
            return 1
        if 0 <= elevation_angle <= np.pi / 2:
            if radiation_pattern == RadiationPattern.cos_cube_elevation_only:
                return np.cos(elevation_angle) ** 3
            elif radiation_pattern == RadiationPattern.x_band_horn_elevation_only:
                return np.cos(elevation_angle) ** 62
        elif np.pi / 2 < elevation_angle <= np.pi:
            return 0
        else:
            raise ValueError("Invalid elevation angle!")


class FreeSpace:
    @staticmethod
    def get_path_loss(tx_transceiver: RfTransceiver, rx_transceiver: RfTransceiver, frequency=CARRIER_FREQUENCY):
        distance_3d = tx_transceiver.coords.get_distance_to(rx_transceiver.coords)
        tx_transceiver_gain = tx_transceiver.radiation_pattern.gain
        rx_transceiver_gain = rx_transceiver.radiation_pattern.gain
        tx_rx_elevation, _, rx_tx_elevation, _ = get_elevation_azimuth_antennas(*tx_transceiver.coords.np_array(),
                                                                                tx_transceiver.orientation.rotation_matrix,
                                                                                *rx_transceiver.coords.np_array(),
                                                                                rx_transceiver.orientation.rotation_matrix)
        rx_tx_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=rx_tx_elevation,
                                                                radiation_pattern=tx_transceiver.radiation_pattern)
        tx_rx_pattern_gain = RadiationPatterns.get_pattern_gain(elevation_angle=tx_rx_elevation,
                                                                radiation_pattern=rx_transceiver.radiation_pattern)

        return 20 * np.log10(4 * np.pi * frequency * distance_3d / (np.sqrt(
            rx_tx_pattern_gain * tx_rx_pattern_gain * tx_transceiver_gain * rx_transceiver_gain) * speed_of_light))


if __name__ == '__main__':
    lane_length = 1000
    lane_step = 20
    lane_height = 0
    dbs_height = 50
    ris_height = 0
    lane_1_start = Coords3d(10, 0, lane_height)
    lane_2_start = Coords3d(20, lane_length / 2, lane_height)
    lane_1_end = Coords3d(10, lane_length, lane_height)
    lane_2_end = Coords3d(lane_length, lane_length / 2, lane_height)
    dbs_loc = Coords3d(50, lane_length / 8, dbs_height)
    ris_1 = RIS(Coords3d(0, lane_length / 2, ris_height))
    dbs_transceiver = RfTransceiver(dbs_loc)
    dbs_transceiver.radiation_pattern = RadiationPattern.isotropic
    lane_1_transceiver = RfTransceiver(lane_1_start, vehicular=True)
    lane_1_transceiver.radiation_pattern = RadiationPattern.isotropic
    lane_2_transceiver = RfTransceiver(lane_2_start, vehicular=True)
    lane_2_transceiver.radiation_pattern = RadiationPattern.isotropic
    # rx_transceiver = RfTransceiver(Coords3d(50, 100, 10))
    # rx_transceiver.radiation_pattern = RadiationPattern.isotropic
    # tx_transceiver = RfTransceiver(Coords3d(100, 0, 10))
    # tx_transceiver.radiation_pattern = RadiationPattern.isotropic
    # pl = lin2db(ris_1.get_path_loss_beamforming(tx_transceiver, rx_transceiver))

    # rx_transceiver = RfTransceiver(Coords3d(10, 0, 10))
    # rx_transceiver.radiation_pattern = RadiationPattern.isotropic
    # tx_transceiver = RfTransceiver(Coords3d(10, 10, 10))
    # tx_transceiver.radiation_pattern = RadiationPattern.isotropic
    # pl = lin2db(ris_1.get_path_loss_beamforming(tx_transceiver, rx_transceiver))
    # pl_direct = FreeSpace.get_path_loss(rx_transceiver, tx_transceiver)

    lane_1 = Lane(lane_1_start, lane_1_end)
    lane_2 = Lane(lane_2_start, lane_2_end)

    lane_1_segs_pl_ris = np.zeros(lane_length // lane_step)
    lane_2_segs_pl_ris = np.zeros(lane_length // lane_step)
    lane_1_segs_pl_direct = np.zeros(lane_length // lane_step)
    lane_2_segs_pl_direct = np.zeros(lane_length // lane_step)

    lane_1_elevs = np.zeros(lane_length // lane_step)
    lane_2_elevs = np.zeros(lane_length // lane_step)
    lane_1_azimuths = np.zeros(lane_length // lane_step)
    lane_2_azimuths = np.zeros(lane_length // lane_step)
    lane_1_elevs_2 = np.zeros(lane_length // lane_step)
    lane_2_elevs_2 = np.zeros(lane_length // lane_step)
    lane_1_azimuths_2 = np.zeros(lane_length // lane_step)
    lane_2_azimuths_2 = np.zeros(lane_length // lane_step)

    for idx, _seg in tqdm(enumerate(np.arange(0, lane_length, lane_step))):
        lane_1_transceiver.coords = lane_1_transceiver.coords + lane_1.direction * lane_step
        lane_1_segs_pl_ris[idx] = lin2db(ris_1.get_path_loss_beamforming(dbs_transceiver, lane_1_transceiver))
        lane_1_segs_pl_direct[idx] = FreeSpace.get_path_loss(dbs_transceiver, lane_1_transceiver)

        lane_2_transceiver.coords = lane_2_transceiver.coords + lane_2.direction * lane_step
        lane_2_segs_pl_ris[idx] = lin2db(ris_1.get_path_loss_beamforming(dbs_transceiver, lane_2_transceiver))
        lane_2_segs_pl_direct[idx] = FreeSpace.get_path_loss(dbs_transceiver, lane_2_transceiver)
        lane_1_elevs[idx], lane_1_azimuths[idx], lane_1_elevs_2[idx], lane_1_azimuths_2[idx] = get_elevation_azimuth_antennas(
            *lane_1_transceiver.coords.np_array(),
            lane_1_transceiver.orientation.rotation_matrix,
            *ris_1.coords.np_array(), ris_1.orientation.rotation_matrix)
        lane_2_elevs[idx], lane_2_azimuths[idx], lane_2_elevs_2[idx], lane_2_azimuths_2[idx] = get_elevation_azimuth_antennas(
            *lane_2_transceiver.coords.np_array(),
            lane_2_transceiver.orientation.rotation_matrix,
            *ris_1.coords.np_array(), ris_1.orientation.rotation_matrix)

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, lane_length, lane_step), lane_1_segs_pl_ris, label='RIS lane 1')
    ax.plot(np.arange(0, lane_length, lane_step), lane_1_segs_pl_direct, label='Direct lane 1')

    ax.plot(np.arange(0, lane_length, lane_step), lane_2_segs_pl_ris, label='RIS lane 2')
    ax.plot(np.arange(0, lane_length, lane_step), lane_2_segs_pl_direct, label='Direct lane 2')
    ax.set_xlabel(f'DBS height{dbs_height}, RIS height{ris_height}')

    fig.legend()
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, lane_length, lane_step), np.degrees(lane_1_elevs), label='Lane 1 elevs')
    ax.plot(np.arange(0, lane_length, lane_step), np.degrees(lane_1_elevs_2), label='Lane 1 elevs RIS')
    ax.plot(np.arange(0, lane_length, lane_step), np.degrees(lane_2_elevs), label='Lane 2 elevs')
    ax.plot(np.arange(0, lane_length, lane_step), np.degrees(lane_2_elevs_2), label='Lane 2 elevs RIS')
    fig.legend()
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, lane_length, lane_step), np.degrees(lane_1_azimuths), label='Lane 1 azs')
    ax.plot(np.arange(0, lane_length, lane_step), np.degrees(lane_1_azimuths_2), label='Lane 1 azs RIS')
    ax.plot(np.arange(0, lane_length, lane_step), np.degrees(lane_2_azimuths), label='Lane 2 azs')
    ax.plot(np.arange(0, lane_length, lane_step), np.degrees(lane_2_azimuths_2), label='Lane 2 azs RIS')
    fig.legend()
    fig.show()
