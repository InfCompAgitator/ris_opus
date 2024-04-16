"""Based on TR#: 38.901"""

import numpy as np
from src.parameters import *
from src.data_structures import Coords3d
from scipy.constants import speed_of_light
from src.math_tools import decision


class UmaCellular:
    @staticmethod
    def get_path_loss(ue_coords: Coords3d, bs_coords: Coords3d = Coords3d(0, 0, MBS_HEIGHT),
                      carrier_freq_in=DEFAULT_CARRIER_FREQ_MBS, los_probability=None):
        carrier_freq = carrier_freq_in/1e9
        # NOTE: For now neglecting distance inside buildings
        distance_3d_out = ue_coords.get_distance_to(bs_coords)
        distance_2d_out = ue_coords.get_distance_to(bs_coords, flag_2d=True)
        pl_oxy = 0
        if carrier_freq == 60:
            pl_oxy = distance_3d_out * 15 / 1e3

        distance_3d, distance_2d = distance_3d_out, distance_2d_out
        z_bs = 25
        z_ue = min(ue_coords.z, bs_coords.z)
        effective_env_height = UmaCellular.get_effective_environment_height(distance_2d, z_ue)
        effective_bs_height = z_bs - effective_env_height
        effective_ue_height = z_ue - effective_env_height

        breakpoint_distance = 4 * effective_bs_height * effective_ue_height * carrier_freq / speed_of_light

        # Make constant for margin calculations
        shadowing_los = rng.normal(4)

        if distance_2d <= breakpoint_distance:
            pl1 = 28 + 22 * np.log10(distance_3d) + 20 * np.log10(carrier_freq)
            path_loss_los = pl1 + shadowing_los
        else:
            pl2 = 28 + 40 * np.log10(distance_3d) + 20 * np.log10(carrier_freq) - 9 * np.log10(breakpoint_distance ** 2 + (z_bs - z_ue) ** 2)
            path_loss_los = pl2 + shadowing_los

        shadowing_nlos = rng.normal(6)
        pl_nlos_prime = 13.54 + 39.08 * np.log10(distance_3d) + 20*np.log10(carrier_freq) - 0.6 * (z_ue - 1.5)
        path_loss_nlos = pl_nlos_prime + shadowing_nlos

        if los_probability:
            return db2lin(los_probability * path_loss_los + (1 - los_probability) * path_loss_nlos + pl_oxy)

        plos = UmaCellular.get_los_probability(distance_2d_out, z_ue)
        return db2lin(plos * path_loss_los + (1 - plos) * path_loss_nlos + pl_oxy)

    @staticmethod
    def get_los_probability(distance_2d_out, ue_height):
        if distance_2d_out <= 18:
            return 1
        else:
            c_prime = 0 if ue_height <= 13 else ((ue_height-13)/10)**1.5
            plos = (18/distance_2d_out + np.exp(-distance_2d_out/63)*(1-18/distance_2d_out)) *\
                   (1+c_prime*5/4 * (distance_2d_out/100)**3 * np.exp(-distance_2d_out/150))
            return plos

    @staticmethod
    def get_effective_environment_height(distance_2d, user_height, vehicular=True):
        if vehicular:
            return 0.25
        # users height should be below 23m
        if user_height < 13:
            return 1
        elif distance_2d <= 18:
            return 1
        else:
            g = 5 / 4 * (distance_2d / 100) ** 3 * np.exp(-distance_2d / 150)
            _prob = g * ((user_height - 13) / 10) ** 1.5

        if decision(_prob):
            return 1
        else:
            return rng.choice(np.arange(12, user_height - 1.5, 3))


class UmiCellular:
    @staticmethod
    def get_path_loss(ue_coords: Coords3d, bs_coords: Coords3d = Coords3d(0, 0, MBS_HEIGHT),
                      carrier_freq_in=DEFAULT_CARRIER_FREQ_MBS, los_probability=None):
        assert (ue_coords.z >= 1.5)
        carrier_freq = carrier_freq_in/1e9
        # NOTE: For now neglecting distance inside buildings
        distance_3d_out = ue_coords.get_distance_to(bs_coords)
        distance_2d_out = ue_coords.get_distance_to(bs_coords, flag_2d=True)

        distance_3d, distance_2d = distance_3d_out, distance_2d_out

        effective_env_height = 1                                        ### TBC with Salim
        effective_bs_height = bs_coords.z - effective_env_height
        effective_ue_height = ue_coords.z - effective_env_height

        breakpoint_distance = 4 * effective_bs_height * effective_ue_height * carrier_freq / speed_of_light

        # Make constant for margin calculations
        shadowing_los = rng.normal(4)

        if distance_2d <= breakpoint_distance:
            pl1 = 32.4 + 21 * np.log10(distance_3d) + 20 * np.log10(carrier_freq)
            path_loss_los = pl1 + shadowing_los
        else:
            pl2 = 32.4 + 40 * np.log10(distance_3d) + 20 * np.log10(carrier_freq) - 9.5 * np.log10(
                breakpoint_distance ** 2 + (bs_coords.z - ue_coords.z) ** 2)
            path_loss_los = pl2 + shadowing_los

        shadowing_nlos = rng.normal(7.82)
        pl_nlos_prime = 35.3 * np.log10(distance_3d) + 22.4 + 21.3* np.log10(carrier_freq) - 0.3 * (ue_coords.z - 1.5)
        path_loss_nlos = pl_nlos_prime + shadowing_nlos

        if los_probability:
            return db2lin(los_probability * path_loss_los + (1 - los_probability) * path_loss_nlos)

        plos = UmiCellular.get_los_probability(distance_2d_out, ue_coords.z)
        return db2lin(plos * path_loss_los + (1 - plos) * path_loss_nlos)

    @staticmethod
    def get_los_probability(distance_2d_out, ue_height):

        if distance_2d_out <= 18:
            return 1
        else:
            plos = 18 / distance_2d_out + np.exp(-distance_2d_out / 63) * (1 - 18 / distance_2d_out)
            return plos