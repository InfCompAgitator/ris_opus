from src.data_structures import Coords3d
import numpy as np
from src.channel_model.ris_model import CARRIER_FREQUENCY
from src.parameters import rng
from src.math_tools import db2lin
from src.channel_model.rf_g2g import UmaCellular

class ThreegppModel_H:
    @staticmethod
    def get_los_probability(dist=None):
        # NLOSv in this case, highway
        if dist <= 475:
            return min(1, (2.1013e-6) * dist ** 2 - 0.002 * dist + 1.0193)
        else:
            return max(0, 0.54 - 0.001 * (dist - 475))

    @staticmethod
    def get_path_loss_los(dist):
        # highway
        pl_oxy = dist * 15 / 1e3
        shadowing = rng.normal(0, 3)
        pl = 32.4 + 20 * np.log10(dist) + 20 * np.log10(CARRIER_FREQUENCY / 1e9) + shadowing + pl_oxy
        return pl

    @staticmethod
    def get_path_loss_v2v(v1_coords: Coords3d, v2_coords: Coords3d, blocker=None):
        dist = v1_coords.get_distance_to(v2_coords)
        pr_los = ThreegppModel_H.get_los_probability(dist)
        pl_los = ThreegppModel_H.get_path_loss_los(dist)
        if blocker is None:
            return db2lin(pl_los)
        if min(v1_coords.z, v2_coords.z) > blocker.z:
            a_nlos = 0
        elif max(v1_coords.z, v2_coords.z) < blocker.z:
            a_nlos = rng.normal(9 + max(0, 15 * np.log10(dist) - 41), 4.5)
        else:
            a_nlos = rng.normal(9 + max(0, 15 * np.log10(dist) - 41), 4)
        return db2lin((pl_los + a_nlos) * (1 - pr_los) + pl_los * pr_los)

    @staticmethod
    def get_path_loss_v2i(v1_coords: Coords3d, v2_coords: Coords3d, blocker=None):
        return UmaCellular.get_path_loss(v1_coords, v2_coords, carrier_freq_in=CARRIER_FREQUENCY, los_probability=1)


class ThreegppModel_U:
    @staticmethod
    def get_los_probability(dist=None):
        return min(1, 1.05*np.exp(-0.0114*dist))

    @staticmethod
    def get_path_loss_los(dist):
        # highway
        pl_oxy = dist * 15 / 1e3
        shadowing = rng.normal(0, 3)
        pl = 32.4 + 20 * np.log10(dist) + 20 * np.log10(CARRIER_FREQUENCY / 1e9) + shadowing + pl_oxy
        return pl

    @staticmethod
    def get_path_loss_v2v(v1_coords: Coords3d, v2_coords: Coords3d, blocker=None):
        dist = v1_coords.get_distance_to(v2_coords)
        pr_los = ThreegppModel_U.get_los_probability(dist)
        pl_los = ThreegppModel_U.get_path_loss_los(dist)
        if blocker is None:
            return db2lin(pl_los)
        if min(v1_coords.z, v2_coords.z) > blocker.z:
            a_nlos = 0
        elif max(v1_coords.z, v2_coords.z) < blocker.z:
            a_nlos = rng.normal(9 + max(0, 15 * np.log10(dist) - 41), 4.5)
        else:
            a_nlos = rng.normal(9 + max(0, 15 * np.log10(dist) - 41), 4)
        return db2lin((pl_los + a_nlos) * (1 - pr_los) + pl_los * pr_los)

    @staticmethod
    def get_path_loss_v2i(v1_coords: Coords3d, v2_coords: Coords3d, blocker=None):
        return UmaCellular.get_path_loss(v1_coords, v2_coords, carrier_freq_in=CARRIER_FREQUENCY, los_probability=1
        if blocker is None else 0)
