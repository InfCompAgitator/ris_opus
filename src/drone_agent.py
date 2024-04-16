from src.data_structures import Coords3d
from src.parameters import DRONE_TX_POWER_RF, UAV_SPEED, ROTATION_SPEED
from itertools import count
import numpy as np
from src.channel_model.ris_model import RIS, get_rotation_matrix, Orientation

SPATIAL_STEP_SIZE = 10  # 10 meters per simulation step


class DroneAgent:
    _ids = count(0)
    next_location = None
    start_location = None
    allowed_2d_states = None
    states_actions_mapping = None
    ris = None

    def __init__(self, coords: Coords3d, drone_id: int = None):
        self.id = next(self._ids) if drone_id is None else drone_id
        self.coords = coords
        self.start_location = coords.copy()
        self.initialize_ris()
        self.d_orientation = 0
        self.d_location = np.array([0, 0, 0])
        self.speed = UAV_SPEED
        self.rotation_speed = ROTATION_SPEED

    def initialize_ris(self):
        self.ris = RIS(self.coords)
        xy_rotation_vector = np.array([1, 1, 0])
        self.ris.orientation = Orientation(get_rotation_matrix(xy_rotation_vector / np.linalg.norm(xy_rotation_vector),
                                                               np.array([0, 0, -1])), xy_rotation_vector)

    def move(self, t_step, center_loc=None, move_to_center=False, boundaries=None):
        if center_loc is not None and move_to_center:
            direction = (center_loc - self.coords)
            direction_norm = direction.norm()
            if direction_norm > 1:
                direction /= direction.norm()
            else:
                print("Reached center!")
            new_coords = self.coords + direction * t_step * self.speed
            if new_coords.in_boundary(boundaries[0], boundaries[1]):
                self.coords.update_coords_from_array(new_coords)
        else:
            self.coords.update_coords_from_array(self.coords + self.d_location * t_step * self.speed)
        self.ris.orientation.rotate(self.rotation_speed * t_step * self.d_orientation)
        self.ris.update_elems_coords()
        self.d_orientation = 0
        self.d_location = np.array([0, 0])


if __name__ == '__main__':
    pass
