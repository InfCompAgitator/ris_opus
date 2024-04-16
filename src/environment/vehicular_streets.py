import itertools

from src.data_structures import Coords3d
import numpy as np
from src.parameters import SIMULATION_TIME, TX_POWER_VEHICLE, TIME_STEP, rng
import heapq
from src.apparatus.rf_transciever import RfTransceiver, RadiationPattern, Orientation
from matplotlib.animation import FuncAnimation
import matplotlib;
from collections import namedtuple
import matplotlib.path as mpath
import matplotlib.patches as mpatches

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.math_tools import get_azimuth

LAMBDA_VAL = 0.1
VEH_SPEED_RANGE = [15, 15]
VEH_ANTENNA_HEIGHT = [1.5, 2]
MAX_LINK_DIST = 2000
MIN_LINK_DIST = 500
fig, ax = plt.subplots()
vehs_plot = []


def create_quadrotor_marker():
    # Path vertices and codes
    vertices = []
    codes = []
    # Define the cross
    cross_length = 8  # Length of each arm of the cross
    vertices += [(-cross_length / 2, 0), (cross_length / 2, 0), (0, 0), (0, -cross_length / 2), (0, cross_length / 2)]
    codes += [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.MOVETO, mpath.Path.MOVETO, mpath.Path.LINETO]

    # Define the circles at the end of each arm
    circle_radius = 5
    for dx, dy in [(-cross_length / 2, 0), (cross_length / 2, 0), (0, -cross_length / 2), (0, cross_length / 2)]:
        circle = mpatches.Circle((dx, dy), circle_radius)
        circle.set_radius(circle_radius)
        circle_path = circle.get_path()
        for point, code in zip(circle_path.vertices, circle_path.codes):
            vertices.append((point[0] + dx, point[1] + dy))
            codes.append(code)

    return mpath.Path(vertices, codes)


UAV_MARKER = create_quadrotor_marker()


# def create_marker(matrix):
#     marker =mpath.Path.unit_regular_star(3)
#     return marker.transformed(matplotlib.transforms.Affine2D(matrix))


class LanesController:
    plot_flag = False
    dbs_list = []

    def __init__(self, t_end=SIMULATION_TIME):
        self.lanes = []
        self.rsu_list = []
        self.t_step = TIME_STEP
        self.current_time = 0
        self.t_end = t_end

    def append_rsu(self, coords):
        self.rsu_list.append(Rsu(coords))

    def append_lane(self, start_coords, end_coords):
        new_lane = Lane(start_coords, end_coords)
        self.lanes.append(new_lane)
        new_lane.generate_vehicle_arrival_times()

    def append_curved_lane(self, start_coords, end_coords, circle_center: Coords3d, arc_angle, init_theta):
        new_lane = CurvedLane(start_coords, end_coords, circle_center, arc_angle, init_theta)
        self.lanes.append(new_lane)
        new_lane.generate_vehicle_arrival_times()

    def simulate_time_step(self, t_step):
        self.current_time += t_step
        if self.current_time >= self.t_end:
            self.current_time = 0
            for _lane in self.lanes:
                _lane.generate_vehicle_arrival_times()
            return
        if self.plot_flag:
            fig.canvas.flush_events()
            plt.pause(0.05)
        [lane.update_vehicles(self.current_time, t_step) for lane in self.lanes]

    def init_plot(self):
        self.plot_flag = True
        min_x, min_y = self.lanes[0].coords[0].x, self.lanes[0].coords[0].y
        max_x, max_y = self.lanes[0].coords[1].x, self.lanes[0].coords[1].y
        for _lane in self.lanes[1:]:
            min_x, min_y = min(_lane.coords[0].x, min_x), min(_lane.coords[0].y, min_y)
            max_x, max_y = max(_lane.coords[1].x, max_x), max(_lane.coords[1].y, max_y)
        ax.set_xlim(min_x - 500, max_x + 500)
        ax.set_ylim(min_y - 500, max_y + 500)
        ax.set_aspect('equal')
        return vehs_plot

    def update_plot(self, i):
        for _v in vehs_plot[:]:
            _v.remove()
        vehs_plot.clear()
        new_coords = np.array([v.coords.as_2d_array() for lane in self.lanes for v in lane.vehicles if v.in_simulation])
        colors = [v.color for lane in self.lanes for v in lane.vehicles if v.in_simulation]
        for (x, y), _c in zip(new_coords, colors):
            veh_plot, = ax.plot(x, y, color=_c, marker='o')
            vehs_plot.append(veh_plot)
        for dbs in self.dbs_list:
            dbs_plot, = ax.plot(dbs.coords.x, dbs.coords.y, color='green', markersize=20, marker=UAV_MARKER.transformed(
                matplotlib.transforms.Affine2D(dbs.ris.orientation.rotation_matrix)));
            vehs_plot.append(dbs_plot)
        for _rsu in self.rsu_list:
            rs, = ax.plot(_rsu.coords.x, _rsu.coords.y, color=_rsu.color, marker='P')
            vehs_plot.append(rs)
        return vehs_plot

    def generate_plot(self, dbs_list):
        self.ani = FuncAnimation(fig, self.update_plot,
                                 init_func=self.init_plot, blit=False, interval=500, frames=None, repeat=False)
        self.plot_flag = True
        self.dbs_list = dbs_list
        plt.ion()
        plt.show()

    def select_v2v_pair(self, max_dist=MAX_LINK_DIST, min_dist=MIN_LINK_DIST):
        # From same lane exclusively
        _vehs = []
        _lanes = rng.choice(self.lanes, len(self.lanes), replace=False)
        for i, _lane in enumerate(_lanes):
            _vehs = [_v for _v in _lane.vehicles if _v.data_transfer_flag == False]
            if len(_vehs) > 1:
                break
        if len(_vehs) < 2:
            return None, None, None
        v1, v2, blocker = None, None, None
        _vehs_random = rng.choice(_vehs, len(_vehs), replace=False)
        if max_dist is not None or min_dist is not None:
            for idx_1 in range(len(_vehs_random)):
                for idx_2 in range(idx_1 + 1, len(_vehs_random)):
                    dist = _vehs_random[idx_1].coords.get_distance_to(_vehs_random[idx_2].coords)
                    if dist <= max_dist and dist >= MIN_LINK_DIST:
                        v1, v2 = _vehs_random[idx_1], _vehs_random[idx_2]
                        break
                if v1 is not None and v2 is not None:
                    break

        else:
            v1, v2 = rng.choice(_vehs, 2, replace=False)
        if v1:
            idx_1, idx_2 = np.sort([_vehs.index(v1), _vehs.index(v2)])
            v_between = [_lane.vehicles[i].coords.z for i in range(idx_1 + 1, idx_2)]
            if v_between:
                max_z_idx = np.array(v_between).argmax()
                blocker = _lane.vehicles[max_z_idx]
            v1.data_transfer_flag = True
            v2.data_transfer_flag = True
            v1.color = 'red'
            v2.color = 'red'
        return v1, v2, blocker

    def select_v2i_pair(self, max_dist=MAX_LINK_DIST, min_dist=MIN_LINK_DIST):
        _vehs = []
        _lanes = rng.choice(self.lanes, len(self.lanes), replace=False)
        for i, _lane in enumerate(_lanes):
            _vehs = [_v for _v in _lane.vehicles if _v.data_transfer_flag == False]
            if len(_vehs) > 0:
                break
        if len(_vehs) < 1:
            return None, None
        veh_list = rng.choice(_vehs, len(_vehs), replace=False)
        rsu_list = rng.choice(self.rsu_list, len(self.rsu_list), replace=False)
        v1, rsu1 = None, None
        if max_dist is not None or min_dist is not None:
            for _rsu in rsu_list:
                for _veh in veh_list:
                    dist = _veh.coords.get_distance_to(_rsu.coords)
                    if dist <= max_dist and dist >= MIN_LINK_DIST:
                        v1, rsu1 = _veh, _rsu
                        break
                if v1 is not None and rsu1 is not None:
                    break
        else:
            v1, rsu1 = veh_list[0], rsu_list[0]
        if v1:
            v1.data_transfer_flag = True
            v1.color = 'red'
            rsu1.color = 'red'
        return v1, rsu1


class Lane:
    def __init__(self, start_coords: Coords3d, end_coords: Coords3d):
        self.coords = [start_coords, end_coords]
        self.vehicles = []
        self.arrival_times = []
        self.direction = (self.coords[1] - self.coords[0]) / (self.coords[1] - self.coords[0]).norm()
        self.distance = self.coords[0].get_distance_to(self.coords[1])

    def generate_vehicle_arrival_times(self, lambda_val=LAMBDA_VAL, T_end=SIMULATION_TIME):
        self.arrival_times = []
        t = 0
        t_shift = self.distance / VEH_SPEED_RANGE[0]
        arrival_t_end = T_end + t_shift
        while t < arrival_t_end:
            interarrival_time = rng.exponential(
                1 / lambda_val)  # Generate interarrival time from exponential distribution
            t += interarrival_time  # Update time and append arrival time to list
            self.arrival_times.append(t)
            # print("arrival times: ", arrival_times)
        return self.arrival_times

    def generate_vehicle(self, arrival_time, current_t):
        speed = VEH_SPEED_RANGE[0]  # Min speed for now
        start_coords = self.coords[0] + self.direction * (current_t - arrival_time) * speed
        self.vehicles.append(Vehicle(start_coords, speed))

    def update_vehicles(self, current_t, t_step):
        for _ve in self.vehicles:
            if not _ve.in_simulation:
                continue
            _ve.update_coords(t_step, self.direction)
            if not (self.direction.x * _ve.coords.x <= self.coords[1].x and self.direction.y * _ve.coords.y <=
                    self.coords[1].y):
                _ve.in_simulation = False
                self.vehicles.remove(_ve)
        while self.arrival_times[0] < current_t:
            new_t = heapq.heappop(self.arrival_times)
            self.generate_vehicle(new_t, current_t)


class CurvedLane(Lane):
    def __init__(self, start_coords: Coords3d, end_coords: Coords3d, circle_center: Coords3d, arc_angle, init_theta):
        super().__init__(start_coords, end_coords)
        self.radius = circle_center.get_distance_to(start_coords)
        self.circle_center = circle_center
        self.arc_angle = arc_angle
        self.init_theta = init_theta
        self.init_theta = get_azimuth(circle_center.x, circle_center.y, start_coords.x, start_coords.y)
        self.distance = 2 * np.pi * self.radius * abs(self.arc_angle - self.init_theta)

    def update_vehicles(self, current_t, t_step):
        for _ve in self.vehicles:
            if not _ve.in_simulation:
                continue
            _ang_step = t_step * _ve.speed / self.radius
            _ve.update_coords(t_step, None, _ang_step, self.circle_center, self.radius)
            if not _ve.theta <= self.arc_angle:
                _ve.in_simulation = False
                self.vehicles.remove(_ve)
        while self.arrival_times[0] < current_t:
            new_t = heapq.heappop(self.arrival_times)
            self.generate_vehicle(new_t, current_t)

    def generate_vehicle(self, arrival_time, current_t):
        init_theta = self.init_theta
        speed = VEH_SPEED_RANGE[0]  # Min speed for now
        _x = self.circle_center.x + self.radius * np.cos(init_theta)
        _y = self.circle_center.y + self.radius * np.sin(init_theta)
        self.vehicles.append(Vehicle(Coords3d(_x, _y, 0), speed, init_theta))


class Vehicle:
    def __init__(self, coords: Coords3d, speed=VEH_SPEED_RANGE[0], init_theta=0):
        self.coords = coords
        self.transceiver = RfTransceiver(self.coords, t_power=TX_POWER_VEHICLE, vehicular=True,
                                         radiation_pattern=RadiationPattern.isotropic)
        self.speed = speed
        self.in_simulation = True
        self.theta = init_theta
        self.data_transfer_flag = False
        self.color = 'k'
        self.coords.z = rng.uniform(VEH_ANTENNA_HEIGHT[0], VEH_ANTENNA_HEIGHT[1])
        self.lane_direction = None

    def update_coords(self, t_step, lane_direction, del_theta=None, circle_center=None, radius=None):
        self.lane_direction =lane_direction
        if del_theta is None:
            self.coords.update_coords_from_array(self.coords + t_step * lane_direction * self.speed)
        else:
            self.theta += del_theta
            self.coords.x = circle_center.x + radius * np.cos(self.theta)
            self.coords.y = circle_center.y + radius * np.sin(self.theta)

    def get_loc_after_t(self, t_step):
        if self.lane_direction:
            return self.coords + t_step * self.lane_direction * self.speed
        else:
            return self.coords

class Rsu:
    def __init__(self, coords: Coords3d):
        self.coords = coords
        self.color = 'black'
        self.in_simulation = True
        self.transceiver = RfTransceiver(self.coords, t_power=TX_POWER_VEHICLE, vehicular=True,
                                         radiation_pattern=RadiationPattern.isotropic)
    def get_loc_after_t(self, t_step):
        return self.coords

if __name__ == '__main__':
    lane_length = 1000
    lane_step = 50
    lane_height = 0
    dbs_height = 500
    ris_height = 30
    lane_1_start = Coords3d(0, 0, lane_height)
    lane_2_start = Coords3d(0, 0, lane_height)
    lane_1_end = Coords3d(lane_length, 0, lane_height)
    lane_2_end = Coords3d(0, lane_length, lane_height)
    lane_c = LanesController()
    lane_c.append_lane(lane_1_start, lane_1_end)
    lane_c.append_lane(lane_2_start, lane_2_end)
    lane_c.append_curved_lane(Coords3d(0, 0, lane_height), Coords3d(10, lane_length, lane_height),
                              Coords3d(-500, lane_length / 2, 0), np.pi / 4, -np.pi / 4)
    lane_c.simulate_time_step(1)
    lane_c.simulate_time_step(1)
    lane_c.generate_plot()
    lane_c.simulate_time_step(1)
    # lane_c.simulate_time_step()
    # lane_c.simulate_time_step()
    # lane_c.simulate_time_step()
    # lane_c.simulate_time_step()
    # lane_c.simulate_time_step()
    # lane_c.simulate_time_step()
    # lane_c.simulate_time_step()
