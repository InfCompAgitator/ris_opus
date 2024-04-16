from src.channel_model.ris_model import RIS, RadiationPattern, RfTransceiver, Coords3d, Orientation, \
    get_rotation_matrix, lin2db
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib;
matplotlib.use('QtAgg')
from src.parameters import TIME_STEP, RELAYS_BANDWIDTH_5G, rng, TX_POWER_VEHICLE, DEBUG
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm
from src.channel_model.v2v import ThreegppModel_H
from src.channel_model.v2v import ThreegppModel_H
from src.channel_model.mmwave_modeling import get_throughput_5g

# matplotlib.use("TkAgg")
from numpy import arctan, cos
from scipy.optimize import minimize
def get_optimal_height(d, H_initial, H_bounds):
    def f(H, d):
        return np.float64((d ** 2 + H ** 2) ** 2 / (cos(arctan(d / H,dtype=np.float64),dtype=np.float64) ** 6))

    result = minimize(lambda H: np.float64(f(H, d)), H_initial, method='L-BFGS-B', bounds=[H_bounds])
    return result.x

if __name__ == '__main__':
    transceiver_1 = RfTransceiver(Coords3d(0, 0, 25), vehicular=True)
    transceiver_1.radiation_pattern = RadiationPattern.isotropic
    transceiver_2 = RfTransceiver(Coords3d(0, 200, 2), vehicular=True)
    transceiver_2.radiation_pattern = RadiationPattern.isotropic
    # ris_1 = RIS(Coords3d(0, 0.5, get_optimal_height(transceiver_2.coords.get_distance_to(transceiver_1.coords), 500, [0,10000])[0]))
    # # ris_1 = RIS(Coords3d(0, 500, 200))
    # xy_rotation_vector = np.array([1, 0, 0])
    # ris_1.orientation = Orientation(
    #     get_rotation_matrix(xy_rotation_vector / np.linalg.norm(xy_rotation_vector), np.array([0, 0, -1])),
    #     xy_rotation_vector)
    # ris_1.update_elems_coords()
    # print(lin2db(ris_1.get_path_loss_beamforming(transceiver_1, transceiver_2)))
    # print(lin2db(ThreegppModel_H.get_path_loss_v2i(transceiver_2.coords, transceiver_1.coords)))
    # rate_ris = get_throughput_5g(TX_POWER_VEHICLE/ris_1.get_path_loss_beamforming(transceiver_1, transceiver_2))
    # rate_direct = get_throughput_5g(TX_POWER_VEHICLE/ThreegppModel_H.get_path_loss_v2i(transceiver_2.coords, transceiver_1.coords))
    # print(rate_ris/rate_direct)
    ris_1 = RIS((transceiver_2.coords + transceiver_1.coords)/2,)
    ris_1.coords.z = 50

    xy_rotation_vector = np.array([1, 0, 0])
    ris_1.orientation = Orientation(
        get_rotation_matrix(xy_rotation_vector / np.linalg.norm(xy_rotation_vector), np.array([0, 0, -1])),
        xy_rotation_vector)
    ris_1.update_elems_coords()
    theta_rot = np.pi / 20
    print(lin2db(ris_1.get_path_loss_beamforming(transceiver_1, transceiver_2)))

    # print(lin2db(ThreegppModel.get_path_loss(transceiver_1.coords, transceiver_2.coords)))
    ris_1.coords = Coords3d(100, 200, 100)
    ris_1.update_elems_coords()
    # print(lin2db(ThreegppModel.get_path_loss(transceiver_1.coords, transceiver_2.coords)))
    coords_xs = [0, 100]
    coords_ys = [100, 300, 500]
    fig, ax = plt.subplots()
    path_loss_map_rot = np.zeros(int(2*np.pi/theta_rot))
    for _x in coords_xs:
        for _y in coords_ys:
            ris_1.coords = Coords3d(_x, _y, 100)
            ris_1.update_elems_coords()
            for idx in tqdm(range(int(2*np.pi/theta_rot))):
                ris_1.orientation.rotate(theta_rot)
                ris_1.update_elems_coords()
                path_loss_map_rot[idx] = lin2db(ris_1.get_path_loss_beamforming(transceiver_1, transceiver_2))
            ax.plot(np.rad2deg(np.arange(path_loss_map_rot.shape[0])*theta_rot), path_loss_map_rot.astype(int))
    # ax.plot(np.rad2deg(np.arange(path_loss_map_rot.shape[0])*theta_rot),
    #         lin2db(ThreegppModel.get_path_loss(transceiver_1.coords, transceiver_2.coords)) * np.ones(path_loss_map_rot.shape[0]))

    ax.set_ylabel('Path loss [dB]')
    ax.set_xlabel('Yaw rotation (about vertical axis) [\u00B0]')
    fig.show()

    xy_granularity = 200
    xy_margin = 100
    x_s, y_s = np.linspace(transceiver_1.coords.x - xy_margin, transceiver_2.coords.x + xy_margin,
                           num=xy_granularity), np.linspace(
        transceiver_1.coords.y - xy_margin, transceiver_2.coords.y + xy_margin, num=xy_granularity)
    xs, ys = np.meshgrid(x_s, y_s)
    path_loss_map = np.zeros((len(xs), len(ys)))

    for row in tqdm(range(len(x_s))):
        for col in range(len(y_s)):
            ris_1.coords.x, ris_1.coords.y = xs[row, col], ys[row, col]
            ris_1.update_elems_coords()
            path_loss_map[row, col] = lin2db(ris_1.get_path_loss_beamforming(transceiver_1, transceiver_2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xs, ys, path_loss_map, cmap='viridis', zorder=path_loss_map.min() - 10)


    radius=10
    circle1 = plt.Circle((transceiver_2.coords.x, transceiver_2.coords.y), radius=radius, color='r', fill=True)
    circle2 = plt.Circle((transceiver_1.coords.x, transceiver_1.coords.y), radius=radius, color='r', fill=True)
    # ax.add_patch (circle2)
    # ax.add_patch (circle1)

    ax.plot([transceiver_2.coords.x], [transceiver_2.coords.y], [path_loss_map.min() - 10], 'ro', label='Vehicle Location')
    ax.plot([transceiver_1.coords.x], [transceiver_1.coords.y], [path_loss_map.min() - 10], 'ro')


    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('Path Loss [dB]')
    fig.colorbar(surf)
    fig.legend(loc='upper center')
    # fig.show()
#
#Movie


    # # Set initial view
    # ax.view_init(elev=30, azim=30)
    #
    # # Function to update the angle of rotation
    # def rotate(elev, az):
    #     ax.view_init(elev=elev, azim=az)
    # # Function to convert plot to image
    # def plt2img(fig):
    #     fig.canvas.draw()
    #     img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #     img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     return img
    # # Create an animation by rotating the plot and capture frames
    # angle_increment = 2 # defines smoothness of the rotation, lower value means smoother video
    # frames = []
    # for az, elev in zip(range(0, 360, angle_increment), range(-45, 90, angle_increment)):
    #     rotate(elev, az)
    #     plt.draw()
    #     frame = plt2img(fig) # Capture the current plot as an image
    #     frames.append(frame)
    #
    #
    #
    # # Use moviepy to create a video clip
    # clip = ImageSequenceClip(frames, fps=10)
    # clip.write_videofile('rotation3_2.mp4', codec='libx264')