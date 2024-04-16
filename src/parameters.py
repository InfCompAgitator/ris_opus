from src.types_constants import LinkType
import numpy as np
from numpy import exp
from numpy.random import default_rng
rng = default_rng(seed=95)


def lin2db(linear_input):
    return 10 * np.log10(linear_input)


def db2lin(db_input):
    return 10 ** (db_input / 10)


#Boundaries of the area in meters
X_BOUNDARY = [0, 10000]
Y_BOUNDARY = [0, 10000]
MBS_HEIGHT = 25  # m
MBS_LOCATIONS = [
    (X_BOUNDARY[0], Y_BOUNDARY[1], MBS_HEIGHT),
    (X_BOUNDARY[1], Y_BOUNDARY[0], MBS_HEIGHT),
    (X_BOUNDARY[0], Y_BOUNDARY[0], MBS_HEIGHT),
    (X_BOUNDARY[1], Y_BOUNDARY[1], MBS_HEIGHT)
]
N_MBS = len(MBS_LOCATIONS)

# Obstacles
EXTEND_TIMES_FOUR = True
BOUNDARY_MARGIN = 20

# Users Mobility Model
NUM_OF_USERS = 140
USER_SPEED = [0.5, 0.8]
PAUSE_INTERVAL = [0, 60]
TIME_STEP = 0.5  # Between subsequent users mobility model updates
TIME_SLEEP = 2  # Sleep between updates to allow plotting to keep up
BUILDINGS_AREA_MARGIN = 50
SIMULATION_TIME = 10
USER_SPEED_DIVISOR = 1

# Channel model PLOS
PLOS_AVG_LOS_LOSS = 1
PLOS_AVG_NLOS_LOSS = 20
# PLOS_A_PARAM = 9.61
# PLOS_B_PARAM = 0.16
PLOS_A_PARAM = 5.05  # Obtained using the method
PLOS_B_PARAM = 0.38

# Channel model RF
DRONE_TX_POWER_RF = 0.2  # W
USER_BANDWIDTH = 500e3  # *2  # Hz
DRONE_BANDWIDTH = 20e6  # +5e6 # Hz
MBS_BANDWIDTH = 20e6  # Hz
DEFAULT_CARRIER_FREQ_MBS = 2e9  # Hz
DEFAULT_CARRIER_FREQ_DRONE = 2e9 + MBS_BANDWIDTH  # Hz
NOISE_SPECTRAL_DENSITY = -174  # dBm/Hz
NOISE_POWER_RF = db2lin(NOISE_SPECTRAL_DENSITY - 30 + lin2db(USER_BANDWIDTH))  # dBm input -> linear in W
DEFAULT_SNR_THRESHOLD = db2lin(25)  # linear
MBS_TX_POWER_RF = 0.5  # W
DEFAULT_SINR_THRESHOLD = db2lin(10)
ASSOCIATION_SCHEME = 'SINR'
DEFAULT_SINR_SENSITIVITY_LEVEL = db2lin(-10)

# Mobility Management
DEFAULT_A3_INDIV_OFFSET = 0

# Channel model FSO
RX_DIAMETER = 0.2  # m
DIVERGENCE_ANGLE = 0.06  # rads
RX_RESPONSIVITY = 0.5
AVG_GML = 3
WEATHER_COEFF = 4.3 * 10 ** -4  # /m
POWER_SPLIT_RATIO = 0.005
FSO_ENERGY_HARVESTING_EFF = 0.2
TX_POWER_FSO_DRONE = 0.2  # W
TX_POWER_FSO_MBS = 380  # W
BANDWIDTH_FSO = 1e9  # Hz
NOISE_VARIANCE_FSO = 0.8 * 1e-9
NOISE_POWER_FSO = 1e-6
EMPIRICAL_SNR_LOSSES = db2lin(15)  # Linear
BEAMWAIST_RADIUS = 0.25e-3
WAVELENGTH = 1550e-9
AvgGmlLoss = {LinkType.A2G: 3, LinkType.A2A: 3 / 1.5, LinkType.G2G: 5}  # TODO: Get refs

# BEAMWAIST_RADII = [0.0045, 0.015, 0.0045, 0.015, 0.0045, 0.015, 0.0045, 0.015]
BEAMWAIST_RADII = [0.01, 0.01, 0.02, 0.02]

UAV_HEIGHT = 25
MBS_HEIGHT = 25  # m
UE_HEIGHT = 1.5  # To conform with channel models

# MBS_LOCATIONS = [Coords3d(400, 0, MBS_HEIGHT)]
# N_MBS = len(MBS_LOCATIONS)

# USER_MOBILITY_SAVE_NAME = 'users_200_truncated'
USER_MOBILITY_SAVE_NAME = 'extended_4_madrids_500_users'

# RIS
RIS_ACTIVE = True
N_HOPS_ALLOWED = 8
RELAYS_BANDWIDTH = 20e6
RELAYS_BANDWIDTH_5G_NOISE = 18720000  # 38.306 model by Stroke
RELAYS_BANDWIDTH_5G = 17472000  # 38.306 model by Stroke
NOISE_SPECTRAL_DENSITY_RELAY = -174
RELAYS_NOISE_POWER = db2lin(NOISE_SPECTRAL_DENSITY_RELAY - 30 + lin2db(RELAYS_BANDWIDTH))  # dBm input -> linear in W
RELAYS_NOISE_POWER_5G = db2lin(
    NOISE_SPECTRAL_DENSITY_RELAY - 30 + lin2db(RELAYS_BANDWIDTH_5G_NOISE))  # dBm input -> linear in W
EFFICIENCY_5G_RATE_FACTOR = 0.82
RELAYS_TRANSMISSION_POWER = 100e-3
MAXIMUM_PL = 90  # 85, 100, 111 for 200, 100, 50 mbps 90 for 150
ACTIVE_RIS_GAIN = 20  # dB
RELAYS_REQUIRED_RATE = (200e6)
USE_POZNAN = False

# Vehicular
TX_POWER_VEHICLE = 0.2
# DBS
UAV_SPEED = 15
ROTATION_SPEED = np.deg2rad(20)  # rad/s
MAX_DIST_FROM_DBS = 1200
#FINAL
#140 50, 50
#FINAL_BENCH

#150 50, 50 fixed Q states update
#151 50, 50 fixed Q states update reward absolute

#152 50, 50 BATCH 0.2/ 1/x 152
#153 50, 50 BATCH 0.2/ 0.4 152
#154 100, 100 BATCH 0.2/ 1/x 175 1000-1200 max dist
#155 100, 100  0.2/ 1/x 178 1000-1200 max dist

#156 100, 100  0.2/ 1/x 178 500-2000 max dist

class QLearningParams:
    ELEVATION_CARDINALITY = 100
    AZIMUTH_CARDINALITY = 100
    DISTANCES_CARDINALITY = 10
    DISCOUNT_RATIO = 0.2
    LOOK_AHEAD_COUNTS = 30
    CHECKPOINT_ID = f'156'
    CHECKPOINTS_FILE = 'C:\\Users\\user\\PycharmProjects\\ris_project\\src\\machine_learning\\checkpoints'
    REWARDS_FILE = 'C:\\Users\\user\\PycharmProjects\\ris_project\\src\\machine_learning\\rewards_per_cycle'
    SAVE_ON = None
    LOAD_MODEL = True
    TESTING_FLAG = False  # Turns off exploration
    EXPLORATION_PROB = lambda x: 0.2
    # EXPLORATION_PROB = lambda x: 0.8 * exp(-0.0000004 * x)
    # LEARNING_RATE = lambda x: 0.4
    LEARNING_RATE = lambda x: 1 / (x ** (2 / 14))
    Q_MANAGER_ID = '1'
    BATCH_INITIALIZATION = False
    EXPLORE_UNEXPLORED = True
    BENCHLINE = False
    N_SELECTED_PAIRS = 1
    N_Q_PAIRS = 1



DEBUG = True