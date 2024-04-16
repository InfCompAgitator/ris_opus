# class QLearningParams:
#     ELEVATION_CARDINALITY = 100
#     AZIMUTH_CARDINALITY = 100
#     DISTANCES_CARDINALITY = 10
#     DISCOUNT_RATIO = 0.2
#     LOOK_AHEAD_COUNTS = 30
#     CHECKPOINT_ID = f'156'
#     CHECKPOINTS_FILE = 'C:\\Users\\user\\PycharmProjects\\ris_project\\src\\machine_learning\\checkpoints'
#     REWARDS_FILE = 'C:\\Users\\user\\PycharmProjects\\ris_project\\src\\machine_learning\\rewards_per_cycle'
#     SAVE_ON = None
#     LOAD_MODEL = False
#     TESTING_FLAG = False  # Turns off exploration
#     EXPLORATION_PROB = lambda x: 0.2
#     # EXPLORATION_PROB = lambda x: 0.8 * exp(-0.0000004 * x)
#     # LEARNING_RATE = lambda x: 0.4
#     LEARNING_RATE = lambda x: 1 / (x ** (2 / 14))
#     Q_MANAGER_ID = '1'
#     BATCH_INITIALIZATION = False
#     EXPLORE_UNEXPLORED = True
#     BENCHLINE = True
#     N_SELECTED_PAIRS = 1
#     N_Q_PAIRS = 1

# percom_sim_v2x_single_orientation_look_ahead or percom_sim_v2x_single_orientation_separate

# Parameters.py
# from src.types_constants import LinkType
# import numpy as np
# from numpy import exp
# from numpy.random import default_rng
#
# rng = default_rng(seed=95)
#
#
# def lin2db(linear_input):
#     return 10 * np.log10(linear_input)
#
#
# def db2lin(db_input):
#     return 10 ** (db_input / 10)
#
#
# # Obstacles
# EXTEND_TIMES_FOUR = True
# BOUNDARY_MARGIN = 20
#
# # Users Mobility Model
# NUM_OF_USERS = 140
# USER_SPEED = [0.5, 0.8]
# PAUSE_INTERVAL = [0, 60]
# TIME_STEP = 0.5  # Between subsequent users mobility model updates
# TIME_SLEEP = 2  # Sleep between updates to allow plotting to keep up
# BUILDINGS_AREA_MARGIN = 50
# SIMULATION_TIME = 10
# USER_SPEED_DIVISOR = 1
#
# # Channel model PLOS
# PLOS_AVG_LOS_LOSS = 1
# PLOS_AVG_NLOS_LOSS = 20
# # PLOS_A_PARAM = 9.61
# # PLOS_B_PARAM = 0.16
# PLOS_A_PARAM = 5.05  # Obtained using the method
# PLOS_B_PARAM = 0.38
#
# # Channel model RF
# DRONE_TX_POWER_RF = 0.2  # W
# USER_BANDWIDTH = 500e3  # *2  # Hz
# DRONE_BANDWIDTH = 20e6  # +5e6 # Hz
# MBS_BANDWIDTH = 20e6  # Hz
# DEFAULT_CARRIER_FREQ_MBS = 2e9  # Hz
# DEFAULT_CARRIER_FREQ_DRONE = 2e9 + MBS_BANDWIDTH  # Hz
# NOISE_SPECTRAL_DENSITY = -174  # dBm/Hz
# NOISE_POWER_RF = db2lin(NOISE_SPECTRAL_DENSITY - 30 + lin2db(USER_BANDWIDTH))  # dBm input -> linear in W
# DEFAULT_SNR_THRESHOLD = db2lin(25)  # linear
# MBS_TX_POWER_RF = 0.5  # W
# DEFAULT_SINR_THRESHOLD = db2lin(10)
# ASSOCIATION_SCHEME = 'SINR'
# DEFAULT_SINR_SENSITIVITY_LEVEL = db2lin(-10)
#
# # Mobility Management
# DEFAULT_A3_INDIV_OFFSET = 0
#
# # Channel model FSO
# RX_DIAMETER = 0.2  # m
# DIVERGENCE_ANGLE = 0.06  # rads
# RX_RESPONSIVITY = 0.5
# AVG_GML = 3
# WEATHER_COEFF = 4.3 * 10 ** -4  # /m
# POWER_SPLIT_RATIO = 0.005
# FSO_ENERGY_HARVESTING_EFF = 0.2
# TX_POWER_FSO_DRONE = 0.2  # W
# TX_POWER_FSO_MBS = 380  # W
# BANDWIDTH_FSO = 1e9  # Hz
# NOISE_VARIANCE_FSO = 0.8 * 1e-9
# NOISE_POWER_FSO = 1e-6
# EMPIRICAL_SNR_LOSSES = db2lin(15)  # Linear
# BEAMWAIST_RADIUS = 0.25e-3
# WAVELENGTH = 1550e-9
# AvgGmlLoss = {LinkType.A2G: 3, LinkType.A2A: 3 / 1.5, LinkType.G2G: 5}  # TODO: Get refs
#
# # BEAMWAIST_RADII = [0.0045, 0.015, 0.0045, 0.015, 0.0045, 0.015, 0.0045, 0.015]
# BEAMWAIST_RADII = [0.01, 0.01, 0.02, 0.02]
#
# UAV_HEIGHT = 25
# MBS_HEIGHT = 25  # m
# UE_HEIGHT = 1.5  # To conform with channel models
#
# # MBS_LOCATIONS = [Coords3d(400, 0, MBS_HEIGHT)]
# # N_MBS = len(MBS_LOCATIONS)
#
# # USER_MOBILITY_SAVE_NAME = 'users_200_truncated'
# USER_MOBILITY_SAVE_NAME = 'extended_4_madrids_500_users'
#
# # RIS
# RIS_ACTIVE = True
# N_HOPS_ALLOWED = 8
# RELAYS_BANDWIDTH = 20e6
# RELAYS_BANDWIDTH_5G_NOISE = 18720000  # 38.306 model by Stroke
# RELAYS_BANDWIDTH_5G = 17472000  # 38.306 model by Stroke
# NOISE_SPECTRAL_DENSITY_RELAY = -174
# RELAYS_NOISE_POWER = db2lin(NOISE_SPECTRAL_DENSITY_RELAY - 30 + lin2db(RELAYS_BANDWIDTH))  # dBm input -> linear in W
# RELAYS_NOISE_POWER_5G = db2lin(
#     NOISE_SPECTRAL_DENSITY_RELAY - 30 + lin2db(RELAYS_BANDWIDTH_5G_NOISE))  # dBm input -> linear in W
# EFFICIENCY_5G_RATE_FACTOR = 0.82
# RELAYS_TRANSMISSION_POWER = 100e-3
# MAXIMUM_PL = 100  # 85, 100, 111 for 200, 100, 50 mbps
# ACTIVE_RIS_GAIN = 20  # dB
# RELAYS_REQUIRED_RATE = (200e6)
# USE_POZNAN = True
#
# # Vehicular
# TX_POWER_VEHICLE = 0.2
# # DBS
# UAV_SPEED = 15
# ROTATION_SPEED = np.deg2rad(20)  # rad/s
# MAX_DIST_FROM_DBS = 1200
# #PAyload Ration Diff
# #9: Q_1 3 streets 0.00001, 2/12
# #10: Q_2 3 streets 0.00001, 2/12
# #11:medians_z 3 streets
# #12: medians_z 1 streets, v2v init = 0.1, v init = 0.4
#
# #Payload ration
# #13: Q_1 3 streets 0.00001, 2/12
# #14: Q_2 3 streets 0.00001, 2/12 v2v init = 0.05, v init = 0.15
# #15:medians_z 3 streets
# #16: medians_z 1 streets, v2v init = 0.1, v init = 0.4
#
# #21 single pair Q1 6
# #22 Single pair Q2 az 1
# #23 Single pair Q3 az dist 2
# #24 Single pair Q4 az dist elev ELEVATION_CARDINALITY = 5  AZIMUTH_CARDINALITY = 5 3
# #25 Single pair Q2 az ELEVATION_CARDINALITY = 5  AZIMUTH_CARDINALITY = 5 4
# #26 Single pair Q1 ELEVATION_CARDINALITY = 5  AZIMUTH_CARDINALITY = 5 5
# #27 Single pair Q5 ELEVATION_CARDINALITY = 5  AZIMUTH_CARDINALITY = 5 5
# #28 Single pair Q5 ELEVATION_CARDINALITY =10  AZIMUTH_CARDINALITY = 10 5
# #29 Single pair Q5 ELEVATION_CARDINALITY =5  AZIMUTH_CARDINALITY = 10  DIST_CARDINALITY = 15
#
#
# #30 Q5 5,5,5,0.6 BATCH_INITIALIZATION, 1 / (x ** (2 / 12)) 1
# #31 Q5 5,5,5,0.6 EXPLORE_UNEXPLORED, 1 / (x ** (2 / 12)) 2
# #32 Q5 5,5,5,0.6 EXPLORE_UNEXPLORED, 0.4 3
# #33 Q5 5,5,5,0.6 BATCH_INITIALIZATION, 0.4 0
#
# #60GHz only loc
# #34 Q5 5,5,5,0.6 BATCH_INITIALIZATION, 0.4 4
# #35 Q5 5,5,5,0.6 EXPLORE_UNEXPLORED, 0.4 5
#
# #36 Q5 5,10,5,0.6 BATCH_INITIALIZATION, 0.4 0
# #37 Q5 5,10,10,0.6 BATCH_INITIALIZATION, 0.4 0
#
# #38 Q5 5,10,5,0.3 BATCH_INITIALIZATION, 1 / (x ** (2 / 12)) 0.2
# #39 Q5 5,10,5,0.3 EXPLORE_UNEXPLORED, 1 / (x ** (2 / 12)) 0.2
# #40 Q5 5,10,5,0.3 BATCH_INITIALIZATION,  0.4 1, 0.2
# #41 Q5 5,10,5,0.3 EXPLORE_UNEXPLORED,  0.4 2, 0.2
#
#
# #42 Q1 5,5,5,0.3 EXPLORE_UNEXPLORED,  0.4 1, 0.2
# #43 Q2 5,5,5,0.3 EXPLORE_UNEXPLORED,  0.4 2, 0.2
# #44 Q3 5,5,5,0.3 EXPLORE_UNEXPLORED,  0.4 2, 0.2
# #45 Q2_t 5,5,5,0.3 EXPLORE_UNEXPLORED,  0.4 2, 0.2
#
# #Random car 5,10, 5
#
# #45 BATCH_ 0.2, 0.4
# #46 EXPLORE_ 0.2, 0.4
#
# #47 5, 10, 10 0.2, 0.4 EXPLORE
# #48 5, 10, 10 0.2, 0.4 BATCH
#
# #*10 rewards
# #49, 10, 10 0.2, 0.4 EXPLORE
# #50 5, 10, 10 0.2, 0.4 BATCH
# #51 5, 10, 5 1 / (x ** (2 / 12)) BATCH
# #52 5, 10, 5 1 / (x ** (2 / 12)) EXPLORE
#
# #TODO: reverse street
# #53 5, 10, 5 1 / (x ** (2 / 12)) EXPLORE
# #54 5, 10, 5 1 / (x ** (2 / 12)) BATCH
#
# #56 5, 10, 10  0.2, 0.4  BATCH
# #57 5, 10, 10  0.2, 0.4  EXPLORE
#
# #58 5, 10, 10  1 / (x ** (2 / 12))  EXPLORE
# #58_2 5, 15, 10  1 / (x ** (2 / 12))  EXPLORE
# #59 5, 10, 10  1 / (x ** (2 / 12))  BATCH
#
# #60 5, 10, 5  0.2, 0.4  BATCH
# #61 5, 10, 5 0.2, 0.4  EXPLORE
#
# #62 5, 10, 5  1 / (x ** (2 / 12))  EXPLORE
# #63 5, 10, 5  1 / (x ** (2 / 12))  BATCH
#
# #64 Q6 5, 10, 10  1 / (x ** (2 / 12))  EXPLORE
# #65 Q6 5, 15, 15  1 / (x ** (2 / 12))  EXPLORE
# #66 Q5 5, 15, 15  1 / (x ** (2 / 12))  EXPLORE
# #67 Q5 10, 15, 15  1 / (x ** (2 / 12))  EXPLORE
#
# #DISCOUNT_RATIO = 0.5
# """    EXPLORATION_PROB = lambda x: 0.8 * exp(-0.0000004 * x)
#     # LEARNING_RATE = lambda x: 0.4
#     LEARNING_RATE = lambda x: 1 / (x ** (2 / 14))"""
# #68 Q6 5, 10, 10  1 / (x ** (2 / 12))  EXPLORE
# #69 Q6 5, 10, 15  1 / (x ** (2 / 12))  EXPLORE
# #70 Q6 5, 15, 10  1 / (x ** (2 / 12))  EXPLORE
# #71 Q6 5, 15, 15  1 / (x ** (2 / 12))  EXPLORE
# #72 Q6 10, 10, 10  1 / (x ** (2 / 12))  EXPLORE
# #73 Q6 10, 15, 10  1 / (x ** (2 / 12))  EXPLORE
# #74 Q6 10, 15, 15  1 / (x ** (2 / 12))  EXPLORE
#
#
#
# #75 Q6 5, 10, 10  0.2/0.4  EXPLORE 0.2
# #76 Q6 5, 10, 10  0.2/1 / (x ** (2 / 14))  EXPLORE 0.2
#
# '''Trying more far streets 500 instead of 250'''
# #77 Q6 5, 10, 10  0.2/1 / (x ** (2 / 14))  EXPLORE 0.2
# #78 Q1 5, 10, 10  0.8 * exp(-0.0000004 * x)/1 / (x ** (2 / 14))  EXPLORE 0.2
# #79 Q1 5, 10, 10  0.8 * 0.2 / (x ** (2 / 14))  EXPLORE 0.2
# #80 Q1 5, 15, 5  0.8 * 0.2 / (x ** (2 / 14))  EXPLORE 0.2
# #81 Q1 5, 15, 5  0.8 * 0.2 / (x ** (2 / 14))  EXPLORE 0.4
#
# #### time step 5
# #82_2 Q6 5, 10, 10  1 / (x ** (2 / 12))  EXPLORE
# #83_2 Q5 5, 10, 10  1 / (x ** (2 / 12))  EXPLORE
# #84_2 Q1 5, 10, 10  1 / (x ** (2 / 12))  EXPLORE #WINNER
#
#
# #Time step 5 new Q
# #85 Q2_t 5 10 25 1 / (x ** (2 / 12))  EXPLORE
# #86 Q2_t 5 10 25 1 / (x ** (2 / 12))  EXPLORE 2 secs
# #87 Q2_t 5 10 35 1 / (x ** (2 / 12))  EXPLORE 1 secs
#
# #ORIENTATION ONLY
# #90 Q1 5,5 0.8 * exp(-0.0000004 * x) , 1 / (x ** (2 / 14))
# #91 Q1 5,5 0.2 , 1 / (x ** (2 / 14))
# #92 Q1 5,10 0.2 , 1 / (x ** (2 / 14))
# #93 Q1 10,10 0.2 , 1 / (x ** (2 / 14))
# #94 Q1 15,10 0.2 , 1 / (x ** (2 / 14))
# #95 Q2 15,10 0.2 , 1 / (x ** (2 / 14))
# #96 Q2 10,10 0.2 , 1 / (x ** (2 / 14))
#
# #NEW try 0.5 timestep
# #99 Q1 10,10 0.2 , 1 / (x ** (2 / 14))
# #100 Q1 15,10 0.2 , 1 / (x ** (2 / 14))
# #101 Q2 10, 10 0.2 , 1 / (x ** (2 / 14))
# #102 Q2 15, 10 0.2 , 1 / (x ** (2 / 14))
# #103 Q2 5, 5 0.2 , 1 / (x ** (2 / 14))
#
#
# #2 pairs z
# #104 5,5
# #105 5,10
#
# #1 pairs z
# #106 10 10
#
# #2 pairs Or
# #107 10 10
#
#
# #'TEST_OR_2' 'TEST_OR_1', 1 pair and two pairs orientation tylko
# #'TEST_OR_Z_2' 'TEST_OR_Z_1', 1 pair and two pairs orientation tylko
#
# #'TEST_OR_1_2', 1 pair and two pairs orientation tylko timestep = 2
# #Or 1 pair
# #108 Q1 10,10
# #109 Q2 10,10
#
# #1 time step
# #110 Q2 30,30
# #0.5 time step
# #111 Q2 30,30
#
#
# #0.5 time step
# #112 Q2 30,30
# #TEST_05_BENCH
# #113 Q1 20,20
#
# #0.1 time step
# #114 Q2 30,30
# #TEST_01_BENCH
#
# #OLD lambda vals
# #1 time step
# #115 Q2 30,30
# #TEST_1_BENCH
#
# #2 time step
# #116 Q2 30,30
# #TEST_2_BENCH
#
# #0.5 time step
# #117 Q2 30,30
# #TEST_05_BENCH_2
#
# #0.5 time step
# #118 Q2 50,50
# #TEST_05_BENCH_50
#
#
# #NEW Q Separate movement and orientation 0.5 (compare with TEST_05_BENCH_50_2)
# #119 Q2 30, 30
# #120 Q2 20, 20
# #123 Q1 20, 20
# #124 Q1 only current state 20, 20
# #More orientation actions 0.5 -0.5
# #121 Q2 30, 30
# #122 Q2 20, 20
# #125 Q1 only current state 20, 20
#
#
# #different az idx and elev idx
# #126 Q1 only current state 40, 40
# #127 Q2 state 40, 40
# #128 Q1 only current state 60, 60
# #129 Q2 state 60, 60
# #130 Q1 only current state 100,100
# #131 Q2  100,100
# #more moves
# #132 Q1 only current state 100,100
#
# #more moves
# #134 Q1 only current state 100,100
# #135 Q1 only current state 50,50
#
# #FINAL
# #140 50, 50
# #FINAL_BENCH
#
# #150 50, 50 fixed Q states update
# #151 50, 50 fixed Q states update reward absolute
#
# #152 50, 50 BATCH 0.2/ 1/x 152
# #153 50, 50 BATCH 0.2/ 0.4 152
# #154 100, 100 BATCH 0.2/ 1/x 175 1000-1200 max dist
# #155 100, 100  0.2/ 1/x 178 1000-1200 max dist
#
# #156 100, 100  0.2/ 1/x 178 500-2000 max dist
#
# class QLearningParams:
#     ELEVATION_CARDINALITY = 10
#     AZIMUTH_CARDINALITY = 10
#     DISTANCES_CARDINALITY = 10
#     DISCOUNT_RATIO = 0.2
#     LOOK_AHEAD_COUNTS = 30
#     CHECKPOINT_ID = f'156'
#     CHECKPOINTS_FILE = 'C:\\Users\\user\\PycharmProjects\\ris_project\\src\\machine_learning\\checkpoints'
#     REWARDS_FILE = 'C:\\Users\\user\\PycharmProjects\\ris_project\\src\\machine_learning\\rewards_per_cycle'
#     SAVE_ON = None
#     LOAD_MODEL = False
#     TESTING_FLAG = False  # Turns off exploration
#     EXPLORATION_PROB = lambda x: 0.2
#     # EXPLORATION_PROB = lambda x: 0.8 * exp(-0.0000004 * x)
#     # LEARNING_RATE = lambda x: 0.4
#     LEARNING_RATE = lambda x: 1 / (x ** (2 / 14))
#     Q_MANAGER_ID = '1'
#     BATCH_INITIALIZATION = False
#     EXPLORE_UNEXPLORED = True
#     BENCHLINE = True
#     N_SELECTED_PAIRS = 1
#     N_Q_PAIRS = 1
#
#
#
# DEBUG = True