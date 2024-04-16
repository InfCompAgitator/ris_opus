# from src.percom_sim import qlp, q_managers_dict
# from src.percom_sim_v2x_single import qlp, q_managers_dict
from src.percom_sim_v2x_single_orientation_separate import qlp, q_managers_dict
import numpy as np
import os
import matplotlib.pyplot as plt

from src.channel_model.fso_a2a import StatisticalModel
from src.parameters import *
import matplotlib as mpl

def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    padded_data = np.pad(data, (0, window_size - 1), mode='constant')
    convolved_data = np.convolve(padded_data, window, 'valid')
    return convolved_data



if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')
    csfont = {'fontname': 'Times New Roman'}

    window_size = 10

    n_check_point = 'FINAL_BENCH'
    avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    rewards_avg = np.load(avg_file_name)
    # scale = np.append(np.linspace(1, 1.07,rewards_avg.shape[0]//2), np.linspace(1.07, 1,rewards_avg.shape[0]//2))
    aveged = moving_average(rewards_avg * 1, window_size)[window_size:-window_size]/10
    zero_idx = np.where(aveged == 0)[0][0] - 20
    aveged = aveged[:zero_idx]
    plt.plot(aveged, label='Fixed orientation', linestyle='--', markersize=50)


    n_check_point = '142'
    avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    rewards_avg = np.load(avg_file_name)
    aveged = moving_average(rewards_avg, window_size)[window_size:len(aveged)]/10
    aveged = aveged[:zero_idx]
    plt.plot(aveged, label=n_check_point)

    # n_check_point = '151'
    # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # rewards_avg = np.load(avg_file_name)
    # aveged = moving_average(rewards_avg, window_size)[window_size:len(aveged)]/10
    # aveged = aveged[:zero_idx]
    # plt.plot(aveged, label=n_check_point)

    # n_check_point = '152'
    # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # rewards_avg = np.load(avg_file_name)
    # aveged = moving_average(rewards_avg, window_size)[window_size:len(aveged)]/10
    # aveged = aveged[:zero_idx]
    # plt.plot(aveged, label=n_check_point)

    # n_check_point = '153'
    # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # rewards_avg = np.load(avg_file_name)
    # aveged = moving_average(rewards_avg, window_size)[window_size:len(aveged)] / 10
    # aveged = aveged[:zero_idx]
    # plt.plot(aveged, label=n_check_point)
    #
    # n_check_point = '154'
    # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # rewards_avg = np.load(avg_file_name)
    # aveged = moving_average(rewards_avg, window_size)[window_size:len(aveged)] / 10
    # aveged = aveged[:zero_idx]
    # plt.plot(aveged, label=n_check_point)
    # n_check_point = '155'
    # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # rewards_avg = np.load(avg_file_name)
    # aveged = moving_average(rewards_avg, window_size)[window_size:len(aveged)] / 10
    # aveged = aveged[:zero_idx]
    # plt.plot(aveged, label=n_check_point)
    #
    # # n_check_point = '135'
    # # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # # rewards_avg = np.load(avg_file_name)
    # # aveged = moving_average(rewards_avg, window_size)[window_size:-window_size]
    # # plt.plot(aveged, label=n_check_point)
    #
    # n_check_point = '140'
    # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # rewards_avg = np.load(avg_file_name)
    # aveged = moving_average(rewards_avg, window_size)[window_size:len(aveged)]/10
    # aveged = aveged[:zero_idx]
    # plt.plot(aveged, label='Controlled orientation with Q-learning' , linestyle='-.', markersize=50)

    # n_check_point = '129'
    # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # rewards_avg = np.load(avg_file_name)
    # aveged = moving_average(rewards_avg, window_size)[window_size:-window_size]
    # plt.plot(aveged, label=n_check_point)
    #
    # n_check_point = '130'
    # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # rewards_avg = np.load(avg_file_name)
    # aveged = moving_average(rewards_avg, window_size)[window_size:-window_size]
    # plt.plot(aveged, label=n_check_point)
    #
    # n_check_point = '131'
    # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # rewards_avg = np.load(avg_file_name)
    # aveged = moving_average(rewards_avg, window_size)[window_size:-window_size]
    # plt.plot(aveged, label=n_check_point)
    #
    # n_check_point = '132'
    # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # rewards_avg = np.load(avg_file_name)
    # aveged = moving_average(rewards_avg, window_size)[window_size:-window_size]
    # plt.plot(aveged, label=n_check_point)

    # n_check_point = '133'
    # avg_file_name = os.path.join(qlp.CHECKPOINTS_FILE + '_rewards_per_veh' + f'{n_check_point}' + '.npy')
    # rewards_avg = np.load(avg_file_name)
    # aveged = moving_average(rewards_avg, window_size)[window_size:-window_size]
    # plt.plot(aveged, label=n_check_point)

    # plt.plot(aveged, label=n_check_point, linestyle='-.', markersize=10)

    plt.legend()
    fig =plt.gcf()
    ax =plt.gca()
    ax.set_ylabel('Path-loss [dB]')
    ax.set_xlabel('Cycle index')
    # fig.savefig('rewards.eps', format='eps')

    fig.show()
    # rewards_total = np.load(os.path.join(qlp.REWARDS_FILE, q_managers_dict('1').__repr__(f'{n_check_point}')))
    # rewards_total = rewards_total.ravel()
    # mean = rewards_total.mean() / 1e1
    # print(mean)
    # discount = 1
    # look_ahead_n = 1000
    # jump = 10000
    # rewards_discounted_cumsum = np.zeros((rewards_total.shape[0] - look_ahead_n) // jump + 1)
    # for i in np.arange(len(rewards_total) - look_ahead_n, step=jump):
    #     sum = 0
    #     discount = discount
    #     for j in range(i, i + look_ahead_n):
    #         sum += rewards_total[j] * discount
    #         discount *= discount
    #     rewards_discounted_cumsum[i // jump] = sum / look_ahead_n
    #
    # fig, ax = plt.subplots()
    # ax.plot(rewards_discounted_cumsum)
    # ax.set_xlabel('Cycle', fontsize=15)
    # ax.set_ylabel('Average Reward', fontsize=15)
    # fig.show()
