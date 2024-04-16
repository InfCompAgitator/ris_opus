import matplotlib.pyplot as plt
import numpy as np
from src.parameters import  RELAYS_NOISE_POWER_5G
from src.math_tools import lin2db
if __name__ == '__main__':
    # xs = np.linspace(0, np.pi / 2, 300)
    # ys = np.floor(5 *(xs / (np.pi / 2))**0.5).astype(int)
    # plt.plot(np.rad2deg(xs), ys)
    # plt.show()
    # xs = np.linspace(0, 2*np.pi, 300)
    # ys = np.floor(9*((np.cos(xs) + 1)/2)**0.4)
    # ys = (xs) // (2 * np.pi / (10 - 1))
    #
    # plt.plot(np.rad2deg(xs), ys)
    # # plt.plot(np.rad2deg(xs), np.cos(xs))
    # plt.show()
    # xs = np.linspace(-360,360)
    # ys= xs%360
    #
    # xs = np.linspace(1, 50, 300)
    # ys = np.floor((xs/50)** 0.5 * 10)
    # ys = np.floor((xs / 50) ** 0.5 * (10-1))
    # plt.plot(xs, ys.astype(int))
    # plt.show()

    # xs = np.linspace(-2000, 2000, 4000)
    #
    # # ys = np.floor(((xs + 2000) / 2000) ** 0.5 * (10 - 1)).astype(int)
    # ys = np.floor((xs + 2000)/2000 * (10 - 1)).astype(int)
    # plt.plot(xs, ys.astype(int))
    # plt.show()

    # Data for the bar plot
    categories = ['400-600', '600-800', '800-1000', '1000-1200']
    heights = [[103, 101], [128, 104], [172, 124], [253, 180]]

    categories = ['400-600', '600-800', '800-1000', '1000-1200']
    heights = [[100.6, 100.6], [104.5, 104.1], [125.1, 124.5], [180.6, 173.5]]

    # Separate the heights for red and blue bars
    heights_red = [h[0] for h in heights]
    heights_blue = [h[1] for h in heights]

    # Creating the bar plot
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = range(len(categories))

    bar1 = ax.bar(index, heights_red, bar_width, color='red', label='With Q Learning')
    bar2 = ax.bar([i + bar_width for i in index], heights_blue, bar_width, color='blue', label='Without Q Learning')
    ax.bar_label(bar1)
    ax.bar_label(bar2)
    # Adding labels and title
    ax.set_xlabel('Inter-pair distance [m]')
    ax.set_ylabel('Rate improvement [%]')
    ax.set_title('Comparison of rate improvement for different inter-V2X pair distances')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(categories)
    ax.legend()

    # Display the plot
    plt.show()