import numpy as np
from src.parameters import RELAYS_TRANSMISSION_POWER, RELAYS_NOISE_POWER, RELAYS_BANDWIDTH, MAXIMUM_PL, ACTIVE_RIS_GAIN, \
    RELAYS_BANDWIDTH_5G_NOISE, RELAYS_BANDWIDTH_5G, EFFICIENCY_5G_RATE_FACTOR, RELAYS_NOISE_POWER_5G
from src.math_tools import db2lin, lin2db


def get_ris_path_loss(d_p1_r, d_r_p2, active_ris=False):
    if not active_ris:
        return 39 + 10 * np.log10(3 ** 2 * (d_p1_r + d_r_p2) ** 2.13)
    else:
        return 39 + 10 * np.log10(3 ** 2 * (d_p1_r + d_r_p2) ** 2.13) - ACTIVE_RIS_GAIN


def get_simple_pl(distance):
    return 39 + 10 * 2.13 * np.log10(distance)


def get_throughput(rx_power, noise_power=RELAYS_NOISE_POWER, bandwidth=RELAYS_BANDWIDTH):
    return bandwidth * np.log2(1 + rx_power / noise_power)


def get_throughput_5g(rx_power, noise_power=RELAYS_NOISE_POWER_5G, bandwidth=RELAYS_BANDWIDTH_5G):
    efficiency = EFFICIENCY_5G_RATE_FACTOR
    return efficiency * bandwidth * np.log2(1 + rx_power / noise_power)


def get_required_snr_pl_5g(rate, noise_power=RELAYS_NOISE_POWER_5G, bandwidth=RELAYS_BANDWIDTH_5G, tx_power=RELAYS_TRANSMISSION_POWER):
    efficiency = EFFICIENCY_5G_RATE_FACTOR
    snr = 2 ** (rate / (efficiency * bandwidth)) - 1
    maximum_pl = snr * noise_power / tx_power
    return snr, maximum_pl


def get_required_rx_power(rate, noise_power=RELAYS_NOISE_POWER, bandwidth=RELAYS_BANDWIDTH):
    return (2 ** (rate / bandwidth) - 1) * noise_power


if __name__ == '__main__':
    print(get_throughput(db2lin((lin2db(RELAYS_TRANSMISSION_POWER) - MAXIMUM_PL))) / 1e6)
    print(lin2db(RELAYS_TRANSMISSION_POWER / get_required_rx_power(1e6)))
    get_required_snr_pl_5g(100e6)
