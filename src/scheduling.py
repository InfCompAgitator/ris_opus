import numpy as np
from src.parameters import TIME_STEP, RELAYS_BANDWIDTH_5G, rng, TX_POWER_VEHICLE, DEBUG
from sortedcontainers import SortedList
from src.environment.vehicular_streets import Vehicle
from src.channel_model.mmwave_modeling import get_throughput_5g

SLOT_DURATION = 1e-3
RATE_HIST_SIZE = 10 * TIME_STEP / SLOT_DURATION
PAYLOAD_RANGE = [200e12, 500e12]  # b
THROUGHPUT_AVERAGE_ALPHA = 0.5


class ActiveLink:
    def __init__(self, t_1: Vehicle, t_2: Vehicle, stats, ris_pl, direct_power):
        self.t_1 = t_1
        self.t_2 = t_2
        self.payload: int = rng.uniform(PAYLOAD_RANGE[0], PAYLOAD_RANGE[1])
        self.throughput_avg = 0
        self.direct_power, self.ris_pl = direct_power, ris_pl
        self.current_rate = None
        self.update_current_rate(ris_pl)
        self.priority = self.current_rate  # Priority is current_rate/average_throughput
        self.slots_assigned = 0
        self.stats = stats
        self.prev_stats = stats
        self.previous_payload = self.payload
        self.previous_payload_2 = self.payload
        self.payload_served = None
        self.previous_payload_served = None
        self.rate_improvement = 100

    def update_current_rate(self, ris_pl):
        ris_power = TX_POWER_VEHICLE / ris_pl
        self.current_rate = get_throughput_5g(ris_power + self.direct_power)
        if DEBUG:
            direct_rate = get_throughput_5g(self.direct_power)
            self.rate_improvement = 100 * self.current_rate / direct_rate
            if self.current_rate > direct_rate * 1.05:
                print("Direct Rate:", direct_rate,"Rate Improvement:", 100 * self.current_rate / direct_rate)

    def update_priority(self):
        if self.throughput_avg != 0:
            self.priority = self.current_rate / self.throughput_avg
        else:
            self.priority = self.current_rate

    def update_throughput_average(self, new_throughput):
        new_throughput = (
                                 1 - THROUGHPUT_AVERAGE_ALPHA) * self.throughput_avg + THROUGHPUT_AVERAGE_ALPHA * new_throughput
        self.throughput_avg = np.where(new_throughput > 1e-100, new_throughput, 0)

    def __eq__(self, other):
        return self.priority == other.priority

    def __lt__(self, other):
        return self.priority < other.priority


class Pfs:
    single_pair = None
    single_pair_2 = None

    def __init__(self):
        self.active_pairs = []

    def add_pair(self, t_1, t_2, stats, ris_pl, direct_power):
        new_link = ActiveLink(t_1, t_2, stats, ris_pl, direct_power)
        self.active_pairs.append(new_link)

    def update_links(self, t_step):
        n_slots = t_step / SLOT_DURATION
        if self.single_pair:
            _l = self.single_pair
            if not self.single_pair.t_1.in_simulation or not self.single_pair.t_2.in_simulation:
                self.remove_pair(self.single_pair)
            else:
                n_slots_active = min(n_slots, _l.slots_assigned)
                _l.slots_assigned -= n_slots_active
                _l.slots_assigned = max(_l.slots_assigned, 0)
                _payload = min(_l.current_rate * SLOT_DURATION * n_slots_active, _l.payload)
                _l.previous_payload_2 = _l.previous_payload
                _l.previous_payload = _l.payload
                _l.previous_payload_served = _l.payload_served if _l.payload_served is not None else _payload
                _l.payload_served = _payload
                _l.payload -= _payload
                if _l.payload <= 0:
                    self.remove_pair(_l)
                else:
                    _l.update_throughput_average(_payload / t_step)
            return

        for _l in self.active_pairs[:]:
            if not _l.t_1.in_simulation or not _l.t_2.in_simulation:
                self.remove_pair(_l)
                continue
            n_slots_active = min(n_slots, _l.slots_assigned)
            _l.slots_assigned -= n_slots_active
            _l.slots_assigned = max(_l.slots_assigned, 0)
            _payload = min(_l.current_rate * SLOT_DURATION * n_slots_active, _l.payload)
            _l.previous_payload_2 = _l.previous_payload
            _l.previous_payload = _l.payload
            _l.previous_payload_served = _l.payload_served if _l.payload_served is not None else _payload
            _l.payload_served = _payload
            _l.payload -= _payload
            if _l.payload <= 0:
                self.remove_pair(_l)
                continue
            _l.update_throughput_average(_payload / t_step)

    def remove_pair(self, pair: ActiveLink):
        if self.single_pair is not None and pair == self.single_pair:
            self.single_pair = self.single_pair_2
        pair.t_1.color = 'k'
        pair.t_1.data_transfer_flag = False
        pair.t_2.color = 'k'
        pair.t_2.data_transfer_flag = False
        if self.single_pair_2:
            self.single_pair_2.t_1.color = 'blue'
            self.single_pair_2.t_2.color = 'blue'
            self.single_pair_2 = None
        self.active_pairs.remove(pair)

    def schedule_slots(self, t_step):
        if not self.active_pairs:
            return
        n_slots = t_step / SLOT_DURATION
        if self.single_pair:
            slots_assigned = n_slots * np.ones(len(self.active_pairs))
            for idx, _n_slots in enumerate(slots_assigned):
                self.active_pairs[idx].slots_assigned = _n_slots
            return
        priorities = np.array([l.priority for l in self.active_pairs])
        priorities_sum = priorities.sum()
        slots_assigned = np.array(
            np.floor(n_slots * priorities / (priorities_sum if priorities_sum != 0 else 1))).astype(
            int)
        n_slots -= slots_assigned.sum()
        while n_slots > 0:
            for idx in range(len(self.active_pairs)):
                n_slots -= 1
                slots_assigned[idx] += 1
                if n_slots <= 0:
                    break
        for idx, _n_slots in enumerate(slots_assigned):
            self.active_pairs[idx].slots_assigned = _n_slots
        # print("n_slots:", [_pair.slots_assigned for _pair in self.active_pairs])
        # print("rate:", [_pair.current_rate / 1e6 for _pair in self.active_pairs])

    def update_link_rate(self, link, ris_pl):
        link.update_current_rate(ris_pl)
        link.update_priority()

    def get_stats(self):
        if self.single_pair:
            return self.single_pair.payload_served / self.single_pair.previous_payload, \
                   self.single_pair.previous_payload_served / self.single_pair.previous_payload_2
        payload_served, previous_payload, previous_payload_served, previous_payload_2 = 0, 0, 0, 0
        for _l in self.active_pairs[:]:
            previous_payload += _l.previous_payload
            previous_payload_served += _l.previous_payload_served
            payload_served += _l.payload_served
            previous_payload_2 += _l.previous_payload_2
        return payload_served / previous_payload, previous_payload_served / previous_payload_2


if __name__ == '__main__':
    p = Pfs()
    p.add_pair(1, 1, 50e6)
    p.add_pair(1, 1, 10e6)
    p.add_pair(1, 1, 10e6)
    p.add_pair(1, 1, 20e6)

    p.schedule_slots(1)
    priorities_1 = [_.priority for _ in p.active_pairs]
    payloads_1 = [_.payload for _ in p.active_pairs]
    slots_assigned_1 = [_.slots_assigned for _ in p.active_pairs]
    p.update_links(1)
    [p.update_link_rate(l, l.current_rate) for l in p.active_pairs]
    [_.payload for _ in p.active_pairs]
    while 1:
        p.schedule_slots(1)
        print("Assinged Slots:", [_.slots_assigned for _ in p.active_pairs])
        p.update_links(1)
        [p.update_link_rate(l, l.current_rate) for l in p.active_pairs]
        print("Remaining Payloads:", [_.payload for _ in p.active_pairs])
        print("Priorities:", [_.priority for _ in p.active_pairs])
        print("Throughput Avgs:", [_.throughput_avg for _ in p.active_pairs])
