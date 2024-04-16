import numpy as np
from numpy.random import default_rng
from src.parameters import TIME_STEP, RELAYS_BANDWIDTH_5G
from sortedcontainers import SortedList

SLOT_DURATION = 1e-3
RATE_HIST_SIZE = 10 * TIME_STEP / SLOT_DURATION
PAYLOAD_RANGE = [500e6, 500e6]  # b
rng = default_rng()
THROUGHPUT_AVERAGE_ALPHA = 0.5


class ActiveLink:
    def __init__(self, t_1, t_2, current_rate):
        self.t_1 = t_1
        self.t_2 = t_2
        self.payload : int = rng.uniform(PAYLOAD_RANGE[0], PAYLOAD_RANGE[1])
        self.rate_hist = np.zeros(int(RATE_HIST_SIZE))
        self.priority = current_rate  # Priority is current_rate/average_throughput
        self.current_rate = current_rate
        self.slot_hist_idx: int = 0
        self.slots_assigned: int = 0
        self.slots_active = 0

    def update_rate_hist(self, n_slots):
        high_idxs = np.arange(self.slot_hist_idx, min(n_slots, RATE_HIST_SIZE), dtype=int)
        low_idxs = np.arange(0, max(n_slots - len(high_idxs), 0), dtype=int)
        self.rate_hist[np.r_[high_idxs, low_idxs]] = self.current_rate
        self.slot_hist_idx = int((self.slot_hist_idx + n_slots) % RATE_HIST_SIZE)

    def update_current_rate(self, current_rate):
        self.current_rate = current_rate
        self.priority = current_rate / self.rate_hist[0:self.slots_active].mean()

    def __eq__(self, other):
        return self.priority == other.priority

    def __lt__(self, other):
        return self.priority < other.priority




class Pfs:
    def __init__(self):
        self.active_pairs = []

    def add_pair(self, t_1, t_2, current_rate):
        new_link = ActiveLink(t_1, t_2, current_rate)
        self.active_pairs.append(new_link)

    def update_links(self, t_step):
        n_slots = t_step / SLOT_DURATION
        for _l in self.active_pairs:
            n_slots_active = min(n_slots, _l.slots_assigned)
            _l.slots_assigned -= n_slots_active
            _l.slots_assigned = max(_l.slots_assigned, 0)
            _l.update_rate_hist(n_slots_active)
            _payload = _l.current_rate * SLOT_DURATION * n_slots_active
            _l.payload -= _payload
            if _l.payload <= 0:
                self.active_pairs.remove(_l)
            # _l.n_slots_active += n_slots_active
            # _l.n_slots_active = _l.n_slots_active % RATE_HIST_SIZE

    def schedule_slots(self, t_step):
        n_slots = t_step / SLOT_DURATION
        priorities = np.array([l.priority for l in self.active_pairs])
        slots_assigned = np.floor(n_slots * priorities / priorities.sum()).astype(int)
        n_slots -= slots_assigned.sum()
        for idx, _n_slots in enumerate(slots_assigned):
            self.active_pairs[idx].slots_assigned = _n_slots
        idx = 0
        while n_slots > 0:
            n_slots -= 1
            slots_assigned[idx] += 1

        # n_slots = t_step / SLOT_DURATION
        # d_priorities = np.diff([l.priority for l in self.active_pairs])
        # rate_hist_sums = [l.rate_hist[0:l.n_slots_active].sum() for l in self.active_pairs]
        # done = False
        # for idx, d_priority in enumerate(d_priorities):
        #     for i in range(idx + 1):
        #         best_pair = self.active_pairs[i]
        #         if best_pair.n_slots_active == 0:
        #             n_slots_new = np.ceil(d_priority / best_pair.current_rate)
        #         else:
        #             rate_hist_sum = rate_hist_sums[i]
        #             n_slots_new = np.ceil(d_priority * rate_hist_sum ** 2 / (best_pair.current_rate * (
        #                     best_pair.current_rate * best_pair.n_slots_active - rate_hist_sum + d_priority * rate_hist_sum)))
        #         best_pair.slots_assigned += min(n_slots_new, n_slots)
        #         n_slots -= n_slots_new
        #         if n_slots <= 0:
        #             done = True
        #             break
        #     if done:
        #         break
        # self.active_pairs[-1].slots_assigned += 1
        # n_slots -= 1
        #
        # def get_new_priority(link: ActiveLink, new_slots, idx):
        #     return link.current_rate * (link.n_slots_active + link.slots_assigned + new_slots) / (
        #                 rate_hist_sums[idx] + (link.slots_assigned + new_slots) * link.current_rate)
        #
        # new_priorities = [get_new_priority(link, 0, idx) for idx, link in enumerate(self.active_pairs)]
        # while n_slots > 0:
        #     idx = np.argmax(new_priorities)
        #     _pair = self.active_pairs[idx]
        #     n_slots -= 1
        #     _pair.slots_assigned += 1
        #     new_priorities[idx] = get_new_priority(_pair, 1, idx)  # Make more slots to make faster

    def update_link_rate(self, link, new_rate):
        # self.active_pairs.remove(link)
        link.update_current_rate(new_rate)
        # self.active_pairs.append(link)


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

