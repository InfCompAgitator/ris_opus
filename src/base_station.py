#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from itertools import count
from src.data_structures import Coords3d
from src.apparatus.rf_transciever import MacroRfTransceiver
from src.parameters import MBS_BANDWIDTH, MBS_TX_POWER_RF, DEFAULT_CARRIER_FREQ_MBS, MBS_LOCATIONS, DEFAULT_A3_INDIV_OFFSET
from src.types_constants import StationType


class BaseStation:
    _ids = count(0, -1)

    def __init__(self, bs_id: int = None, coords: Coords3d = MBS_LOCATIONS[0], station_type=StationType.UMa):
        self.id = next(self._ids) if bs_id is None else bs_id
        self.coords = coords.copy()
        self.station_type = station_type
        self.rf_transceiver = MacroRfTransceiver(coords=self.coords, t_power=MBS_TX_POWER_RF, bandwidth=MBS_BANDWIDTH,
                                                 bs_id=self.id, carrier_frequency=DEFAULT_CARRIER_FREQ_MBS,
                                                 station_type=station_type)

    @staticmethod
    def reset_ids():
        BaseStation._ids = count(0, -1)
