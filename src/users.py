#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

from itertools import count
from src.apparatus.rf_transciever import UserRfTransceiver
from src.types_constants import StationType
from src.parameters import *


class User:
    _ids = count(0)

    def __init__(self, user_walker=None):
        self.id = next(self._ids)
        self.user_walker = user_walker
        self.coords = self.user_walker.current_coords
        self.rf_transceiver = UserRfTransceiver(coords=self.coords, user_id=self.id, bandwidth=USER_BANDWIDTH)
