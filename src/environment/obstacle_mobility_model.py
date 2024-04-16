from src.parameters import *
from src.environment.obstacles import get_madrid_buildings, get_poznan_buildings





class ObstaclesMobilityModel:
    def __init__(self, time_step=TIME_STEP, number_of_users=NUM_OF_USERS):
        self.obstacles_objects = get_madrid_buildings() if not USE_POZNAN else get_poznan_buildings()
    def get_obstacles(self):
        return self.obstacles_objects.obstaclesList