import math
import numpy as np
from enum import Enum

class RobotType(Enum):
    circle = 0
    rectangle = 1

class Config:
    def __init__(self):
        # Parametros do robo
        self.max_speed = 1.0  # m/s
        self.min_speed = -0.5 # m/s
        self.max_yaw_rate = 40.0 * math.pi / 180.0
        self.max_accel = 0.2
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0
        self.v_resolution = 0.01
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0
        self.dt = 0.1
        self.predict_time = 3.0
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001
        self.robot_type = RobotType.circle
        self.robot_radius = 0.5 # Raio de seguranca