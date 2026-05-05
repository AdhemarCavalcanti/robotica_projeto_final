import math

class Config:
    def __init__(self):
        # Parâmetros físicos do robô
        self.max_speed = 0.5        # Velocidade máxima (m/s) - reduzi para segurança
        self.min_speed = 0.0        # Mínimo 0 para o robô não dar ré sem ver
        self.max_yaw_rate = 40.0 * math.pi / 180.0
        self.max_accel = 0.2
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0
        self.v_resolution = 0.01
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0
        self.dt = 0.1
        self.predict_time = 3.0
        
        # Pesos do algoritmo (Ganhos)
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001
        self.robot_radius = 0.3     # Raio de segurança do robô