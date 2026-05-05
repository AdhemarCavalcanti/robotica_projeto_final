import math
import numpy as np

def motion(x, u, dt):
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def dwa_control(x, config, goal, ob):
    # Janela Dinâmica
    Vs = [config.min_speed, config.max_speed, -config.max_yaw_rate, config.max_yaw_rate]
    Vd = [x[3]-config.max_accel*config.dt, x[3]+config.max_accel*config.dt,
          x[4]-config.max_delta_yaw_rate*config.dt, x[4]+config.max_delta_yaw_rate*config.dt]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    min_cost = float("inf")
    best_u = [0.0, 0.0]

    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            # Predição de trajetória
            traj_x = np.array(x)
            time = 0
            while time <= config.predict_time:
                traj_x = motion(traj_x, [v, y], config.dt)
                time += config.dt
            
            # Cálculo de Custos
            dx, dy = goal[0] - traj_x[0], goal[1] - traj_x[1]
            error_angle = math.atan2(dy, dx)
            cost_goal = abs(math.atan2(math.sin(error_angle - traj_x[2]), math.cos(error_angle - traj_x[2])))
            cost_speed = config.max_speed - traj_x[3]
            
            # Custo de Obstáculo
            dist_ob = np.hypot(ob[:, 0] - traj_x[0], ob[:, 1] - traj_x[1])
            cost_ob = float("Inf") if np.any(dist_ob <= config.robot_radius) else 1.0 / np.min(dist_ob)

            final_cost = config.to_goal_cost_gain*cost_goal + config.speed_cost_gain*cost_speed + config.obstacle_cost_gain*cost_ob

            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
    return best_u