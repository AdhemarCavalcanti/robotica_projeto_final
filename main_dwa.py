import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from config_dwa import Config, RobotType

# --- FUNCOES MATEMATICAS (DWA) ---

def motion(x, u, dt):
    """Modelo de movimento do robô"""
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def calc_dynamic_window(x, config):
    """Calcula a janela dinâmica baseada na velocidade atual e limites"""
    Vs = [config.min_speed, config.max_speed, -config.max_yaw_rate, config.max_yaw_rate]
    
    Vd = [x[3] - config.max_accel * config.dt, 
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt, 
          x[4] + config.max_delta_yaw_rate * config.dt]
    
    return [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), 
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

def predict_trajectory(x_init, v, y, config):
    """Preve a trajetória futura para um par de velocidades (v, w)"""
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt
    return trajectory

def calc_to_goal_cost(trajectory, goal):
    """Custo baseado na distância angular para o objetivo"""
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    return abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

def calc_obstacle_cost(trajectory, ob, config):
    """Custo baseado na proximidade de obstáculos"""
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)
    
    if np.array(r <= config.robot_radius).any():
        return float("Inf")
    
    return 1.0 / np.min(r)

def dwa_control(x, config, goal, ob):
    """Seleciona o melhor comando (v, w)"""
    dw = calc_dynamic_window(x, config)
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            trj = predict_trajectory(x_init, v, y, config)
            
            cost = (config.to_goal_cost_gain * calc_to_goal_cost(trj, goal) + 
                    config.speed_cost_gain * (config.max_speed - trj[-1, 3]) + 
                    config.obstacle_cost_gain * calc_obstacle_cost(trj, ob, config))
            
            if min_cost >= cost:
                min_cost = cost
                best_u = [v, y]
                
    return best_u

# --- INTEGRACAO COM COPPELIASIM ---

def main():
    print("Conectando ao CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    # HANDLES conforme sua imagem do Model Browser
    try:
        # Se os nomes tiverem caminhos diferentes, tente sem o '/Cuboid/' na frente
        robot = sim.getObject('/Cuboid') 
        m_dir = sim.getObject('/Cuboid/MOTOR_DIREITO')
        m_esq = sim.getObject('/Cuboid/MOTOR_ESQUERDO')
        print("Sucesso: Robô e Motores encontrados!")
    except Exception as e:
        print(f"Erro ao encontrar objetos: {e}")
        return

    config = Config()
    
    # Defina o objetivo onde você quer que o robô chegue (X, Y)
    goal = np.array([5.0, 5.0]) 
    
    # Lista de obstáculos (X, Y). O robô tentará desviar destes pontos.
    ob = np.array([
        [2.0, 2.0],
        [3.0, 0.5],
        [1.0, 4.0]
    ])

    sim.startSimulation()
    print("Simulação Iniciada!")

    # Estado inicial do robô: [x, y, yaw, v_linear, v_angular]
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    try:
        while True:
            # 1. Ler dados REAIS do CoppeliaSim
            p = sim.getObjectPosition(robot, sim.handle_world)
            o = sim.getObjectOrientation(robot, sim.handle_world)
            
            # Atualiza o estado atual (x, y, theta)
            x[0], x[1], x[2] = p[0], p[1], o[2]

            # 2. Calcular o melhor comando usando DWA
            u = dwa_control(x, config, goal, ob)

            # 3. Cinemática Inversa (Ajuste R e L para o tamanho do seu robô)
            # R = raio da roda, L = distância entre as duas rodas
            R = 0.05  
            L = 0.20  
            v, w = u[0], u[1]
            
            # Velocidade das rodas em rad/s
            v_dir_roda = (v + (w * L / 2)) / R
            v_esq_roda = (v - (w * L / 2)) / R

            # 4. Enviar comandos para os motores
            sim.setJointTargetVelocity(m_dir, v_dir_roda)
            sim.setJointTargetVelocity(m_esq, v_esq_roda)

            # Verificar se chegou perto o suficiente do objetivo (ex: 20cm)
            if math.hypot(x[0]-goal[0], x[1]-goal[1]) < 0.2:
                print("Objetivo alcançado!")
                sim.setJointTargetVelocity(m_dir, 0)
                sim.setJointTargetVelocity(m_esq, 0)
                break
                
    except Exception as e:
        print(f"Ocorreu um erro durante a execução: {e}")
    finally:
        # Para a simulação se o código for interrompido
        sim.stopSimulation()
        print("Simulação finalizada.")

if __name__ == '__main__':
    main()