from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
from config_dwa import Config
from dwa_logic import dwa_control

def main():
    print("--- Iniciando Conexão com CoppeliaSim ---")
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    # Handles do Robô (Baseado na sua imagem da hierarquia)
    try:
        robot = sim.getObject('/Cuboid')
        m_dir = sim.getObject('/Cuboid/MOTOR_DIREITO')
        m_esq = sim.getObject('/Cuboid/MOTOR_ESQUERDO')
        
        # Sensores de Proximidade (Ultrassônicos)
        s_names = [
            '/SENSOR_MEIO', 
            '/SENSOR_DIREITO', 
            '/SENSOR_DIAG_DIREITO', 
            '/SENSOR_ESQUERDO', 
            '/SENSOR_DIAG_ESQUERDO'
        ]
        sensors = [sim.getObject(n) for n in s_names]
        print("Sucesso: Todos os objetos foram encontrados na cena.")
    except Exception as e:
        print(f"Erro ao localizar objetos no CoppeliaSim: {e}")
        return

    # Inicialização de parâmetros
    config = Config()
    goal = np.array([5.0, 5.0]) # Coordenada do objetivo (X, Y)
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # Estado: [x, y, yaw, v, w]

    sim.startSimulation()
    print("Simulação Iniciada no CoppeliaSim.")

    try:
        while True:
            # 1. Posição e Orientação real do robô
            p = sim.getObjectPosition(robot, sim.handle_world)
            o = sim.getObjectOrientation(robot, sim.handle_world)
            x[0:3] = [p[0], p[1], o[2]]

            # 2. Leitura dos Sensores (Mapeamento de Obstáculos em tempo real)
            ob_list = []
            for s in sensors:
                res, dist, pt, _, _ = sim.readProximitySensor(s)
                if res > 0:
                    # Converte o ponto detectado (local do sensor) para coordenadas do mundo
                    m = sim.getObjectMatrix(s, sim.handle_world)
                    p_abs = sim.multiplyVector(m, pt)
                    ob_list.append([p_abs[0], p_abs[1]])
            
            # Se não houver obstáculos próximos, envia um ponto fictício muito distante
            ob = np.array(ob_list) if ob_list else np.array([[99.0, 99.0]])

            # 3. Inteligência DWA (Calcula as melhores velocidades v e w)
            u = dwa_control(x, config, goal, ob)

            # 4. Cinemática Diferencial (Conversão para motores rad/s)
            R = 0.05  # Raio da roda (ajuste conforme o robô do laboratório)
            L = 0.20  # Distância entre rodas (ajuste conforme o robô do laboratório)
            v, w = u[0], u[1]
            
            v_dir = (v + (w * L / 2)) / R
            v_esq = (v - (w * L / 2)) / R
            
            sim.setJointTargetVelocity(m_dir, v_dir)
            sim.setJointTargetVelocity(m_esq, v_esq)

            # 5. Feedback no Terminal
            dist_to_goal = math.hypot(x[0]-goal[0], x[1]-goal[1])
            print(f"Alvo: ({goal[0]},{goal[1]}) | Dist: {dist_to_goal:.2f}m | V: {v:.2f} W: {w:.2f}", end='\r')

            # Condição de parada
            if dist_to_goal < 0.2:
                print("\n\n[FINALIZADO] Objetivo atingido com sucesso!")
                sim.setJointTargetVelocity(m_dir, 0)
                sim.setJointTargetVelocity(m_esq, 0)
                break
                
    except Exception as e:
        print(f"\n[ERRO NA EXECUÇÃO]: {e}")
    finally:
        # Garante que a simulação pare se o script for interrompido
        sim.stopSimulation()
        print("Simulação encerrada.")

if __name__ == "__main__":
    main()