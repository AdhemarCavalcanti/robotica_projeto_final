from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
from config_dwa import Config
from dwa_logic import dwa_control

def main():
    print("--- Iniciando Navegação DWA: Robô do Laboratório ---")
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    # 1. Definição dos Caminhos (Baseado na sua Scene Hierarchy)
    # Na sua imagem, os itens estão dentro de /Cuboid
    try:
        robot = sim.getObject('/Cuboid')
        m_dir = sim.getObject('/Cuboid/MOTOR_DIREITO')
        m_esq = sim.getObject('/Cuboid/MOTOR_ESQUERDO')
        
        # Lista de Sensores conforme a hierarquia da imagem
        s_names = [
            '/Cuboid/SENSOR_MEIO', 
            '/Cuboid/SENSOR_DIREITO', 
            '/Cuboid/SENSOR_DIAG_DIREITO', 
            '/Cuboid/SENSOR_ESQUERDO', 
            '/Cuboid/SENSOR_DIAG_ESQUERDO'
        ]
        sensors = [sim.getObject(n) for n in s_names]
        print("✅ Conexão estabelecida: Robô, Motores e 5 Sensores prontos.")
    except Exception as e:
        print(f"❌ Erro ao localizar componentes. Verifique os nomes no CoppeliaSim: {e}")
        return

    # 2. Inicialização de Parâmetros
    config = Config()
    # Define o alvo (ajuste as coordenadas X e Y conforme desejar)
    goal = np.array([3.0, 3.0]) 
    # Estado inicial: [x, y, yaw, v, w]
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 

    sim.startSimulation()
    print(f"🚀 Simulação iniciada! Destino: {goal}")

    try:
        while True:
            # --- PERCEPÇÃO ---
            # Pegar posição e orientação real no mundo
            p = sim.getObjectPosition(robot, sim.handle_world)
            o = sim.getObjectOrientation(robot, sim.handle_world)
            x[0:3] = [p[0], p[1], o[2]]

            # Mapeamento dos Obstáculos via Sensores Ultrassônicos
            ob_list = []
            for s in sensors:
                # res: 1 se detectar algo, dist: distância, pt: ponto relativo ao sensor
                res, dist, pt, _, _ = sim.readProximitySensor(s)
                if res > 0:
                    # Converte o ponto local do sensor para coordenadas globais do mundo
                    matrix = sim.getObjectMatrix(s, sim.handle_world)
                    p_abs = sim.multiplyVector(matrix, pt)
                    ob_list.append([p_abs[0], p_abs[1]])
            
            # Converte lista para array numpy (ou coloca ponto distante se vazio)
            ob = np.array(ob_list) if ob_list else np.array([[99.0, 99.0]])

            # --- PROCESSAMENTO (DWA) ---
            # O coração do método: escolhe a melhor velocidade desviando dos obstáculos
            u = dwa_control(x, config, goal, ob)

            # --- AÇÃO (Cinemática Diferencial) ---
            R = 0.045  # Raio da roda (ajuste conforme o modelo real)
            L = 0.16   # Distância entre rodas (ajuste conforme o modelo real)
            v, w = u[0], u[1]
            
            # Cálculo da velocidade para cada motor
            v_dir = (v + (w * L / 2)) / R
            v_esq = (v - (w * L / 2)) / R
            
            sim.setJointTargetVelocity(m_dir, v_dir)
            sim.setJointTargetVelocity(m_esq, v_esq)

            # --- FEEDBACK ---
            dist_to_goal = math.hypot(x[0]-goal[0], x[1]-goal[1])
            print(f"Distância: {dist_to_goal:.2f}m | V: {v:.2f} | W: {w:.2f}", end='\r')

            # Condição de parada (chegou a 20cm do alvo)
            if dist_to_goal < 0.2:
                print("\n\n🏁 [DESTINO ATINGIDO] Desligando motores.")
                sim.setJointTargetVelocity(m_dir, 0)
                sim.setJointTargetVelocity(m_esq, 0)
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário.")
    except Exception as e:
        print(f"\n⚠️ Ocorreu um erro: {e}")
    finally:
        sim.stopSimulation()
        print("📴 Conexão encerrada.")

if __name__ == "__main__":
    main()