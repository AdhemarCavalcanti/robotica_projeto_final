import math
import sys
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from robot_controller import DifferentialRobot
from dwa_navigation import DWA_Planner 

def calcular_distancia_ponto_linha(x0, y0, x1, y1, x2, y2):
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den if den != 0 else 0.0

def transformar_leituras_em_obstaculos(x_rob, y_rob, yaw_rob, dist_sensores):
    obstaculos = []
    angulos_sensores = {
        'SENSOR_ESQUERDO': math.radians(90),
        'SENSOR_DIAG_ESQUERDO': math.radians(45),
        'SENSOR_MEIO': math.radians(0),
        'SENSOR_DIAG_DIREITO': math.radians(-45),
        'SENSOR_DIREITO': math.radians(-90)
    }
    for nome, dist in dist_sensores.items():
        if dist != float('inf'):
            angulo_global = yaw_rob + angulos_sensores[nome]
            ob_x = x_rob + dist * math.cos(angulo_global)
            ob_y = y_rob + dist * math.sin(angulo_global)
            obstaculos.append([ob_x, ob_y])
    return obstaculos

def calcular_melhor_direcao_escape(dist_sensores):
    esq = (dist_sensores['SENSOR_ESQUERDO'] if dist_sensores['SENSOR_ESQUERDO'] != float('inf') else 1.5) + \
          (dist_sensores['SENSOR_DIAG_ESQUERDO'] if dist_sensores['SENSOR_DIAG_ESQUERDO'] != float('inf') else 1.5)
          
    dir_ = (dist_sensores['SENSOR_DIREITO'] if dist_sensores['SENSOR_DIREITO'] != float('inf') else 1.5) + \
           (dist_sensores['SENSOR_DIAG_DIREITO'] if dist_sensores['SENSOR_DIAG_DIREITO'] != float('inf') else 1.5)
           
    return 1 if esq >= dir_ else -1

def main():
    print("Conectando ao CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    robo = DifferentialRobot(sim, motor_esq_name='MOTOR_ESQUERDO', motor_dir_name='MOTOR_DIREITO')
    dwa = DWA_Planner()
    
    NOME_ROBO = '/Cuboid'
    NOME_ALVO = '/Alvo'
    DISTANCIA_TOLERANCIA = 0.25  
    
    try:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_state_stopped: time.sleep(0.05)
    except: pass

    sim.setStepping(True)
    sim.startSimulation()
    client.step()
    sim.step()
    
    x_ini, y_ini = robo.obter_posicao_objeto(NOME_ROBO)
    x_alvo_antigo, y_alvo_antigo = robo.obter_posicao_objeto(NOME_ALVO)
    
    modo_contorno = False
    lado_contorno = 1             
    distancia_ponto_impacto = float('inf')
    
    print("=" * 60)
    print(" [SISTEMA BLINDADO] GIRO FORÇADO NO EIXO SEM LOGICA DE RÉ")
    print("=" * 60)

    while True:
        client.step()
        sim.step()

        x_atual, y_atual = robo.obter_posicao_objeto(NOME_ROBO)
        yaw_atual = robo.obter_orientacao_robo(NOME_ROBO)
        x_alvo, y_alvo = robo.obter_posicao_objeto(NOME_ALVO)
        
        distancia_alvo = math.hypot(x_alvo - x_atual, y_alvo - y_atual)
        
        if distancia_alvo <= DISTANCIA_TOLERANCIA:
            print(f"\n\n[SUCESSO] Alvo Alcançado!")
            robo.parar()
            sim.stopSimulation()
            break 

        # Rastreamento dinâmico se o alvo mudar de posição
        if math.hypot(x_alvo - x_alvo_antigo, y_alvo - y_alvo_antigo) > 0.4:
            x_ini, y_ini = x_atual, y_atual
            x_alvo_antigo, y_alvo_antigo = x_alvo, y_alvo
            modo_contorno = False 

        dist_sensores = robo.ler_distancias_separadas()
        obstaculos_mapeados = transformar_leituras_em_obstaculos(x_atual, y_atual, yaw_atual, dist_sensores)

        dx = x_alvo - x_atual
        dy = y_alvo - y_atual
        angulo_alvo = math.atan2(dy, dx)
        erro_rumo = math.atan2(math.sin(angulo_alvo - yaw_atual), math.cos(angulo_alvo - yaw_atual))

        # ─── REGRA SUPREMA: APONTAR DIRETAMENTE PARA O ALVO SE ELE MUDAR ───
        if abs(erro_rumo) > math.radians(45):
            # Força o robô a girar estritamente parado no lugar até alinhar com o vetor do alvo
            velocidade_giro = 1.3 if erro_rumo > 0 else -1.3
            robo.mover(0.0, velocidade_giro)
            sys.stdout.write(f"\r[RASTREAMENTO] Alvo mudou! Girando no eixo: {math.degrees(erro_rumo):.1f}°   ")
            sys.stdout.flush()
            continue 

        # Detecção de obstáculo à frente
        bloqueio_frontal = (dist_sensores['SENSOR_MEIO'] < 0.55 or 
                            (dist_sensores['SENSOR_DIAG_ESQUERDO'] < 0.42 and erro_rumo > 0) or 
                            (dist_sensores['SENSOR_DIAG_DIREITO'] < 0.42 and erro_rumo < 0))

        # Máquina de estados Bug-2
        if not modo_contorno:
            if bloqueio_frontal:
                modo_contorno = True
                distancia_ponto_impacto = distancia_alvo
                lado_contorno = calcular_melhor_direcao_escape(dist_sensores)
        else:
            dist_linha = calcular_distancia_ponto_linha(x_atual, y_atual, x_ini, y_ini, x_alvo, y_alvo)
            if dist_linha < 0.18 and distancia_alvo < (distancia_ponto_impacto - 0.20) and not bloqueio_frontal:
                modo_contorno = False

        if not modo_contorno:
            alvo_dwa_x, alvo_dwa_y = x_alvo, y_alvo
        else:
            angulo_parede = yaw_atual + (math.pi / 2 * lado_contorno)
            alvo_dwa_x = x_atual + 0.35 * math.cos(angulo_parede)
            alvo_dwa_y = y_atual + 0.35 * math.sin(angulo_parede)

        # ─── INTERCEPTOR MANUAL DE GIRO PURO (SUBSTITUIÇÃO TOTAL DO DWA EM CRISES) ───
        limite_critico = 0.38
        
        # Se os sensores indicarem que bateu de frente ou a quina está muito perto, 
        # ignoramos completamente o DWA para eliminar qualquer tentativa matemática de recuar.
        if dist_sensores['SENSOR_MEIO'] < limite_critico or dist_sensores['SENSOR_DIAG_ESQUERDO'] < (limite_critico - 0.05) or dist_sensores['SENSOR_DIAG_DIREITO'] < (limite_critico - 0.05):
            
            # Escolhe para onde girar baseado no lado livre dos sensores
            sentido_giro = calcular_melhor_direcao_escape(dist_sensores)
            
            # Em vez de enviar (v, w) pro DWA decodificar, injetamos diretamente um giro simétrico perfeito
            # 1.2 rad/s positivo ou negativo força rotação pura sobre o próprio centro geométrico do robô.
            melhor_v = 0.0
            melhor_w = 1.3 * sentido_giro
            
            sys.stdout.write(f"\r[GIRO NO EIXO] Quina ativa! Ajustando angulação parado no lugar... ")
        else:
            # Caminho livre à frente: o DWA assume o controle normal para navegar
            melhor_v, melhor_w = dwa.planejar(x_atual, y_atual, yaw_atual, alvo_dwa_x, alvo_dwa_y, obstaculos_mapeados)

        # ─── TRAVA DE SEGURANÇA CONTRA QUALQUER VELOCIDADE NEGATIVA RESIDUAL ───
        if melhor_v < 0.0:
            melhor_v = 0.0
            melhor_w = 1.3 * lado_contorno

        # Executa o comando de movimento limpo
        robo.mover(melhor_v, melhor_w)
        sys.stdout.flush()

if __name__ == '__main__':
    main()