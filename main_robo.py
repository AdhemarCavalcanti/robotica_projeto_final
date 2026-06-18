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
    print(" [SISTEMA BLINDADO] GIRO PIVOTADO - ZERO MARCHA RÉ")
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

        # Reseta busca se o alvo mudar dinamicamente de lugar
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

        # ─── REALINHAMENTO INICIAL/CRÍTICO NO PRÓPRIO EIXO SEGURO ───
        if abs(erro_rumo) > math.radians(45):
            velocidade_giro = 1.4 if erro_rumo > 0 else -1.4
            # Ativa a trava anti-ré mesmo girando parado para alinhar com o alvo
            robo.mover(0.0, velocidade_giro, forçar_avanco=True)
            sys.stdout.write(f"\r[ALINHAMENTO] Buscando ângulo do alvo: {math.degrees(erro_rumo):.1f}°   ")
            sys.stdout.flush()
            continue 

        # Zoneamento preventivo de colisões
        limite_critico = 0.50  
        limite_frontal = 0.65  

        bloqueio_frontal = (dist_sensores['SENSOR_MEIO'] < limite_frontal or 
                            (dist_sensores['SENSOR_DIAG_ESQUERDO'] < limite_critico and erro_rumo > 0) or 
                            (dist_sensores['SENSOR_DIAG_DIREITO'] < limite_critico and erro_rumo < 0))

        # Máquina de estados Bug-2 tradicional
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

        # Flag de interceptação manual
        bloqueio_ativo = False

        # ─── INTERCEPTOR MANUAL DE MANOBRA EM PIVÔ FRENTE ───
        if dist_sensores['SENSOR_MEIO'] < limite_frontal or dist_sensores['SENSOR_DIAG_ESQUERDO'] < limite_critico or dist_sensores['SENSOR_DIAG_DIREITO'] < limite_critico:
            
            bloqueio_ativo = True
            sentido_giro = calcular_melhor_direcao_escape(dist_sensores)
            
            # Definimos v_linear baixa e w_angular alta. Juntando com a trava nas rodas, 
            # a roda de dentro para e a de fora empurra o robô para a frente e para o lado.
            melhor_v = 0.02  
            melhor_w = 1.6 * sentido_giro  
            
            sys.stdout.write(f"\r[CRÍTICO] Pivotando para frente/lateral. Lado livre: {sentido_giro} ")
        else:
            # Caminho livre à frente: DWA gerencia as velocidades fluidas
            melhor_v, melhor_w = dwa.planejar(x_atual, y_atual, yaw_atual, alvo_dwa_x, alvo_dwa_y, obstaculos_mapeados)

        # ─── SEGUNDA TRAVA DE SEGURANÇA ANTIDRIVETRAIN INVERSO ───
        if melhor_v < 0.0:
            melhor_v = 0.02
            melhor_w = 1.4 * lado_contorno
            bloqueio_ativo = True

        # Envia comandos de movimentação injetando a flag anti-ré ativa em momentos de crise
        robo.mover(melhor_v, melhor_w, forçar_avanco=bloqueio_ativo)
        sys.stdout.flush()

if __name__ == '__main__':
    main()