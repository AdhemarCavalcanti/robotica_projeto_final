import math
import sys
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from robot_controller import DifferentialRobot

def calcular_distancia_ponto_linha(x0, y0, x1, y1, x2, y2):
    """Calcula a distância perpendicular de um ponto (x0,y0) até a linha formada por (x1,y1) e (x2,y2)"""
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num / den if den != 0 else 0.0

def main():
    print("Conectando ao CoppeliaSim via ZeroMQ API...")
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    robo = DifferentialRobot(sim, motor_esq_name='MOTOR_ESQUERDO', motor_dir_name='MOTOR_DIREITO')
    
    NOME_ROBO = '/Cuboid'
    NOME_ALVO = '/Alvo'
    DISTANCIA_TOLERANCIA = 0.25  
    
    # Reseta a simulação
    try:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_state_stopped:
            time.sleep(0.05)
    except:
        pass

    sim.setStepping(True)
    sim.startSimulation()
    client.step()
    sim.step()
    
    # Coleta coordenadas iniciais para criar a linha mestre de navegação (Linha Bug)
    x_ini, y_ini = robo.obter_posicao_objeto(NOME_ROBO)
    x_alvo, y_alvo = robo.obter_posicao_objeto(NOME_ALVO)
    
    # Variáveis de controle da máquina de estados do Bug-2
    modo_contorno = False
    lado_contorno = 1             # 1 = Contornar pela Esquerda, -1 = Direita
    distancia_ponto_impacto = float('inf')
    dist_segurança_parede = 0.38  # Distância ideal para passar em vãos estreitos
    
    print("=" * 60)
    print(" [SISTEMA BUG-2 REAL] Contorno Otimizado de Menor Distância")
    print("=" * 60)

    while True:
        client.step()
        sim.step()

        # Telemetria em tempo real
        x_atual, y_atual = robo.obter_posicao_objeto(NOME_ROBO)
        yaw_atual = robo.obter_orientacao_robo(NOME_ROBO)
        x_alvo, y_alvo = robo.obter_posicao_objeto(NOME_ALVO)
        
        distancia_alvo = math.hypot(x_alvo - x_atual, y_alvo - y_atual)
        
        if distancia_alvo < 0.01:
            continue

        if distancia_alvo <= DISTANCIA_TOLERANCIA:
            print(f"\n\n[SUCESSO] Alvo Alcançado! Distância final: {distancia_alvo:.2f}m")
            robo.parar()
            sim.stopSimulation()
            break 

        dist_sensores = robo.ler_distancias_separadas()

        # Rumo matemático ao alvo
        dx = x_alvo - x_atual
        dy = y_alvo - y_atual
        angulo_alvo = math.atan2(dy, dx)
        erro_rumo = math.atan2(math.sin(angulo_alvo - yaw_atual), math.cos(angulo_alvo - yaw_atual))

        # Detecção de Bloqueio frontal (Indica colisão com o L)
        bloqueio_frontal = (dist_sensores['SENSOR_MEIO'] < 0.55 or 
                            dist_sensores['SENSOR_DIAG_ESQUERDO'] < 0.45 or 
                            dist_sensores['SENSOR_DIAG_DIREITO'] < 0.45)

        # ─── TRANSIÇÃO DE ESTADOS (ALGORITMO BUG-2) ───
        
        if not modo_contorno:
            if bloqueio_frontal:
                # Transiciona para o modo de contorno
                modo_contorno = True
                distancia_ponto_impacto = distancia_alvo
                
                # Escolhe o lado de menor trabalho analisando as diagonais
                if dist_sensores['SENSOR_DIAG_DIREITO'] < dist_sensores['SENSOR_DIAG_ESQUERDO'] or dist_sensores['SENSOR_DIREITO'] < dist_sensores['SENSOR_ESQUERDO']:
                    lado_contorno = 1  # Vira para a esquerda, mureta fica na sua direita
                else:
                    lado_contorno = -1 # Vira para a direita, mureta fica na sua esquerda
                print(f"\n[BUG-2] Obstáculo detectado! Iniciando contorno tático. Lado: {lado_contorno}")
        
        else:
            # Critério de saída do contorno: Cruzar a linha imaginária inicial estando mais perto do alvo
            dist_linha = calcular_distancia_ponto_linha(x_atual, y_atual, x_ini, y_ini, x_alvo, y_alvo)
            
            # Se ele voltou para a linha mestre, está mais perto do alvo do que quando bateu, e a frente está livre
            if dist_linha < 0.15 and distancia_alvo < (distancia_ponto_impacto - 0.2) and not bloqueio_frontal:
                # Condição adicional: o bico do robô deve conseguir olhar para o alvo sem ver mureta perto
                if dist_sensores['SENSOR_MEIO'] > 0.8:
                    modo_contorno = False
                    print("\n[BUG-2] Linha mestre recuperada! Voltando a ir direto ao Alvo.")

        # ─── EXECUÇÃO DO MOVIMENTO POR ESTADO ───
        
        # REGRA SUPREMA DE SEGURANÇA: Se o alvo mudar bruscamente e ficar para trás, força o giro antes de andar
        if abs(erro_rumo) > math.radians(60) and dist_sensores['SENSOR_MEIO'] > 0.50:
            melhor_v = 0.0
            melhor_w = 1.3 if erro_rumo > 0 else -1.3
            sys.stdout.write(f"\r[ALINHANDO CHASSI] Corrigindo ângulo para o Alvo: {math.degrees(erro_rumo):.1f}°      ")
            
        # RÉ DE EMERGÊNCIA: Se colou de cara na parede, recua um pouco para obter espaço de giro
        elif dist_sensores['SENSOR_MEIO'] < 0.32:
            melhor_v = -0.08
            melhor_w = 0.0
            sys.stdout.write(f"\r[RÉ DE SEGURANÇA] Afastando chassi da quina interna do L...               ")

        # MOVIMENTAÇÃO DO MODO: IR DIRETO AO ALVO
        elif not modo_contorno:
            melhor_v = 0.20
            melhor_w = 2.5 * erro_rumo
            sys.stdout.write(f"\r[MODO: ALVO] Avançando em linha reta | Dist: {distancia_alvo:.2f}m             ")

        # MOVIMENTAÇÃO DO MODO: WALL-FOLLOWING (CONTORNAR PAREDE)
        else:
            melhor_v = 0.14  # Velocidade controlada para passar nos vãos estreitos com precisão
            melhor_w = 0.0
            
            # Identifica qual sensor deve ler a lateral da mureta
            sensor_lateral = dist_sensores['SENSOR_DIREITO'] if lado_contorno == 1 else dist_sensores['SENSOR_ESQUERDO']
            sensor_diag = dist_sensores['SENSOR_DIAG_DIREITO'] if lado_contorno == 1 else dist_sensores['SENSOR_DIAG_ESQUERDO']
            
            # Se bater de frente ou na quina interna do L, faz giro puro no eixo para limpar a frente
            if dist_sensores['SENSOR_MEIO'] < 0.45 or sensor_diag < 0.35:
                melhor_v = 0.0
                melhor_w = 0.8 * lado_contorno
                sys.stdout.write("\r[CONTORNO] Rotacionando bico para livrar quina interna do L...         ")
            else:
                if sensor_lateral != float('inf'):
                    # Malha fechada Proporcional para manter a distância exata de segurança da mureta lateral
                    erro_parede = sensor_lateral - dist_segurança_parede
                    melhor_w = 3.5 * erro_parede * lado_contorno
                    sys.stdout.write(f"\r[CONTORNO] Margeando vão estreito | Dist parede: {sensor_lateral:.2f}m    ")
                else:
                    # Se perder a parede de vista (chegou no fim do bloco), faz curva suave para abraçar a quina externa
                    melhor_w = -0.5 * lado_contorno
                    sys.stdout.write("\r[CONTORNO] Dobrando quina externa do obstáculo...                      ")

        # Saturação de segurança e envio aos motores
        melhor_w = max(min(melhor_w, 1.4), -1.4)
        robo.mover(melhor_v, melhor_w)
        sys.stdout.flush()

if __name__ == '__main__':
    main()