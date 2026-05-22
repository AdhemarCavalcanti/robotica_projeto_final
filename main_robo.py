import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from robot_controller import DifferentialRobot
from dwa_navigation import DWA_Planner

def main():
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    # Instancia as classes do seu projeto
    robo = DifferentialRobot(sim, motor_esq_name='MOTOR_ESQUERDO', motor_dir_name='MOTOR_DIREITO')
    planejador = DWA_Planner()
    
    # Nomes dos marcadores na Scene Hierarchy
    NOME_ROBO = '/Cuboid'
    NOME_ALVO = '/Alvo'
    DISTANCIA_TOLERANCIA = 0.25  # Raio de proximidade (25 centímetros)
    
    print("=" * 60)
    print(" [SISTEMA FINAL] Navegação Autónoma DWA com Parada Ativa")
    print("=" * 60)
    
    sim.startSimulation()
    
    try:
        while True:
            # Captura de telemetria via POO
            x_atual, y_atual = robo.obter_posicao_objeto(NOME_ROBO)
            yaw_atual = robo.obter_orientacao_robo(NOME_ROBO)
            x_alvo, y_alvo = robo.obter_posicao_objeto(NOME_ALVO)
            
            distancia_alvo = math.hypot(x_alvo - x_atual, y_alvo - y_atual)
            obstaculos_detectados = robo.ler_sensores_obstaculos()
            
            # verificação de chegada com paragem mecânica forçada
            if distancia_alvo <= DISTANCIA_TOLERANCIA:
                print(f"\n[SUCESSO] Alvo Alcançado com Êxito! Distância final: {distancia_alvo:.2f}m")
                
                # 🛑 CRÍTICO: Imobiliza os motores e quebra o loop imediatamente
                robo.parar()
                break 
                
            else:
                # O DWA calcula as velocidades ótimas de avanço e curva
                melhor_v, melhor_w = planejador.planejar(
                    x_atual, y_atual, yaw_atual,
                    x_alvo, y_alvo,
                    obstaculos_detectados
                )
                
                # Envia o comando para a estrutura diferencial
                robo.mover(melhor_v, melhor_w)
                
                # Exibe métricas de telemetria no terminal
                dx = x_alvo - x_atual
                dy = y_alvo - y_atual
                angulo_alvo = math.atan2(dy, dx)
                erro_rumo = math.atan2(math.sin(angulo_alvo - yaw_atual), math.cos(angulo_alvo - yaw_atual))
                
                print(f"Dist: {distancia_alvo:.2f}m | Erro Rumo: {math.degrees(erro_rumo):.1f}° | V: {melhor_v:.2f} W: {melhor_w:.2f}", end="\r")

            client.step()

    except KeyboardInterrupt:
        print("\nProcesso interrompido pelo terminal.")
    finally:
        # Garante o encerramento seguro tanto do script quanto do simulador 3D
        robo.parar()
        try:
            sim.stopSimulation()
        except:
            pass
        print("Simulação finalizada com sucesso.")

if __name__ == '__main__':
    main()