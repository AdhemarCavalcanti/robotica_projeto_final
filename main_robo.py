import math
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from robot_controller import DifferentialRobot
from dwa_navigation import DWA_Planner

def main():
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    
    # Instancia os componentes do sistema
    robo = DifferentialRobot(sim, motor_esq_name='MOTOR_ESQUERDO', motor_dir_name='MOTOR_DIREITO')
    planejador = DWA_Planner()
    
    NOME_ROBO = '/Cuboid'
    NOME_ALVO = '/Alvo'
    DISTANCIA_TOLERANCIA = 0.25
    
    print("=" * 60)
    print(" [SISTEMA INTEGADO POO] Executando Navegação Estabilizada")
    print("=" * 60)
    
    sim.startSimulation()
    
    try:
        while True:
            # Coleta de dados cinemáticos via métodos da classe
            x_atual, y_atual = robo.obter_posicao_objeto(NOME_ROBO)
            yaw_atual = robo.obter_orientacao_robo(NOME_ROBO) # Retorna o ângulo já alinhado
            x_alvo, y_alvo = robo.obter_posicao_objeto(NOME_ALVO)
            
            distancia_alvo = math.hypot(x_alvo - x_atual, y_alvo - y_atual)
            obstaculos_detectados = robo.ler_sensores_obstaculos()
            
            if distancia_alvo <= DISTANCIA_TOLERANCIA:
                print(f"\r[SUCESSO] Alvo Alcançado com Êxito!                        ", end="")
                robo.parar()
            else:
                # O DWA recebe o yaw_atual estável e calcula as velocidades ótimas
                melhor_v, melhor_w = planejador.planejar(
                    x_atual, y_atual, yaw_atual,
                    x_alvo, y_alvo,
                    obstaculos_detectados
                )
                
                # Envia o comando direto para a estrutura diferencial
                robo.mover(melhor_v, melhor_w)
                
                # Exibe métricas de telemetria em tempo real no console
                dx = x_alvo - x_atual
                dy = y_alvo - y_atual
                angulo_alvo = math.atan2(dy, dx)
                erro_rumo = math.atan2(math.sin(angulo_alvo - yaw_atual), math.cos(angulo_alvo - yaw_atual))
                
                print(f"Dist: {distancia_alvo:.2f}m | Erro Rumo: {math.degrees(erro_rumo):.1f}° | Obstáculos: {len(obstaculos_detectados)} | V: {melhor_v:.2f} W: {melhor_w:.2f}", end="\r")

            client.step()

    except KeyboardInterrupt:
        print("\nProcesso interrompido pelo terminal.")
    finally:
        robo.parar()
        try:
            sim.stopSimulation()
        except:
            pass
        print("Simulação finalizada.")

if __name__ == '__main__':
    main()