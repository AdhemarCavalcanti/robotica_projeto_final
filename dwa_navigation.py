import math

class DWA_Planner:
    def __init__(self):
        # Limites dinâmicos mais agressivos para contornar L's
        self.max_v = 0.20        # Aumentada para dar velocidade em retas
        self.min_v = -0.10       # Permitir ré suave para o robô sair do L se entrar
        self.max_w = 0.9         # Aumentada para dar giros rápidos
        
        self.v_passos = 5
        self.w_passos = 16       # Resolução angular
        
        # Pesos da Função de Custo REVISADOS para esse cenário
        self.peso_alvo = 3.2       # Atração ao Dummy
        self.peso_velocidade = 0.4 # Queremos que ele ande rápido
        self.peso_obstaculo = 5.5  # Peso ALTO para evitar quinas (era 4.0)
        
        self.raio_colisao = 0.35    # Distância de segurança do chassi para contornar as paredes em L

    def planejar(self, x_rob, y_rob, yaw_rob, x_alvo, y_alvo, obstaculos):
        melhor_v = 0.0
        melhor_w = 0.0
        melhor_custo = -float('inf')
        
        # Cria a janela dinâmica de velocidades de busca
        lista_v = [i * (self.max_v - self.min_v) / self.v_passos + self.min_v for i in range(self.v_passos + 1)]
        lista_w = [i * (2 * self.max_w) / self.w_passos - self.max_w for i in range(self.w_passos + 1)]
        
        for v in lista_v:
            for w in lista_w:
                dt = 0.6
                # Projeção cinemática de onde o robô estaria no futuro
                proximo_x = x_rob + v * math.cos(yaw_rob) * dt
                proximo_y = y_rob + v * math.sin(yaw_rob) * dt
                proximo_yaw = yaw_rob + w * dt
                
                min_dist_obstaculo = float('inf')
                # Varre os obstáculos lidos pelos feixes rosas nas quinas das paredes L
                for ob in obstaculos:
                    dist = math.hypot(proximo_x - ob[0], proximo_y - ob[1])
                    if dist < min_dist_obstaculo:
                        min_dist_obstaculo = dist
                
                # Se colidir ou invadir o raio de segurança, descarta a trajetória
                if min_dist_obstaculo < self.raio_colisao:
                    continue
                
                # Geometria do vetor até o Alvo Dummy
                dx = x_alvo - proximo_x
                dy = y_alvo - proximo_y
                angulo_alvo = math.atan2(dy, dx)
                
                # Diferença angular normalizada entre -PI e +PI
                erro_rumo = math.atan2(math.sin(angulo_alvo - proximo_yaw), math.cos(angulo_alvo - proximo_yaw))
                
                # Função de custo: Maximizar alinhamento e velocidade, minimizando proximidade à mureta L
                custo_alvo = math.pi - abs(erro_rumo)
                custo_vel = v
                custo_ob = min_dist_obstaculo if min_dist_obstaculo != float('inf') else 1.5
                
                custo_total = (self.peso_alvo * custo_alvo) + (self.peso_velocidade * custo_vel) + (self.peso_obstaculo * custo_ob)
                
                if custo_total > melhor_custo:
                    melhor_custo = custo_total
                    melhor_v = v
                    melhor_w = w
                    
        return melhor_v, melhor_w