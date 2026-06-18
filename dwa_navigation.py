import math

class DWA_Planner:
    def __init__(self):
        self.max_v = 0.20        
        self.min_v = 0.0         # PROIBIDA A RÉ: Mínimo é zero absoluto
        self.max_w = 1.4         
        
        self.v_passos = 5
        self.w_passos = 24       # Alta resolução para encontrar frestas nas quinas
        
        # Pesos da Função de Custo
        self.peso_alvo = 3.5       
        self.peso_velocidade = 0.5 
        self.peso_obstaculo = 6.0  
        
        # Raio de colisão baseado no chassi do robô
        self.raio_colisao = 0.14   

    def planejar(self, x_rob, y_rob, yaw_rob, x_alvo, y_alvo, obstaculos):
        melhor_v = 0.0
        melhor_w = 0.0
        melhor_custo = -float('inf')
        
        # Amostragem focada estritamente em progressão frontal
        lista_v = [0.0]
        passo_v = self.max_v / self.v_passos
        for i in range(1, self.v_passos + 1):
            lista_v.append(max(0.05, i * passo_v)) 
            
        lista_w = [i * (2 * self.max_w) / self.w_passos - self.max_w for i in range(self.w_passos + 1)]
        
        for v in lista_v:
            for w in lista_w:
                dt = 0.4  
                
                # Projeção cinemática descritiva
                proximo_x = x_rob + v * math.cos(yaw_rob) * dt
                proximo_y = y_rob + v * math.sin(yaw_rob) * dt
                proximo_yaw = yaw_rob + w * dt
                
                min_dist_obstaculo = float('inf')
                for ob in obstaculos:
                    dist = math.hypot(proximo_x - ob[0], proximo_y - ob[1])
                    if dist < min_dist_obstaculo:
                        min_dist_obstaculo = dist
                
                # Rejeita velocidades que invadam o raio físico do robô
                if min_dist_obstaculo < self.raio_colisao:
                    continue
                
                # Avaliação de rumo ao alvo
                dx = x_alvo - proximo_x
                dy = y_alvo - proximo_y
                angulo_alvo = math.atan2(dy, dx)
                erro_rumo = math.atan2(math.sin(angulo_alvo - proximo_yaw), math.cos(angulo_alvo - proximo_yaw))
                
                custo_alvo = math.pi - abs(erro_rumo)
                custo_vel = v
                custo_ob = min_dist_obstaculo if min_dist_obstaculo != float('inf') else 1.5
                
                custo_total = (self.peso_alvo * custo_alvo) + (self.peso_velocidade * custo_vel) + (self.peso_obstaculo * custo_ob)
                
                if custo_total > melhor_custo:
                    melhor_custo = custo_total
                    melhor_v = v
                    melhor_w = w
                    
        return melhor_v, melhor_w