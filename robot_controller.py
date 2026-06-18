import math

class DifferentialRobot:
    def __init__(self, sim, motor_esq_name='MOTOR_ESQUERDO', motor_dir_name='MOTOR_DIREITO'):
        self.sim = sim
        self.motor_esquerdo = sim.getObject(f'/{motor_esq_name}')
        self.motor_direito = sim.getObject(f'/{motor_dir_name}')
        
        self.nomes_ordem = [
            'SENSOR_ESQUERDO', 'SENSOR_DIAG_ESQUERDO', 
            'SENSOR_MEIO', 
            'SENSOR_DIAG_DIREITO', 'SENSOR_DIREITO'
        ]
        
        self.sensores = {}
        for nome in self.nomes_ordem:
            try:
                self.sensores[nome] = sim.getObject(f'/{nome}')
            except:
                print(f"[AVISO] Sensor /{nome} não foi localizado na Scene Hierarchy.")
        
        self.raio_roda = 0.035      
        self.distancia_eixos = 0.25   

    def obter_posicao_objeto(self, nome_objeto):
        try:
            handle = self.sim.getObject(nome_objeto)
            posicao = self.sim.getObjectPosition(handle, self.sim.handle_world)
            return posicao[0], posicao[1]
        except:
            return 0.0, 0.0

    def obter_orientacao_robo(self, nome_robo='/Cuboid'):
        try:
            handle = self.sim.getObject(nome_robo)
            orientacao = self.sim.getObjectOrientation(handle, self.sim.handle_world)
            yaw_corrigido = orientacao[2] + math.pi
            return math.atan2(math.sin(yaw_corrigido), math.cos(yaw_corrigido))
        except:
            return 0.0

    def ler_distancias_separadas(self):
        leituras = {}
        max_distancia_sensor = 1.2
        for nome in self.nomes_ordem:
            if nome in self.sensores:
                resultado, dist, _, _, _ = self.sim.readProximitySensor(self.sensores[nome])
                if resultado > 0 and dist < max_distancia_sensor:
                    leituras[nome] = dist
                else:
                    leituras[nome] = float('inf')
            else:
                leituras[nome] = float('inf')
        return leituras

    def mover(self, v_linear, w_angular, forçar_avanco=False):
        # Conversão cinemática inversa diferencial padrão
        vel_rod_esquerda = (v_linear - (w_angular * self.distancia_eixos / 2.0)) / self.raio_roda
        vel_rod_direita = (v_linear + (w_angular * self.distancia_eixos / 2.0)) / self.raio_roda
        
        # BLINDAGEM ANTI-RÉ: Se ativado, impede fisicamente qualquer rotação reversa nas rodas
        if forçar_avanco:
            if vel_rod_esquerda < 0: vel_rod_esquerda = 0.0
            if vel_rod_direita < 0: vel_rod_direita = 0.0
            
            # Força torque estrito de avanço na roda externa caso fiquem nulas
            if vel_rod_esquerda == 0.0 and vel_rod_direita == 0.0:
                if w_angular > 0: vel_rod_direita = 2.5
                else: vel_rod_esquerda = 2.5

        self.sim.setJointTargetVelocity(self.motor_esquerdo, vel_rod_esquerda)
        self.sim.setJointTargetVelocity(self.motor_direito, vel_rod_direita)

    def parar(self):
        try:
            self.sim.setJointTargetVelocity(self.motor_esquerdo, 0.0)
            self.sim.setJointTargetVelocity(self.motor_direito, 0.0)
        except:
            pass