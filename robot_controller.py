import math

class DifferentialRobot:
    def __init__(self, sim, motor_esq_name='MOTOR_ESQUERDO', motor_dir_name='MOTOR_DIREITO'):
        self.sim = sim
        self.motor_esquerdo = sim.getObject(f'/{motor_esq_name}')
        self.motor_direito = sim.getObject(f'/{motor_dir_name}')
        
        nomes_sensores = [
            'SENSOR_ESQUERDO', 'SENSOR_DIAG_ESQUERDO', 
            'SENSOR_MEIO', 
            'SENSOR_DIAG_DIREITO', 'SENSOR_DIREITO'
        ]
        
        self.sensores = []
        for nome in nomes_sensores:
            try:
                handle = sim.getObject(f'/{nome}')
                self.sensores.append(handle)
            except:
                print(f"[AVISO] Sensor /{nome} não mapeado.")
        
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
        """
        Retorna o ângulo Yaw estabilizado e rotacionado em 180 graus
        para alinhar o referencial do Cuboid com a frente dos sensores rosas.
        """
        try:
            handle = self.sim.getObject(nome_robo)
            orientacao = self.sim.getObjectOrientation(handle, self.sim.handle_world)
            
            # Ajuste de 180 graus (math.pi) para corrigir o erro de rumo inicial
            yaw_corrigido = orientacao[2] + math.pi
            
            # Normaliza o ângulo resultante entre -PI e +PI
            return math.atan2(math.sin(yaw_corrigido), math.cos(yaw_corrigido))
        except:
            return 0.0

    def ler_sensores_obstaculos(self):
        pontos_obstaculos = []
        max_distancia_sensor = 1.2 
        
        for sensor in self.sensores:
            resultado, dist, ponto, _, _ = self.sim.readProximitySensor(sensor)
            if resultado > 0 and dist < max_distancia_sensor:
                matriz = self.sim.getObjectMatrix(sensor, self.sim.handle_world)
                ponto_global = self.sim.multiplyVector(matriz, ponto)
                pontos_obstaculos.append([ponto_global[0], ponto_global[1]])
        return pontos_obstaculos

    def mover(self, v_linear, w_angular):
        vel_rod_esquerda = (v_linear - (w_angular * self.distancia_eixos / 2.0)) / self.raio_roda
        vel_rod_direita = (v_linear + (w_angular * self.distancia_eixos / 2.0)) / self.raio_roda
        self.sim.setJointTargetVelocity(self.motor_esquerdo, vel_rod_esquerda)
        self.sim.setJointTargetVelocity(self.motor_direito, vel_rod_direita)

    def parar(self):
        try:
            self.sim.setJointTargetVelocity(self.motor_esquerdo, 0.0)
            self.sim.setJointTargetVelocity(self.motor_direito, 0.0)
        except:
            pass