import math
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import dwa_navigation as dw
import mapa_ocupacao as mo

SINAL_FWD = +1


class CoppeliaRobotNavigator:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.dwa = dw.DWAController()
        
        # Estrutura de chaves do contexto antigo
        self.motor_r = None
        self.motor_l = None
        self.robot_handle = None
        self.goal_handle = None
        self.sensors = []
        
        self.state = np.zeros(5, dtype=float)
        self.initial_z = 0.0
        self.initial_roll = 0.0
        self.initial_pitch = 0.0
        self.heading_offset = 0.0
        
        self.goal_xy = [0.0, 0.0]
        self.static_obstacles = np.empty((0, 2))
        
        # Variáveis estáticas de rota originais para o Matplotlib
        self.global_path = []
        self.key_points = []
        self.rx = []
        self.ry = []
        self.kx = []
        self.ky = []
        
        self.path_index = 0
        self.step_count = 0
        
        self.stuck_reference = None
        self.stuck_steps = 0
        self.recovery_countdown = 0

    def get_valid_object(self, path):
        try:
            return self.sim.getObject(path)
        except Exception as err:
            raise RuntimeError(f"Objeto ausente: {path}") from err

    def is_descendant(self, current_handle, target_parent):
        while current_handle != -1:
            if current_handle == target_parent:
                return True
            current_handle = self.sim.getObjectParent(current_handle)
        return False

    def transform_point_3d(self, matrix, point):
        return np.array([
            matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
            matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
            matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
        ], dtype=float)

    def generate_bounding_box_points(self, x0, x1, y0, y1, step=0.06):
        pts = []
        x = x0
        while x <= x1:
            pts.append([x, y0])
            pts.append([x, y1])
            x += step
        y = y0
        while y <= y1:
            pts.append([x0, y])
            pts.append([x1, y])
            y += step
        return pts

    def extract_filled_rectangle(self, matrix, x0, x1, y0, y1, step=0.04):
        pts = []
        x = x0
        while x <= x1:
            y = y0
            while y <= y1:
                w_pt = self.transform_point_3d(matrix, [x, y, 0.0])
                pts.append([w_pt[0], w_pt[1]])
                y += step
            x += step
        return pts

    def build_fallback_obstacles(self):
        points = []
        floor = self.get_valid_object("/Floor")
        f_pos = self.sim.getObjectPosition(floor, -1)
        
        f_x0 = self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_min_x)
        f_x1 = self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_max_x)
        f_y0 = self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_min_y)
        f_y1 = self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_max_y)

        points.extend(self.generate_bounding_box_points(
            f_pos[0] + f_x0, f_pos[0] + f_x1, f_pos[1] + f_y0, f_pos[1] + f_y1
        ))

        for obj in self.sim.getObjectsInTree(self.sim.handle_scene):
            if self.sim.getObjectType(obj) != self.sim.object_shape_type:
                continue

            alias = self.sim.getObjectAlias(obj, 0)
            if alias in {"Floor", "box", "Goal", "Target", "Alvo", "camera_grade_ocupacao"}:
                continue

            if obj == self.robot_handle or self.is_descendant(obj, self.robot_handle):
                continue

            matrix = self.sim.getObjectMatrix(obj, -1)
            x0 = self.sim.getObjectFloatParam(obj, self.sim.objfloatparam_objbbox_min_x)
            x1 = self.sim.getObjectFloatParam(obj, self.sim.objfloatparam_objbbox_max_x)
            y0 = self.sim.getObjectFloatParam(obj, self.sim.objfloatparam_objbbox_min_y)
            y1 = self.sim.getObjectFloatParam(obj, self.sim.objfloatparam_objbbox_max_y)

            if (x1 - x0) > 4.5 or (y1 - y0) > 4.5:
                continue

            points.extend(self.extract_filled_rectangle(matrix, x0, x1, y0, y1))

        unique_pts = {(round(x, 2), round(y, 2)): [x, y] for x, y in points}
        return np.array(list(unique_pts.values()), dtype=float)

    def update_robot_state(self, linear_v=0.0, angular_w=0.0):
        pos = self.sim.getObjectPosition(self.robot_handle, -1)
        orientation = self.sim.getObjectOrientation(self.robot_handle, -1)
        
        flip = 0.0 if SINAL_FWD >= 0 else math.pi
        heading = math.atan2(math.sin(orientation[2]), math.cos(orientation[2]))
        normalized_th = math.atan2(
            math.sin(heading + self.heading_offset + flip),
            math.cos(heading + self.heading_offset + flip)
        )
        self.state = np.array([pos[0], pos[1], normalized_th, linear_v, angular_w], dtype=float)

    def compute_sensor_alignment(self):
        front_sensor = self.sensors[0]
        matrix = self.sim.getObjectMatrix(front_sensor, -1)
        yaw_f = math.atan2(matrix[6], matrix[2])
        yaw_m = self.sim.getObjectOrientation(self.robot_handle, -1)[2]
        
        diff = yaw_f - yaw_m
        self.heading_offset = math.atan2(math.sin(diff), math.cos(diff))
        print("Offset (graus):", round(math.degrees(self.heading_offset), 1))

    def read_proximity_sensors(self):
        detected_points = []
        for sensor in self.sensors:
            triggered, dist, local_pt, _, _ = self.sim.readProximitySensor(sensor)
            if triggered > 0:
                pt_vector = np.array(local_pt, dtype=float)
                if np.linalg.norm(pt_vector) <= 0.0 and dist > 0.0:
                    pt_vector = np.array([dist, 0.0, 0.0], dtype=float)
                
                matrix = self.sim.getObjectMatrix(sensor, -1)
                world_pt = self.transform_point_3d(matrix, pt_vector)
                detected_points.append([world_pt[0], world_pt[1]])
        return np.array(detected_points, dtype=float)

    def aggregate_local_obstacles(self):
        local_cloud = []
        if self.static_obstacles is not None and len(self.static_obstacles) > 0:
            distances = np.hypot(self.static_obstacles[:, 0] - self.state[0], self.static_obstacles[:, 1] - self.state[1])
            local_cloud.extend(self.static_obstacles[distances <= 1.4].tolist())

        live_scans = self.read_proximity_sensors()
        if len(live_scans) > 0:
            local_cloud.extend(live_scans.tolist())
            
        return np.array(local_cloud, dtype=float)

    def is_colliding(self, simulated_pose):
        if self.static_obstacles is None or len(self.static_obstacles) == 0:
            return False
        distances = np.hypot(self.static_obstacles[:, 0] - simulated_pose[0], self.static_obstacles[:, 1] - simulated_pose[1])
        return float(np.min(distances)) <= self.dwa.collision_radius

    def execute_global_planner(self):
        sx, sy = float(self.state[0]), float(self.state[1])
        gx, gy = float(self.goal_xy[0]), float(self.goal_xy[1])
        
        for radius in (0.22, 0.18, 0.14):
            planner = dw.AStarPlanner(
                self.static_obstacles[:, 0].tolist(), 
                self.static_obstacles[:, 1].tolist(), 
                resolution=0.08, rr=radius
            )
            self.rx, self.ry, self.kx, self.ky = planner.planning(sx, sy, gx, gy)
            if not planner.last_plan_failed:
                print(f"A* OK com rr={radius:.2f}")
                break
            print(f"A* falhou com rr={radius:.2f}")
        else:
            print("A* falhou geral, linha reta.")
            self.rx, self.ry, self.kx, self.ky = [sx, gx], [sy, gy], [sx, gx], [sy, gy]

        self.global_path = list(zip(self.rx, self.ry))
        self.key_points = list(zip(self.kx, self.ky))
        self.path_index = 0

    def extract_dynamic_target(self):
        path = self.global_path if self.global_path else [self.goal_xy]
        while self.path_index < len(path) - 1:
            current_target = path[self.path_index]
            if math.hypot(current_target[0] - self.state[0], current_target[1] - self.state[1]) >= 0.40:
                break
            self.path_index += 1
        
        idx = min(self.path_index + 2, len(path) - 1)
        return path[idx]

    def drive_actuators(self, control_inputs):
        v, w = float(control_inputs[0]), float(control_inputs[1])
        dt = self.dwa.dt
        wheel_radius, track_width = 0.0375, 0.15

        next_state = self.state.copy()
        next_state[2] = math.atan2(math.sin(next_state[2] + w * dt), math.cos(next_state[2] + w * dt))
        next_state[0] += v * math.cos(next_state[2]) * dt
        next_state[1] += v * math.sin(next_state[2]) * dt
        next_state[3], next_state[4] = v, w

        if self.is_colliding(next_state):
            next_state = self.state.copy()
            next_state[2] = math.atan2(math.sin(next_state[2] + w * dt), math.cos(next_state[2] + w * dt))
            next_state[3], next_state[4] = 0.0, w
            v = 0.0

        v_inv, w_inv = -v, -w
        w_right = (2.0 * v_inv + w_inv * track_width) / (2.0 * wheel_radius)
        w_left = (2.0 * v_inv - w_inv * track_width) / (2.0 * wheel_radius)
        
        w_right = max(min(w_right, 20.0), -20.0)
        w_left = max(min(w_left, 20.0), -20.0)

        self.sim.setJointTargetVelocity(self.motor_r, w_right)
        self.sim.setJointTargetVelocity(self.motor_l, w_left)
        self.sim.setObjectPosition(self.robot_handle, -1, [next_state[0], next_state[1], self.initial_z])

        flip = 0.0 if SINAL_FWD >= 0 else math.pi
        computed_yaw = math.atan2(math.sin(next_state[2] - self.heading_offset - flip), 
                                  math.cos(next_state[2] - self.heading_offset - flip))
        
        self.sim.setObjectOrientation(self.robot_handle, -1, [self.initial_roll, self.initial_pitch, computed_yaw])
        
        try:
            self.sim.resetDynamicObject(self.robot_handle)
        except Exception:
            pass

        self.sim.step()
        return next_state

    def load_static_map_data(self):
        exclusions = [
            (float(self.state[0]), float(self.state[1]), self.dwa.robot_radius + 0.15),
            (float(self.goal_xy[0]), float(self.goal_xy[1]), self.dwa.robot_radius + 0.10),
        ]
        try:
            floor_handle = self.get_valid_object("/Floor")
            pts, grid, info = mo.construir_obstaculos_por_visao(self.sim, floor_handle, excluir=exclusions)
            if len(pts) >= 4:
                return pts
        except Exception:
            print("Usando fallback Bounding Box...")
        return self.build_fallback_obstacles()

    def render_static_chart(self):
        if self.static_obstacles is None or len(self.static_obstacles) == 0:
            print("[Aviso] Não foi possível abrir o mapa pois os obstáculos não foram calculados.")
            return

        floor = self.get_valid_object("/Floor")
        f_pos = self.sim.getObjectPosition(floor, -1)
        x0 = f_pos[0] + self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_min_x)
        x1 = f_pos[0] + self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_max_x)
        y0 = f_pos[1] + self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_min_y)
        y1 = f_pos[1] + self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_max_y)

        print("\n---> ABRINDO O MAPA INICIAL <---")
        print("Analise as rotas geradas pelo A*. FECHE A JANELA DO GRÁFICO para iniciar a simulação no CoppeliaSim.\n")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("Planejamento Estático - A*", fontsize=12, fontweight='bold')

        obs_arr = np.array(self.static_obstacles)
        path_arr = np.array(self.global_path)
        kp_arr = np.array(self.key_points)

        for ax, t in zip([ax1, ax2], ["Rota Completa", "Pontos-Chave"]):
            ax.set_title(t)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)
            ax.axis("equal")
            ax.set_xlim(x0 - 0.1, x1 + 0.1)
            ax.set_ylim(y0 - 0.1, y1 + 0.1)
            if len(obs_arr) > 0:
                ax.scatter(obs_arr[:, 0], obs_arr[:, 1], s=4, c="black", marker="s")
            ax.plot(self.state[0], self.state[1], "go", markersize=9)
            ax.plot(self.goal_xy[0], self.goal_xy[1], "ro", markersize=9)

        if len(path_arr) > 0:
            ax1.plot(path_arr[:, 0], path_arr[:, 1], "b-", linewidth=2)
        if len(kp_arr) > 0:
            ax2.plot(kp_arr[:, 0], kp_arr[:, 1], "y-x", markersize=8, linewidth=1.5)

        plt.tight_layout()
        plt.show()

    def bootstrap_system(self):
        self.motor_r = self.get_valid_object("/MOTOR_DIREITO")
        self.motor_l = self.get_valid_object("/MOTOR_ESQUERDO")
        self.robot_handle = self.sim.getObjectParent(self.motor_r)
        self.goal_handle = self.get_valid_object("/Alvo")

        self.sensors = [
            self.get_valid_object("/SENSOR_MEIO"),
            self.get_valid_object("/SENSOR_DIAG_DIREITO"),
            self.get_valid_object("/SENSOR_DIAG_ESQUERDO"),
            self.get_valid_object("/SENSOR_DIREITO"),
            self.get_valid_object("/SENSOR_ESQUERDO"),
        ]

        r_pos = self.sim.getObjectPosition(self.robot_handle, -1)
        r_ori = self.sim.getObjectOrientation(self.robot_handle, -1)
        self.initial_z, self.initial_roll, self.initial_pitch = r_pos[2], r_ori[0], r_ori[1]

        self.compute_sensor_alignment()
        self.update_robot_state()

        g_pos = self.sim.getObjectPosition(self.goal_handle, -1)
        self.goal_xy = [g_pos[0], g_pos[1]]
        self.static_obstacles = self.load_static_map_data()
        self.step_count = 0

        self.execute_global_planner()

    def calculate_escape_velocity(self, target_point):
        ang = math.atan2(target_point[1] - self.state[1], target_point[0] - self.state[0])
        turn = math.atan2(math.sin(ang - self.state[2]), math.cos(ang - self.state[2]))
        
        # Correção do Erro: Substituído o limite interno inexistente por valores manuais do seu código original
        w = max(min(1.2 * turn, 4.0), -4.0) 
        
        if self.recovery_countdown > 22:
            return [0.07, 0.3 * w]
        if abs(turn) > 0.25:
            return [0.0, -0.8 if turn >= 0.0 else 0.8]
        return [-0.10, 0.5 * w]

    def runtime_loop(self):
        if self.goal_handle is not None:
            g_pos = self.sim.getObjectPosition(self.goal_handle, -1)
            self.goal_xy = [g_pos[0], g_pos[1]]

        local_obs = self.aggregate_local_obstacles()
        current_waypoint = self.extract_dynamic_target()

        if self.stuck_reference is None or math.hypot(self.state[0] - self.stuck_reference[0], self.state[1] - self.stuck_reference[1]) > 0.05:
            self.stuck_reference = (float(self.state[0]), float(self.state[1]))
            self.stuck_steps = 0
        else:
            self.stuck_steps += 1

        if self.recovery_countdown > 0:
            control_actions = self.calculate_escape_velocity(current_waypoint)
            self.recovery_countdown -= 1
        else:
            control_actions, _ = self.dwa.plan(self.state[0:3], self.state[3], self.state[4], current_waypoint, local_obs)
            if self.stuck_steps >= 30:
                print("Preso -> replanejando...")
                self.execute_global_planner()
                self.recovery_countdown = 35
                self.stuck_steps = 0

        self.state = self.drive_actuators(control_actions)
        self.step_count += 1

        if math.hypot(self.state[0] - self.goal_xy[0], self.state[1] - self.goal_xy[1]) <= 0.20:
            print("ALVO ATINGIDO!")
            return True
        return False


if __name__ == "__main__":
    print("Conectando ao CoppeliaSim...")
    navigator = CoppeliaRobotNavigator()
    
    navigator.bootstrap_system()
    navigator.render_static_chart()

    print("Iniciando a simulação física...")
    navigator.sim.setStepping(True)
    navigator.sim.startSimulation()

    try:
        while not navigator.runtime_loop():
            pass
    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")
    finally:
        if navigator.motor_r and navigator.motor_l:
            navigator.sim.setJointTargetVelocity(navigator.motor_r, 0.0)
            navigator.sim.setJointTargetVelocity(navigator.motor_l, 0.0)
        
        navigator.sim.stopSimulation()
        print("Simulação encerrada e finalizada.")