import math
import numpy as np


def normalize_angle(angle):
    """Garante que o ângulo permaneça no intervalo [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# ==========================================================================
#  CONTROLADOR LOCAL - DWA (DYNAMIC WINDOW APPROACH)
# ==========================================================================
class DWAController:
    def __init__(self):
        # Limites dinâmicos de velocidade
        self.max_v = 0.20          
        self.min_v = 0.0
        self.max_yaw_rate = 0.70  # Renomeado max_w para expor o atributo exigido no robô
        self.max_acceleration = 0.4 
        self.max_steer_acceleration = 1.5          
        
        # Resoluções e tempos de amostragem
        self.v_resolution = 0.02
        self.w_resolution = 0.08
        self.dt = 0.1
        self.prediction_time = 2.5

        # Dimensões de segurança do robô
        self.robot_radius = 0.24
        self.collision_radius = 0.16
        self.safety_margin = 0.06

        # Pesos das funções de custo
        self.weight_goal = 0.40         
        self.weight_speed = 1.0
        self.weight_obstacle = 0.45
        self.weight_distance = 2.2

    def calculate_dynamic_window(self, current_v, current_w):
        """Calcula a janela dinâmica com base nos limites físicos e aceleração."""
        absolute_limits = [self.min_v, self.max_v, -self.max_yaw_rate, self.max_yaw_rate]
        acceleration_limits = [
            current_v - self.max_acceleration * self.dt,
            current_v + self.max_acceleration * self.dt,
            current_w - self.max_steer_acceleration * self.dt,
            current_w + self.max_steer_acceleration * self.dt,
        ]
        return [
            max(absolute_limits[0], acceleration_limits[0]),
            min(absolute_limits[1], acceleration_limits[1]),
            max(absolute_limits[2], acceleration_limits[2]),
            min(absolute_limits[3], acceleration_limits[3]),
        ]

    def process_motion_step(self, pose, v, w):
        """Aplica o modelo cinemático diferencial para predizer o próximo estado."""
        next_pose = np.array(pose, dtype=float)
        next_pose[2] = normalize_angle(next_pose[2] + w * self.dt)
        next_pose[0] += v * math.cos(next_pose[2]) * self.dt
        next_pose[1] += v * math.sin(next_pose[2]) * self.dt
        return next_pose

    def generate_predicted_trajectory(self, initial_pose, v, w):
        """Gera uma matriz contendo toda a trajetória simulada adiante."""
        current_pose = np.array(initial_pose, dtype=float)
        trajectory = [current_pose.copy()]
        
        time_steps = np.arange(0.0, self.prediction_time + self.dt, self.dt)
        for _ in time_steps:
            current_pose = self.process_motion_step(current_pose, v, w)
            trajectory.append(current_pose.copy())
            
        return np.array(trajectory)

    def evaluate_goal_cost(self, trajectory, goal_xy):
        """Calcula o custo de alinhamento angular com o objetivo final."""
        last_pose = trajectory[-1]
        angle_to_goal = math.atan2(goal_xy[1] - last_pose[1], goal_xy[0] - last_pose[0])
        return abs(normalize_angle(angle_to_goal - last_pose[2]))

    def evaluate_obstacle_cost(self, trajectory, obstacles, v):
        """Mapeia a proximidade de obstáculos retornando infinito caso haja colisão."""
        if obstacles is None or len(obstacles) == 0:
            return 0.0

        # Vetorização do cálculo de distância euclidiana para todos os obstáculos
        diff_x = trajectory[:, 0:1] - obstacles[:, 0]
        diff_y = trajectory[:, 1:2] - obstacles[:, 1]
        distances = np.hypot(diff_x, diff_y)
        min_distance = float(np.min(distances))

        braking_distance = (v * v) / (2.0 * self.max_acceleration) if self.max_acceleration > 0 else 0.0
        clearance_zone = self.robot_radius + self.safety_margin + 0.5 * braking_distance

        if min_distance <= self.collision_radius:
            return float("inf")

        cost = 1.0 / (min_distance - self.collision_radius)
        if min_distance < clearance_zone:
            cost += 8.0 * (clearance_zone - min_distance) / clearance_zone

        return cost

    def plan(self, pose, current_v, current_w, goal_xy, obstacles):
        """Varre o espaço de comandos admissíveis selecionando a melhor velocidade linear e angular."""
        dw = self.calculate_dynamic_window(current_v, current_w)
        best_control = [0.0, 0.0]
        best_trajectory = self.generate_predicted_trajectory(pose, 0.0, 0.0)
        minimum_cost = float("inf")
        initial_distance_to_goal = math.hypot(goal_xy[0] - pose[0], goal_xy[1] - pose[1])

        v_candidates = np.arange(dw[0], dw[1] + self.v_resolution, self.v_resolution)
        w_candidates = np.arange(dw[2], dw[3] + self.w_resolution, self.w_resolution)

        for cv in v_candidates:
            if cv > dw[1]:
                continue
            for cw in w_candidates:
                if cw > dw[3]:
                    continue

                trajectory = self.generate_predicted_trajectory(pose, cv, cw)
                obstacle_cost = self.evaluate_obstacle_cost(trajectory, obstacles, cv)
                
                if math.isinf(obstacle_cost):
                    continue

                goal_cost = self.weight_goal * self.evaluate_goal_cost(trajectory, goal_xy)
                speed_cost = self.weight_speed * (self.max_v - cv)
                
                final_distance_to_goal = math.hypot(goal_xy[0] - trajectory[-1, 0], goal_xy[1] - trajectory[-1, 1])
                distance_cost = self.weight_distance * final_distance_to_goal
                
                progress_reward = 2.0 * max(0.0, initial_distance_to_goal - final_distance_to_goal)

                total_cost = goal_cost + speed_cost + (self.weight_obstacle * obstacle_cost) + distance_cost - progress_reward

                if total_cost < minimum_cost:
                    minimum_cost = total_cost
                    best_control = [float(cv), float(cw)]
                    best_trajectory = trajectory

        # Estratégia de Fallback: Se todas as trajetórias colidirem, o robô rotaciona no próprio eixo em direção ao alvo
        if math.isinf(minimum_cost):
            angle_to_goal = math.atan2(goal_xy[1] - pose[1], goal_xy[0] - pose[0])
            yaw_error = normalize_angle(angle_to_goal - pose[2])
            best_control = [0.08, 0.8 if yaw_error >= 0.0 else -0.8]
            best_trajectory = self.generate_predicted_trajectory(pose, best_control[0], best_control[1])

        return best_control, best_trajectory


# ==========================================================================
#  PLANEJADOR GLOBAL - A* (ALGORITMO DE TRAJETÓRIA OTIMIZADA COM SMOOTHING)
# ==========================================================================
# ==========================================================================
#  PLANEJADOR GLOBAL - A* (ALGORITMO DE TRAJETÓRIA OTIMIZADA COM SMOOTHING)
# ==========================================================================
class AStarPlanner:
    class GridNode:
        def __init__(self, x_idx, y_idx, cost, parent_id):
            self.x = x_idx
            self.y = y_idx
            self.cost = cost
            self.parent_id = parent_id

    # CORREÇÃO: Mantido o parâmetro como 'rr' para compatibilidade com o main_robo.py
    def __init__(self, ox, oy, resolution=0.1, rr=0.28):
        self.resolution = resolution
        self.robot_radius = rr  # Mapeia internamente para a lógica do código
        
        self.min_x = math.floor(min(ox))
        self.min_y = math.floor(min(oy))
        self.max_x = math.ceil(max(ox))
        self.max_y = math.ceil(max(oy))
        
        self.width_x = round((self.max_x - self.min_x) / self.resolution)
        self.width_y = round((self.max_y - self.min_y) / self.resolution)
        
        # Vetores direcionais de deslocamento discreto [dx, dy, custo_da_ação]
        self.motion_directions = [
            [1, 0, 1.0], [0, 1, 1.0], [-1, 0, 1.0], [0, -1, 1.0],
            [1, 1, math.sqrt(2)], [1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)], [-1, -1, math.sqrt(2)],
        ]
        
        self.obstacle_map = [[False for _ in range(self.width_y)] for _ in range(self.width_x)]
        self.inflation_cells = math.ceil(self.robot_radius / self.resolution)
        self.last_plan_failed = False

        # Constrói a matriz de ocupação inflando as células baseada no raio do robô
        for iox, ioy in zip(ox, oy):
            center_x = self.coordinate_to_index(iox, self.min_x)
            center_y = self.coordinate_to_index(ioy, self.min_y)
            
            for ix in range(center_x - self.inflation_cells, center_x + self.inflation_cells + 1):
                for iy in range(center_y - self.inflation_cells, center_y + self.inflation_cells + 1):
                    if ix < 0 or iy < 0 or ix >= self.width_x or iy >= self.width_y:
                        continue
                    
                    pos_x = self.index_to_coordinate(ix, self.min_x)
                    pos_y = self.index_to_coordinate(iy, self.min_y)
                    
                    if math.hypot(iox - pos_x, ioy - pos_y) <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True

    def planning(self, start_x, start_y, goal_x, goal_y):
        """Executa a busca A* com heurística dinâmica inflada."""
        start_node = self.GridNode(self.coordinate_to_index(start_x, self.min_x), 
                                   self.coordinate_to_index(start_y, self.min_y), 0.0, -1)
        goal_node = self.GridNode(self.coordinate_to_index(goal_x, self.min_x), 
                                  self.coordinate_to_index(goal_y, self.min_y), 0.0, -1)
        
        self.last_plan_failed = False
        clear_radius = self.inflation_cells + 1
        self.force_clear_cells(start_node.x, start_node.y, clear_radius)
        self.force_clear_cells(goal_node.x, goal_node.y, clear_radius)

        total_distance = max(math.hypot(goal_node.x - start_node.x, goal_node.y - start_node.y), 1e-6)
        open_set = {self.calculate_grid_id(start_node): start_node}
        closed_set = {}

        while open_set:
            current_id = min(
                open_set, 
                key=lambda k: open_set[k].cost + (1.0 + math.hypot(goal_node.x - open_set[k].x, goal_node.y - open_set[k].y) / total_distance) * math.hypot(goal_node.x - open_set[k].x, goal_node.y - open_set[k].y)
            )
            current_node = open_set[current_id]

            if current_node.x == goal_node.x and current_node.y == goal_node.y:
                goal_node.parent_id = current_node.parent_id
                goal_node.cost = current_node.cost
                rx, ry = self.reconstruct_path_sequence(goal_node, closed_set)
                return self.apply_post_processing(rx, ry)

            del open_set[current_id]
            closed_set[current_id] = current_node

            for dx, dy, movement_cost in self.motion_directions:
                neighbor = self.GridNode(current_node.x + dx, current_node.y + dy, 
                                         current_node.cost + movement_cost, current_id)
                neighbor_id = self.calculate_grid_id(neighbor)

                if not self.is_valid_node(neighbor) or neighbor_id in closed_set:
                    continue

                if neighbor_id not in open_set or open_set[neighbor_id].cost > neighbor.cost:
                    open_set[neighbor_id] = neighbor

        self.last_plan_failed = True
        return self.apply_post_processing([start_x, goal_x], [start_y, goal_y])

    def apply_post_processing(self, raw_x, raw_y):
        """Aplica simplificações geométricas e suavização por curvas de Bezier."""
        path_coordinates = list(zip(raw_x, raw_y))
        if len(path_coordinates) <= 2:
            return raw_x, raw_y, [pt[0] for pt in path_coordinates], [pt[1] for pt in path_coordinates]

        key_points = self._remove_collinear_nodes(path_coordinates)
        key_points = self._compress_by_line_of_sight(key_points)
        smoothed_path = self._generate_bezier_curves(key_points)

        return (
            [pt[0] for pt in smoothed_path], [pt[1] for pt in smoothed_path],
            [pt[0] for pt in key_points], [pt[1] for pt in key_points]
        )

    @staticmethod
    def _remove_collinear_nodes(path, tolerance=1e-6):
        if len(path) <= 2:
            return list(path)
        filtered = [path[0]]
        for i in range(1, len(path) - 1):
            ax, ay = filtered[-1]
            bx, by = path[i]
            cx, cy = path[i + 1]
            if abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) > tolerance:
                filtered.append(path[i])
        filtered.append(path[-1])
        return filtered

    def _compress_by_line_of_sight(self, nodes):
        if len(nodes) <= 2:
            return list(nodes)
        compressed = [nodes[0]]
        idx = 1
        while idx < len(nodes) - 1:
            if self.is_segment_clear(compressed[-1], nodes[idx + 1]):
                idx += 1
            else:
                compressed.append(nodes[idx])
                idx += 1
        compressed.append(nodes[-1])
        return compressed

    def is_segment_clear(self, point_a, point_b):
        xa, ya = point_a
        xb, yb = point_b
        segment_len = math.hypot(xb - xa, yb - ya)
        sampling_steps = max(2, int(segment_len / (self.resolution * 0.5)) + 1)

        for step in range(sampling_steps + 1):
            ratio = step / sampling_steps
            check_x = self.coordinate_to_index(xa + (xb - xa) * ratio, self.min_x)
            check_y = self.coordinate_to_index(ya + (yb - ya) * ratio, self.min_y)
            
            if check_x < 0 or check_y < 0 or check_x >= self.width_x or check_y >= self.width_y:
                return False
            if self.obstacle_map[check_x][check_y]:
                return False
        return True

    @staticmethod
    def _generate_bezier_curves(control_points, density_samples=12, interpolation_ratio=0.35):
        if len(control_points) <= 2:
            return list(control_points)
        
        vectors = [np.array(pt, dtype=float) for pt in control_points]
        interpolated_path = [tuple(vectors[0])]

        for i in range(1, len(vectors) - 1):
            prev_v, current_v, next_v = vectors[i - 1], vectors[i], vectors[i + 1]
            p0 = current_v + interpolation_ratio * (prev_v - current_v)
            p2 = current_v + interpolation_ratio * (next_v - current_v)

            interpolated_path.append(tuple(p0))
            for step in range(1, density_samples):
                t = step / density_samples
                curve_point = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * current_v + t ** 2 * p2
                interpolated_path.append(tuple(curve_point))
            interpolated_path.append(tuple(p2))

        interpolated_path.append(tuple(vectors[-1]))
        return interpolated_path

    def reconstruct_path_sequence(self, end_node, evaluated_nodes):
        ordered_x = [self.index_to_coordinate(end_node.x, self.min_x)]
        ordered_y = [self.index_to_coordinate(end_node.y, self.min_y)]
        trace_id = end_node.parent_id
        
        while trace_id != -1:
            node = evaluated_nodes[trace_id]
            ordered_x.append(self.index_to_coordinate(node.x, self.min_x))
            ordered_y.append(self.index_to_coordinate(node.y, self.min_y))
            trace_id = node.parent_id
            
        ordered_x.reverse()
        ordered_y.reverse()
        return ordered_x, ordered_y

    def force_clear_cells(self, target_x, target_y, radius=1):
        for ix in range(target_x - radius, target_x + radius + 1):
            for iy in range(target_y - radius, target_y + radius + 1):
                if 0 <= ix < self.width_x and 0 <= iy < self.width_y:
                    self.obstacle_map[ix][iy] = False

    def is_valid_node(self, node):
        if node.x < 0 or node.y < 0 or node.x >= self.width_x or node.y >= self.width_y:
            return False
        return not self.obstacle_map[node.x][node.y]

    def index_to_coordinate(self, index, min_value):
        return index * self.resolution + min_value

    def coordinate_to_index(self, position, min_value):
        return round((position - min_value) / self.resolution)

    def calculate_grid_id(self, node):
        return node.y * self.width_x + node.x