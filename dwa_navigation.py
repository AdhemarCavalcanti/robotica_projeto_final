import math
import numpy as np


def angle_wrap(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def diff_drive_predict(pose, v, w, dt):
    nxt = np.array(pose, dtype=float)
    nxt[2] = angle_wrap(nxt[2] + w * dt)
    nxt[0] += v * math.cos(nxt[2]) * dt
    nxt[1] += v * math.sin(nxt[2]) * dt
    return nxt


def bounding_box_to_points(sim, obj_handle, step=0.04):
    x0 = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_x)
    x1 = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_x)
    y0 = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_y)
    y1 = sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_y)
    mat = sim.getObjectMatrix(obj_handle, -1)

    pts = []
    xx = x0
    while xx <= x1:
        yy = y0
        while yy <= y1:
            p = np.array([xx, yy, 0.0], dtype=float)
            wp = np.array([
                mat[0] * p[0] + mat[1] * p[1] + mat[2] * p[2] + mat[3],
                mat[4] * p[0] + mat[5] * p[1] + mat[6] * p[2] + mat[7],
                mat[8] * p[0] + mat[9] * p[1] + mat[10] * p[2] + mat[11],
            ], dtype=float)
            pts.append([wp[0], wp[1]])
            yy += step
        xx += step
    return pts


def rect_outline_points(sim, obj_handle, step=0.06):
    pos = sim.getObjectPosition(obj_handle, -1)
    x0 = pos[0] + sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_x)
    x1 = pos[0] + sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_x)
    y0 = pos[1] + sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_min_y)
    y1 = pos[1] + sim.getObjectFloatParam(obj_handle, sim.objfloatparam_objbbox_max_y)

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


def unique_points(points):
    bucket = {(round(x, 2), round(y, 2)): [x, y] for x, y in points}
    return np.array(list(bucket.values()), dtype=float)


def build_occupancy_from_scene(sim, robot_handle, floor_handle, ignore_aliases, big_limit=4.5):
    pts = []
    pts.extend(rect_outline_points(sim, floor_handle))

    for obj in sim.getObjectsInTree(sim.handle_scene):
        if sim.getObjectType(obj) != sim.object_shape_type:
            continue

        alias = sim.getObjectAlias(obj, 0)
        if alias in ignore_aliases:
            continue
        if obj == robot_handle or is_descendant(sim, obj, robot_handle):
            continue

        x0 = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_x)
        x1 = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_x)
        y0 = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_y)
        y1 = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_y)

        if (x1 - x0) > big_limit or (y1 - y0) > big_limit:
            continue

        pts.extend(bounding_box_to_points(sim, obj))

    return unique_points(pts)


def is_descendant(sim, current_handle, target_parent):
    while current_handle != -1:
        if current_handle == target_parent:
            return True
        current_handle = sim.getObjectParent(current_handle)
    return False


def read_sensor_cloud(sim, sensor_handles):
    hits = []
    for sensor in sensor_handles:
        res, dist, local_pt, _, _ = sim.readProximitySensor(sensor)
        if res > 0:
            pt = np.array(local_pt, dtype=float)
            if np.linalg.norm(pt) <= 0.0 and dist > 0.0:
                pt = np.array([dist, 0.0, 0.0], dtype=float)
            mat = sim.getObjectMatrix(sensor, -1)
            world_pt = np.array([
                mat[0] * pt[0] + mat[1] * pt[1] + mat[2] * pt[2] + mat[3],
                mat[4] * pt[0] + mat[5] * pt[1] + mat[6] * pt[2] + mat[7],
                mat[8] * pt[0] + mat[9] * pt[1] + mat[10] * pt[2] + mat[11],
            ], dtype=float)
            hits.append([world_pt[0], world_pt[1]])
    return np.array(hits, dtype=float)


def local_obstacle_cloud(robot_xy, static_pts, live_pts, radius=1.4):
    cloud = []
    if static_pts is not None and len(static_pts) > 0:
        dist = np.hypot(static_pts[:, 0] - robot_xy[0], static_pts[:, 1] - robot_xy[1])
        cloud.extend(static_pts[dist <= radius].tolist())
    if live_pts is not None and len(live_pts) > 0:
        cloud.extend(live_pts.tolist())
    return np.array(cloud, dtype=float)


class DWAController:
    def __init__(self):
        self.max_v = 0.20
        self.min_v = 0.0
        self.max_yaw_rate = 0.70
        self.max_acceleration = 0.4
        self.max_steer_acceleration = 1.5

        self.v_resolution = 0.02
        self.w_resolution = 0.08
        self.dt = 0.1
        self.prediction_time = 2.5

        self.robot_radius = 0.24
        self.collision_radius = 0.16
        self.safety_margin = 0.06

        self.weight_goal = 0.40
        self.weight_speed = 1.0
        self.weight_obstacle = 0.45
        self.weight_distance = 2.2

    def calculate_dynamic_window(self, current_v, current_w):
        v0 = max(self.min_v, current_v - self.max_acceleration * self.dt)
        v1 = min(self.max_v, current_v + self.max_acceleration * self.dt)
        w0 = max(-self.max_yaw_rate, current_w - self.max_steer_acceleration * self.dt)
        w1 = min(self.max_yaw_rate, current_w + self.max_steer_acceleration * self.dt)
        return [v0, v1, w0, w1]

    def process_motion_step(self, pose, v, w):
        return diff_drive_predict(pose, v, w, self.dt)

    def generate_predicted_trajectory(self, initial_pose, v, w):
        state = np.array(initial_pose, dtype=float)
        path = [state.copy()]
        for _ in np.arange(0.0, self.prediction_time + self.dt, self.dt):
            state = self.process_motion_step(state, v, w)
            path.append(state.copy())
        return np.array(path)

    def evaluate_goal_cost(self, trajectory, goal_xy):
        end = trajectory[-1]
        desired = math.atan2(goal_xy[1] - end[1], goal_xy[0] - end[0])
        return abs(angle_wrap(desired - end[2]))

    def evaluate_obstacle_cost(self, trajectory, obstacles, v):
        if obstacles is None or len(obstacles) == 0:
            return 0.0

        dx = trajectory[:, 0:1] - obstacles[:, 0]
        dy = trajectory[:, 1:2] - obstacles[:, 1]
        min_dist = float(np.min(np.hypot(dx, dy)))

        brake = (v * v) / (2.0 * self.max_acceleration) if self.max_acceleration > 0 else 0.0
        safe = self.robot_radius + self.safety_margin + 0.5 * brake

        if min_dist <= self.collision_radius:
            return float("inf")

        cost = 1.0 / (min_dist - self.collision_radius)
        if min_dist < safe:
            cost += 8.0 * (safe - min_dist) / safe
        return cost

    def plan(self, pose, current_v, current_w, goal_xy, obstacles):
        dw = self.calculate_dynamic_window(current_v, current_w)
        best_u = [0.0, 0.0]
        best_traj = self.generate_predicted_trajectory(pose, 0.0, 0.0)
        best_cost = float("inf")
        start_dist = math.hypot(goal_xy[0] - pose[0], goal_xy[1] - pose[1])

        v_set = np.arange(dw[0], dw[1] + self.v_resolution, self.v_resolution)
        w_set = np.arange(dw[2], dw[3] + self.w_resolution, self.w_resolution)

        for v in v_set:
            if v > dw[1]:
                continue
            for w in w_set:
                if w > dw[3]:
                    continue

                traj = self.generate_predicted_trajectory(pose, v, w)
                obst_cost = self.evaluate_obstacle_cost(traj, obstacles, v)
                if math.isinf(obst_cost):
                    continue

                goal_cost = self.weight_goal * self.evaluate_goal_cost(traj, goal_xy)
                speed_cost = self.weight_speed * (self.max_v - v)
                end_dist = math.hypot(goal_xy[0] - traj[-1, 0], goal_xy[1] - traj[-1, 1])
                dist_cost = self.weight_distance * end_dist
                reward = 2.0 * max(0.0, start_dist - end_dist)

                total = goal_cost + speed_cost + (self.weight_obstacle * obst_cost) + dist_cost - reward

                if total < best_cost:
                    best_cost = total
                    best_u = [float(v), float(w)]
                    best_traj = traj

        if math.isinf(best_cost):
            heading = math.atan2(goal_xy[1] - pose[1], goal_xy[0] - pose[0])
            err = angle_wrap(heading - pose[2])
            best_u = [0.08, 0.8 if err >= 0.0 else -0.8]
            best_traj = self.generate_predicted_trajectory(pose, best_u[0], best_u[1])

        return best_u, best_traj


class AStarPlanner:
    class GridNode:
        def __init__(self, x_idx, y_idx, cost, parent_id):
            self.x = x_idx
            self.y = y_idx
            self.cost = cost
            self.parent_id = parent_id

    def __init__(self, ox, oy, resolution=0.1, rr=0.28):
        self.resolution = resolution
        self.robot_radius = rr

        self.min_x = math.floor(min(ox))
        self.min_y = math.floor(min(oy))
        self.max_x = math.ceil(max(ox))
        self.max_y = math.ceil(max(oy))

        self.width_x = round((self.max_x - self.min_x) / self.resolution)
        self.width_y = round((self.max_y - self.min_y) / self.resolution)

        self.motion_directions = [
            [1, 0, 1.0], [0, 1, 1.0], [-1, 0, 1.0], [0, -1, 1.0],
            [1, 1, math.sqrt(2)], [1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)], [-1, -1, math.sqrt(2)],
        ]

        self.obstacle_map = [[False for _ in range(self.width_y)] for _ in range(self.width_x)]
        self.inflation_cells = math.ceil(self.robot_radius / self.resolution)
        self.last_plan_failed = False

        for ox_i, oy_i in zip(ox, oy):
            cx = self.coordinate_to_index(ox_i, self.min_x)
            cy = self.coordinate_to_index(oy_i, self.min_y)
            for ix in range(cx - self.inflation_cells, cx + self.inflation_cells + 1):
                for iy in range(cy - self.inflation_cells, cy + self.inflation_cells + 1):
                    if ix < 0 or iy < 0 or ix >= self.width_x or iy >= self.width_y:
                        continue
                    px = self.index_to_coordinate(ix, self.min_x)
                    py = self.index_to_coordinate(iy, self.min_y)
                    if math.hypot(ox_i - px, oy_i - py) <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True

    def planning(self, start_x, start_y, goal_x, goal_y):
        start_node = self.GridNode(
            self.coordinate_to_index(start_x, self.min_x),
            self.coordinate_to_index(start_y, self.min_y),
            0.0,
            -1,
        )
        goal_node = self.GridNode(
            self.coordinate_to_index(goal_x, self.min_x),
            self.coordinate_to_index(goal_y, self.min_y),
            0.0,
            -1,
        )

        self.last_plan_failed = False
        pad = self.inflation_cells + 1
        self.force_clear_cells(start_node.x, start_node.y, pad)
        self.force_clear_cells(goal_node.x, goal_node.y, pad)

        total_distance = max(math.hypot(goal_node.x - start_node.x, goal_node.y - start_node.y), 1e-6)
        open_set = {self.calculate_grid_id(start_node): start_node}
        closed_set = {}

        while open_set:
            current_id = min(
                open_set,
                key=lambda k: open_set[k].cost + (1.0 + math.hypot(goal_node.x - open_set[k].x, goal_node.y - open_set[k].y) / total_distance)
                * math.hypot(goal_node.x - open_set[k].x, goal_node.y - open_set[k].y),
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
                neighbor = self.GridNode(
                    current_node.x + dx,
                    current_node.y + dy,
                    current_node.cost + movement_cost,
                    current_id,
                )
                neighbor_id = self.calculate_grid_id(neighbor)

                if not self.is_valid_node(neighbor) or neighbor_id in closed_set:
                    continue

                if neighbor_id not in open_set or open_set[neighbor_id].cost > neighbor.cost:
                    open_set[neighbor_id] = neighbor

        self.last_plan_failed = True
        return self.apply_post_processing([start_x, goal_x], [start_y, goal_y])

    def apply_post_processing(self, raw_x, raw_y):
        coords = list(zip(raw_x, raw_y))
        if len(coords) <= 2:
            return raw_x, raw_y, [p[0] for p in coords], [p[1] for p in coords]

        key_points = self._remove_collinear_nodes(coords)
        key_points = self._compress_by_line_of_sight(key_points)
        smooth = self._generate_bezier_curves(key_points)

        return (
            [p[0] for p in smooth],
            [p[1] for p in smooth],
            [p[0] for p in key_points],
            [p[1] for p in key_points],
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
        samples = max(2, int(segment_len / (self.resolution * 0.5)) + 1)

        for step in range(samples + 1):
            ratio = step / samples
            ix = self.coordinate_to_index(xa + (xb - xa) * ratio, self.min_x)
            iy = self.coordinate_to_index(ya + (yb - ya) * ratio, self.min_y)

            if ix < 0 or iy < 0 or ix >= self.width_x or iy >= self.width_y:
                return False
            if self.obstacle_map[ix][iy]:
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