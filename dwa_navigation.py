import math
import numpy as np


def norm_ang(ang):
    return math.atan2(math.sin(ang), math.cos(ang))


# ==========================================================================
#  CONTROLADOR LOCAL - DWA (CONFIGURADO PARA MOVIMENTOS MAIS LENTOS)
# ==========================================================================
class DWAController:
    def __init__(self):
        
        self.max_v = 0.20          
        self.min_v = 0.0
        self.max_w = 0.70          
        self.max_dv = 0.4         
        self.max_dw = 1.5          
      

        self.v_res = 0.02
        self.w_res = 0.08
        self.dt = 0.1
        self.p_time = 2.5

        self.robot_radius = 0.24
        self.collision_radius = 0.16
        self.margin = 0.06

        
        self.w_goal = 0.40         
        self.w_speed = 1.0
        self.w_obs = 0.45
        self.w_dist = 2.2

    def calc_dw(self, v, w):
        r_lim = [self.min_v, self.max_v, -self.max_w, self.max_w]
        d_lim = [
            v - self.max_dv * self.dt,
            v + self.max_dv * self.dt,
            w - self.max_dw * self.dt,
            w + self.max_dw * self.dt,
        ]
        return [
            max(r_lim[0], d_lim[0]),
            min(r_lim[1], d_lim[1]),
            max(r_lim[2], d_lim[2]),
            min(r_lim[3], d_lim[3]),
        ]

    def motion(self, x, v, w):
        state = np.array(x, dtype=float)
        state[2] = norm_ang(state[2] + w * self.dt)
        state[0] += v * math.cos(state[2]) * self.dt
        state[1] += v * math.sin(state[2]) * self.dt
        return state

    def predict_traj(self, x_init, v, w):
        curr = np.array(x_init, dtype=float)
        traj = [curr.copy()]
        t = 0.0
        while t <= self.p_time:
            curr = self.motion(curr, v, w)
            traj.append(curr.copy())
            t += self.dt
        return np.array(traj)

    def cost_goal(self, traj, goal):
        last = traj[-1]
        g_ang = math.atan2(goal[1] - last[1], goal[0] - last[0])
        return abs(norm_ang(g_ang - last[2]))

    def cost_obs(self, traj, obs, v):
        if obs is None or len(obs) == 0:
            return 0.0

        dx = traj[:, 0:1] - obs[:, 0]
        dy = traj[:, 1:2] - obs[:, 1]
        dists = np.hypot(dx, dy)
        min_d = float(np.min(dists))

        stop_d = (v * v) / (2.0 * self.max_dv) if self.max_dv > 0 else 0.0
        clearance = self.robot_radius + self.margin + 0.5 * stop_d

        if min_d <= self.collision_radius:
            return float("inf")

        cost = 1.0 / (min_d - self.collision_radius)
        if min_d < clearance:
            cost += 8.0 * (clearance - min_d) / clearance

        return cost

    def plan(self, x, v, w, goal, obs):
        dw_lim = self.calc_dw(v, w)
        best_u = [0.0, 0.0]
        best_traj = self.predict_traj(x, 0.0, 0.0)
        min_c = float("inf")
        curr_g_dist = math.hypot(goal[0] - x[0], goal[1] - x[1])

        for cv in np.arange(dw_lim[0], dw_lim[1] + self.v_res, self.v_res):
            if cv > dw_lim[1]:
                continue

            for cw in np.arange(dw_lim[2], dw_lim[3] + self.w_res, self.w_res):
                if cw > dw_lim[3]:
                    continue

                traj = self.predict_traj(x, cv, cw)
                c_obs = self.cost_obs(traj, obs, cv)
                if math.isinf(c_obs):
                    continue

                c_goal = self.w_goal * self.cost_goal(traj, goal)
                c_speed = self.w_speed * (self.max_v - cv)
                final_g_dist = math.hypot(goal[0] - traj[-1, 0], goal[1] - traj[-1, 1])
                c_dist = self.w_dist * final_g_dist
                reward = 2.0 * max(0.0, curr_g_dist - final_g_dist)

                total_c = c_goal + c_speed + self.w_obs * c_obs + c_dist - reward

                if total_c < min_c:
                    min_c = total_c
                    best_u = [float(cv), float(cw)]
                    best_traj = traj

        if math.isinf(min_c):
            g_ang = math.atan2(goal[1] - x[1], goal[0] - x[0])
            turn = norm_ang(g_ang - x[2])
            best_u = [0.08, 0.8 if turn >= 0.0 else -0.8]
            best_traj = self.predict_traj(x, best_u[0], best_u[1])

        return best_u, best_traj


# ==========================================================================
#  PLANEJADOR GLOBAL - A* MODIFICADO
# ==========================================================================
class AStarPlanner:
    class Node:
        def __init__(self, x, y, cost, p_idx):
            self.x = x
            self.y = y
            self.cost = cost
            self.p_idx = p_idx

    def __init__(self, ox, oy, resolution=0.1, rr=0.28):
        self.res = resolution
        self.rr = rr
        self.min_x = math.floor(min(ox))
        self.min_y = math.floor(min(oy))
        self.max_x = math.ceil(max(ox))
        self.max_y = math.ceil(max(oy))
        self.w_x = round((self.max_x - self.min_x) / self.res)
        self.w_y = round((self.max_y - self.min_y) / self.res)
        self.ds = [
            [1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
            [1, 1, math.sqrt(2)], [1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)], [-1, -1, math.sqrt(2)],
        ]
        self.obs_map = [[False for _ in range(self.w_y)] for _ in range(self.w_x)]
        self.inf_cells = math.ceil(self.rr / self.res)
        self.last_plan_failed = False

        for iox, ioy in zip(ox, oy):
            cx = self.to_idx(iox, self.min_x)
            cy = self.to_idx(ioy, self.min_y)
            for ix in range(cx - self.inf_cells, cx + self.inf_cells + 1):
                for iy in range(cy - self.inf_cells, cy + self.inf_cells + 1):
                    if ix < 0 or iy < 0 or ix >= self.w_x or iy >= self.w_y:
                        continue
                    px = self.to_pos(ix, self.min_x)
                    py = self.to_pos(iy, self.min_y)
                    if math.hypot(iox - px, ioy - py) <= self.rr:
                        self.obs_map[ix][iy] = True

    def planning(self, sx, sy, gx, gy):
        start = self.Node(self.to_idx(sx, self.min_x), self.to_idx(sy, self.min_y), 0.0, -1)
        goal = self.Node(self.to_idx(gx, self.min_x), self.to_idx(gy, self.min_y), 0.0, -1)
        
        self.last_plan_failed = False
        limpar = self.inf_cells + 1
        self.clear_cell(start.x, start.y, limpar)
        self.clear_cell(goal.x, goal.y, limpar)

        dist_total = max(math.hypot(goal.x - start.x, goal.y - start.y), 1e-6)
        open_set = {self.to_grid_id(start): start}
        closed_set = {}

        while open_set:
            def eval_node(key):
                n = open_set[key]
                d = math.hypot(goal.x - n.x, goal.y - n.y)
                w = 1.0 + d / dist_total
                return n.cost + w * d

            curr_id = min(open_set, key=eval_node)
            curr = open_set[curr_id]

            if curr.x == goal.x and curr.y == goal.y:
                goal.p_idx = curr.p_idx
                goal.cost = curr.cost
                rx, ry = self.gen_path(goal, closed_set)
                return self.post_process(rx, ry)

            del open_set[curr_id]
            closed_set[curr_id] = curr

            for dx, dy, step_c in self.ds:
                node = self.Node(curr.x + dx, curr.y + dy, curr.cost + step_c, curr_id)
                n_id = self.to_grid_id(node)

                if not self.valid_node(node) or n_id in closed_set:
                    continue

                if n_id not in open_set or open_set[n_id].cost > node.cost:
                    open_set[n_id] = node

        self.last_plan_failed = True
        return self.post_process([sx, gx], [sy, gy])

    def post_process(self, rx, ry):
        path = list(zip(rx, ry))
        if len(path) <= 2:
            return rx, ry, [p[0] for p in path], [p[1] for p in path]

        k_pts = self._clip_collinear(path)
        k_pts = self._los_simplify(k_pts)
        smooth = self._bezier(k_pts)

        return (
            [p[0] for p in smooth], [p[1] for p in smooth],
            [p[0] for p in k_pts], [p[1] for p in k_pts]
        )

    @staticmethod
    def _clip_collinear(path, eps=1e-6):
        if len(path) <= 2:
            return list(path)
        res = [path[0]]
        for i in range(1, len(path) - 1):
            ax, ay = res[-1]
            bx, by = path[i]
            cx, cy = path[i + 1]
            if abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) > eps:
                res.append(path[i])
        res.append(path[-1])
        return res

    def _los_simplify(self, pts):
        if len(pts) <= 2:
            return list(pts)
        res = [pts[0]]
        i = 1
        while i < len(pts) - 1:
            if self.line_free(res[-1], pts[i + 1]):
                i += 1
            else:
                res.append(pts[i])
                i += 1
        res.append(pts[-1])
        return res

    def line_free(self, p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(2, int(length / (self.res * 0.5)) + 1)

        for s in range(steps + 1):
            t = s / steps
            ix = self.to_idx(x0 + (x1 - x0) * t, self.min_x)
            iy = self.to_idx(y0 + (y1 - y0) * t, self.min_y)
            if ix < 0 or iy < 0 or ix >= self.w_x or iy >= self.w_y:
                return False
            if self.obs_map[ix][iy]:
                return False
        return True

    @staticmethod
    def _bezier(pts, samples=12, ratio=0.35):
        if len(pts) <= 2:
            return list(pts)
        arr = [np.array(p, dtype=float) for p in pts]
        smooth = [tuple(arr[0])]

        for i in range(1, len(arr) - 1):
            p_prev, p_curr, p_next = arr[i - 1], arr[i], arr[i + 1]
            p0 = p_curr + ratio * (p_prev - p_curr)
            p2 = p_curr + ratio * (p_next - p_curr)

            smooth.append(tuple(p0))
            for s in range(1, samples):
                t = s / samples
                b = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p_curr + t ** 2 * p2
                smooth.append(tuple(b))
            smooth.append(tuple(p2))

        smooth.append(tuple(arr[-1]))
        return smooth

    def gen_path(self, goal, closed_set):
        rx = [self.to_pos(goal.x, self.min_x)]
        ry = [self.to_pos(goal.y, self.min_y)]
        parent = goal.p_idx
        while parent != -1:
            n = closed_set[parent]
            rx.append(self.to_pos(n.x, self.min_x))
            ry.append(self.to_pos(n.y, self.min_y))
            parent = n.p_idx
        rx.reverse()
        ry.reverse()
        return rx, ry

    def clear_cell(self, cx, cy, r=1):
        for ix in range(cx - r, cx + r + 1):
            for iy in range(cy - r, cy + r + 1):
                if 0 <= ix < self.w_x and 0 <= iy < self.w_y:
                    self.obs_map[ix][iy] = False

    def valid_node(self, node):
        if node.x < 0 or node.y < 0 or node.x >= self.w_x or node.y >= self.w_y:
            return False
        return not self.obs_map[node.x][node.y]

    def to_pos(self, index, min_pos):
        return index * self.res + min_pos

    def to_idx(self, pos, min_pos):
        return round((pos - min_pos) / self.res)

    def to_grid_id(self, node):
        return node.y * self.w_x + node.x