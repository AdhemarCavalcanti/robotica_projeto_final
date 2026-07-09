import math
import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import dwa_navigation as dw
import mapa_cena as mo


OBJ = {
    "motor_dir": "/MOTOR_DIREITO",
    "motor_esq": "/MOTOR_ESQUERDO",
    "alvo": "/Alvo",
    "floor": "/Floor",
    "sensors": [
        "/SENSOR_MEIO",
        "/SENSOR_DIAG_DIREITO",
        "/SENSOR_DIAG_ESQUERDO",
        "/SENSOR_DIREITO",
        "/SENSOR_ESQUERDO",
    ],
    "ignore": {"Floor", "box", "Goal", "Target", "Alvo", "camera_grade_ocupacao"},
}

SINAL_FWD = +1


class NavegadorCoppelia:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")
        self.planner = dw.DWAController()

        self.h_right = None
        self.h_left = None
        self.h_base = None
        self.h_goal = None
        self.h_sensors = []

        self.x = np.zeros(5, dtype=float)
        self.z0 = 0.0
        self.r0 = 0.0
        self.p0 = 0.0
        self.yaw_shift = 0.0

        self.goal = [0.0, 0.0]
        self.map_pts = np.empty((0, 2), dtype=float)

        self.path_full = []
        self.path_key = []
        self.path_x = []
        self.path_y = []
        self.key_x = []
        self.key_y = []
        self.path_i = 0

        self.k = 0
        self.last_anchor = None
        self.stuck_n = 0
        self.escape_n = 0

    def h(self, path):
        try:
            return self.sim.getObject(path)
        except Exception as e:
            raise RuntimeError(f"Objeto ausente: {path}") from e

    def child_of(self, obj, parent):
        while obj != -1:
            if obj == parent:
                return True
            obj = self.sim.getObjectParent(obj)
        return False

    def p3(self, M, p):
        return np.array([
            M[0] * p[0] + M[1] * p[1] + M[2] * p[2] + M[3],
            M[4] * p[0] + M[5] * p[1] + M[6] * p[2] + M[7],
            M[8] * p[0] + M[9] * p[1] + M[10] * p[2] + M[11],
        ], dtype=float)

    def rect_edges(self, x0, x1, y0, y1, step=0.06):
        pts = []
        t = x0
        while t <= x1:
            pts.append([t, y0])
            pts.append([t, y1])
            t += step
        t = y0
        while t <= y1:
            pts.append([x0, t])
            pts.append([x1, t])
            t += step
        return pts

    def rect_fill(self, M, x0, x1, y0, y1, step=0.04):
        pts = []
        xx = x0
        while xx <= x1:
            yy = y0
            while yy <= y1:
                wp = self.p3(M, [xx, yy, 0.0])
                pts.append([wp[0], wp[1]])
                yy += step
            xx += step
        return pts

    def build_map_fallback(self):
        pts = []
        floor = self.h(OBJ["floor"])
        fp = self.sim.getObjectPosition(floor, -1)
        a = self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_min_x)
        b = self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_max_x)
        c = self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_min_y)
        d = self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_max_y)
        pts.extend(self.rect_edges(fp[0] + a, fp[0] + b, fp[1] + c, fp[1] + d))

        for obj in self.sim.getObjectsInTree(self.sim.handle_scene):
            if self.sim.getObjectType(obj) != self.sim.object_shape_type:
                continue
            alias = self.sim.getObjectAlias(obj, 0)
            if alias in OBJ["ignore"]:
                continue
            if obj == self.h_base or self.child_of(obj, self.h_base):
                continue

            M = self.sim.getObjectMatrix(obj, -1)
            x0 = self.sim.getObjectFloatParam(obj, self.sim.objfloatparam_objbbox_min_x)
            x1 = self.sim.getObjectFloatParam(obj, self.sim.objfloatparam_objbbox_max_x)
            y0 = self.sim.getObjectFloatParam(obj, self.sim.objfloatparam_objbbox_min_y)
            y1 = self.sim.getObjectFloatParam(obj, self.sim.objfloatparam_objbbox_max_y)

            if (x1 - x0) > 4.5 or (y1 - y0) > 4.5:
                continue

            pts.extend(self.rect_fill(M, x0, x1, y0, y1))

        uniq = {(round(x, 2), round(y, 2)): [x, y] for x, y in pts}
        return np.array(list(uniq.values()), dtype=float)

    def align_heading(self):
        front = self.h_sensors[0]
        M = self.sim.getObjectMatrix(front, -1)
        sensor_yaw = math.atan2(M[6], M[2])
        robot_yaw = self.sim.getObjectOrientation(self.h_base, -1)[2]
        delta = sensor_yaw - robot_yaw
        self.yaw_shift = math.atan2(math.sin(delta), math.cos(delta))
        print("Offset (graus):", round(math.degrees(self.yaw_shift), 1))

    def update_state(self, v=0.0, w=0.0):
        pos = self.sim.getObjectPosition(self.h_base, -1)
        ori = self.sim.getObjectOrientation(self.h_base, -1)
        yaw = math.atan2(math.sin(ori[2]), math.cos(ori[2]))
        flip = 0.0 if SINAL_FWD >= 0 else math.pi
        th = math.atan2(math.sin(yaw + self.yaw_shift + flip), math.cos(yaw + self.yaw_shift + flip))
        self.x = np.array([pos[0], pos[1], th, v, w], dtype=float)

    def read_sensors(self):
        out = []
        for s in self.h_sensors:
            trig, dist, local_pt, _, _ = self.sim.readProximitySensor(s)
            if trig > 0:
                p = np.array(local_pt, dtype=float)
                if np.linalg.norm(p) <= 0.0 and dist > 0.0:
                    p = np.array([dist, 0.0, 0.0], dtype=float)
                M = self.sim.getObjectMatrix(s, -1)
                wp = self.p3(M, p)
                out.append([wp[0], wp[1]])
        return np.array(out, dtype=float)

    def local_obstacles(self):
        cloud = []
        if self.map_pts is not None and len(self.map_pts) > 0:
            d = np.hypot(self.map_pts[:, 0] - self.x[0], self.map_pts[:, 1] - self.x[1])
            cloud.extend(self.map_pts[d <= 1.4].tolist())

        live = self.read_sensors()
        if len(live) > 0:
            cloud.extend(live.tolist())

        return np.array(cloud, dtype=float)

    def collides(self, pose):
        if self.map_pts is None or len(self.map_pts) == 0:
            return False
        d = np.hypot(self.map_pts[:, 0] - pose[0], self.map_pts[:, 1] - pose[1])
        return float(np.min(d)) <= self.planner.collision_radius

    def global_plan(self):
        sx, sy = float(self.x[0]), float(self.x[1])
        gx, gy = float(self.goal[0]), float(self.goal[1])

        for rr in (0.22, 0.18, 0.14):
            ap = dw.AStarPlanner(
                self.map_pts[:, 0].tolist(),
                self.map_pts[:, 1].tolist(),
                resolution=0.08,
                rr=rr,
            )
            self.path_x, self.path_y, self.key_x, self.key_y = ap.planning(sx, sy, gx, gy)
            if not ap.last_plan_failed:
                print(f"A* OK com rr={rr:.2f}")
                break
            print(f"A* falhou com rr={rr:.2f}")
        else:
            print("A* falhou geral, linha reta.")
            self.path_x, self.path_y, self.key_x, self.key_y = [sx, gx], [sy, gy], [sx, gx], [sy, gy]

        self.path_full = list(zip(self.path_x, self.path_y))
        self.path_key = list(zip(self.key_x, self.key_y))
        self.path_i = 0

    def target_point(self):
        route = self.path_full if self.path_full else [self.goal]
        while self.path_i < len(route) - 1:
            pt = route[self.path_i]
            if math.hypot(pt[0] - self.x[0], pt[1] - self.x[1]) >= 0.40:
                break
            self.path_i += 1
        return route[min(self.path_i + 2, len(route) - 1)]

    def escape_cmd(self, tgt):
        ang = math.atan2(tgt[1] - self.x[1], tgt[0] - self.x[0])
        turn = math.atan2(math.sin(ang - self.x[2]), math.cos(ang - self.x[2]))
        w = max(min(1.2 * turn, 4.0), -4.0)
        if self.escape_n > 22:
            return [0.07, 0.3 * w]
        if abs(turn) > 0.25:
            return [0.0, -0.8 if turn >= 0.0 else 0.8]
        return [-0.10, 0.5 * w]

    def drive(self, cmd):
        v, w = float(cmd[0]), float(cmd[1])
        dt = self.planner.dt
        R, L = 0.0375, 0.15

        nxt = self.x.copy()
        nxt[2] = math.atan2(math.sin(nxt[2] + w * dt), math.cos(nxt[2] + w * dt))
        nxt[0] += v * math.cos(nxt[2]) * dt
        nxt[1] += v * math.sin(nxt[2]) * dt
        nxt[3], nxt[4] = v, w

        if self.collides(nxt):
            nxt = self.x.copy()
            nxt[2] = math.atan2(math.sin(nxt[2] + w * dt), math.cos(nxt[2] + w * dt))
            nxt[3], nxt[4] = 0.0, w
            v = 0.0

        vinv, winv = -v, -w
        wr = (2.0 * vinv + winv * L) / (2.0 * R)
        wl = (2.0 * vinv - winv * L) / (2.0 * R)
        wr = max(min(wr, 20.0), -20.0)
        wl = max(min(wl, 20.0), -20.0)

        self.sim.setJointTargetVelocity(self.h_right, wr)
        self.sim.setJointTargetVelocity(self.h_left, wl)
        self.sim.setObjectPosition(self.h_base, -1, [nxt[0], nxt[1], self.z0])

        flip = 0.0 if SINAL_FWD >= 0 else math.pi
        yaw_cmd = math.atan2(math.sin(nxt[2] - self.yaw_shift - flip), math.cos(nxt[2] - self.yaw_shift - flip))
        self.sim.setObjectOrientation(self.h_base, -1, [self.r0, self.p0, yaw_cmd])

        try:
            self.sim.resetDynamicObject(self.h_base)
        except Exception:
            pass

        self.sim.step()
        return nxt

    def render(self):
        if self.map_pts is None or len(self.map_pts) == 0:
            print("[Aviso] Não foi possível abrir o mapa pois os obstáculos não foram calculados.")
            return

        floor = self.h(OBJ["floor"])
        fp = self.sim.getObjectPosition(floor, -1)
        x0 = fp[0] + self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_min_x)
        x1 = fp[0] + self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_max_x)
        y0 = fp[1] + self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_min_y)
        y1 = fp[1] + self.sim.getObjectFloatParam(floor, self.sim.objfloatparam_objbbox_max_y)

        print("\n---> ABRINDO O MAPA INICIAL <---")
        print("Analise as rotas geradas pelo A*. FECHE A JANELA DO GRÁFICO para iniciar a simulação no CoppeliaSim.\n")

        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("Planejamento Estático - A*", fontsize=12, fontweight="bold")

        obs = np.array(self.map_pts)
        route = np.array(self.path_full)
        key = np.array(self.path_key)

        for ax, ttl in zip(axs, ["Rota Completa", "Pontos-Chave"]):
            ax.set_title(ttl)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)
            ax.axis("equal")
            ax.set_xlim(x0 - 0.1, x1 + 0.1)
            ax.set_ylim(y0 - 0.1, y1 + 0.1)
            if len(obs) > 0:
                ax.scatter(obs[:, 0], obs[:, 1], s=4, c="black", marker="s")
            ax.plot(self.x[0], self.x[1], "go", markersize=9)
            ax.plot(self.goal[0], self.goal[1], "ro", markersize=9)

        if len(route) > 0:
            axs[0].plot(route[:, 0], route[:, 1], "b-", linewidth=2)
        if len(key) > 0:
            axs[1].plot(key[:, 0], key[:, 1], "y-x", markersize=8, linewidth=1.5)

        plt.tight_layout()
        plt.show()

    def init_scene(self):
        self.h_right = self.h(OBJ["motor_dir"])
        self.h_left = self.h(OBJ["motor_esq"])
        self.h_base = self.sim.getObjectParent(self.h_right)
        self.h_goal = self.h(OBJ["alvo"])
        self.h_sensors = [self.h(p) for p in OBJ["sensors"]]

        pos = self.sim.getObjectPosition(self.h_base, -1)
        ori = self.sim.getObjectOrientation(self.h_base, -1)
        self.z0, self.r0, self.p0 = pos[2], ori[0], ori[1]

        self.align_heading()
        self.update_state()

        gpos = self.sim.getObjectPosition(self.h_goal, -1)
        self.goal = [gpos[0], gpos[1]]
        self.map_pts = self._map()
        self.k = 0
        self.global_plan()

    def _map(self):
        ex = [
            (float(self.x[0]), float(self.x[1]), self.planner.robot_radius + 0.15),
            (float(self.goal[0]), float(self.goal[1]), self.planner.robot_radius + 0.10),
        ]
        try:
            floor = self.h(OBJ["floor"])
            pts, _, _ = mo.construir_obstaculos_por_visao(self.sim, floor, excluir=ex)
            if len(pts) >= 4:
                return pts
        except Exception:
            print("Usando fallback Bounding Box...")
        return self.build_map_fallback()

    def loop(self):
        if self.h_goal is not None:
            gpos = self.sim.getObjectPosition(self.h_goal, -1)
            self.goal = [gpos[0], gpos[1]]

        cloud = self.local_obstacles()
        tgt = self.target_point()

        if self.last_anchor is None or math.hypot(self.x[0] - self.last_anchor[0], self.x[1] - self.last_anchor[1]) > 0.05:
            self.last_anchor = (float(self.x[0]), float(self.x[1]))
            self.stuck_n = 0
        else:
            self.stuck_n += 1

        if self.escape_n > 0:
            cmd = self.escape_cmd(tgt)
            self.escape_n -= 1
        else:
            cmd, _ = self.planner.plan(self.x[0:3], self.x[3], self.x[4], tgt, cloud)
            if self.stuck_n >= 30:
                print("Preso -> replanejando...")
                self.global_plan()
                self.escape_n = 35
                self.stuck_n = 0

        self.x = self.drive(cmd)
        self.k += 1

        if math.hypot(self.x[0] - self.goal[0], self.x[1] - self.goal[1]) <= 0.20:
            print("ALVO ATINGIDO!")
            return True
        return False


if __name__ == "__main__":
    print("Conectando ao CoppeliaSim...")
    nav = NavegadorCoppelia()

    nav.init_scene()
    nav.render()

    print("Iniciando a simulação física...")
    nav.sim.setStepping(True)
    nav.sim.startSimulation()

    try:
        while not nav.loop():
            pass
    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")
    finally:
        if nav.h_right and nav.h_left:
            nav.sim.setJointTargetVelocity(nav.h_right, 0.0)
            nav.sim.setJointTargetVelocity(nav.h_left, 0.0)
        nav.sim.stopSimulation()
        print("Simulação encerrada e finalizada.")