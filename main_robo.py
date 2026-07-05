import math
import numpy as np
import matplotlib.pyplot as plt
import threading

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import dwa_navigation as dw
import mapa_ocupacao as mo

SINAL_FWD = +1

client = RemoteAPIClient()
sim = client.require("sim")

ctx = {}
dwa = dw.DWAController()


def norm_ang(ang):
    return math.atan2(math.sin(ang), math.cos(ang))


def get_obj(path):
    try:
        return sim.getObject(path)
    except Exception as err:
        raise RuntimeError(f"Objeto ausente: {path}") from err


def is_child(h, parent):
    curr = h
    while curr != -1:
        if curr == parent:
            return True
        curr = sim.getObjectParent(curr)
    return False


def trans_pt(mat, pt):
    return np.array(
        [
            mat[0] * pt[0] + mat[1] * pt[1] + mat[2] * pt[2] + mat[3],
            mat[4] * pt[0] + mat[5] * pt[1] + mat[6] * pt[2] + mat[7],
            mat[8] * pt[0] + mat[9] * pt[1] + mat[10] * pt[2] + mat[11],
        ],
        dtype=float,
    )


def add_rect_pts(pts, x0, x1, y0, y1, step=0.06):
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


def fill_rot_rect(pts, mat, x0, x1, y0, y1, step=0.04):
    x = x0
    while x <= x1:
        y = y0
        while y <= y1:
            w_pt = trans_pt(mat, [x, y, 0.0])
            pts.append([w_pt[0], w_pt[1]])
            y += step
        x += step


def get_static_obs():
    pts = []
    floor = get_obj("/Floor")
    f_pos = sim.getObjectPosition(floor, -1)

    f_x0 = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_x)
    f_x1 = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_x)
    f_y0 = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_y)
    f_y1 = sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_y)

    add_rect_pts(pts, f_pos[0] + f_x0, f_pos[0] + f_x1, f_pos[1] + f_y0, f_pos[1] + f_y1)

    for obj in sim.getObjectsInTree(sim.handle_scene):
        if sim.getObjectType(obj) != sim.object_shape_type:
            continue

        alias = sim.getObjectAlias(obj, 0)
        if alias in {"Floor", "box", "Goal", "Target", "Alvo", "camera_grade_ocupacao"}:
            continue

        if obj == ctx["rb"] or is_child(obj, ctx["rb"]):
            continue

        mat = sim.getObjectMatrix(obj, -1)
        x0 = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_x)
        x1 = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_x)
        y0 = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_min_y)
        y1 = sim.getObjectFloatParam(obj, sim.objfloatparam_objbbox_max_y)

        if (x1 - x0) > 4.5 or (y1 - y0) > 4.5:
            continue

        fill_rot_rect(pts, mat, x0, x1, y0, y1)

    uniq = {(round(x, 2), round(y, 2)): [x, y] for x, y in pts}
    return np.array(list(uniq.values()), dtype=float)


def get_rb_state(v=0.0, w=0.0):
    pos = sim.getObjectPosition(ctx["rb"], -1)
    ori = sim.getObjectOrientation(ctx["rb"], -1)
    offset = ctx.get("h_off", 0.0)
    flip = 0.0 if SINAL_FWD >= 0 else math.pi
    th = norm_ang(ori[2] + offset + flip)
    return np.array([pos[0], pos[1], th, v, w], dtype=float)


def calc_h_offset():
    s_front = ctx["ss"][0]
    mat = sim.getObjectMatrix(s_front, -1)
    yaw_f = math.atan2(mat[6], mat[2])
    yaw_m = sim.getObjectOrientation(ctx["rb"], -1)[2]
    ctx["h_off"] = norm_ang(yaw_f - yaw_m)
    print("Offset (graus):", round(math.degrees(ctx["h_off"]), 1))


def get_sensor_obs():
    obs = []
    for s in ctx["ss"]:
        res, dist, pt, _, _ = sim.readProximitySensor(s)
        if res > 0:
            d_pt = np.array(pt, dtype=float)
            if np.linalg.norm(d_pt) <= 0.0 and dist > 0.0:
                d_pt = np.array([dist, 0.0, 0.0], dtype=float)
            mat = sim.getObjectMatrix(s, -1)
            w_obs = trans_pt(mat, d_pt)
            obs.append([w_obs[0], w_obs[1]])
    return np.array(obs, dtype=float)


def get_all_obs(x):
    local_obs = []
    st_obs = ctx.get("st_obs")
    if st_obs is not None and len(st_obs) > 0:
        dists = np.hypot(st_obs[:, 0] - x[0], st_obs[:, 1] - x[1])
        local_obs.extend(st_obs[dists <= 1.4].tolist())

    s_obs = get_sensor_obs()
    if len(s_obs) > 0:
        local_obs.extend(s_obs.tolist())
    return np.array(local_obs, dtype=float)


def check_collision(x):
    st_obs = ctx.get("st_obs")
    if st_obs is None or len(st_obs) == 0:
        return False
    dists = np.hypot(st_obs[:, 0] - x[0], st_obs[:, 1] - x[1])
    return float(np.min(dists)) <= dwa.collision_radius


def plan_global():
    obs = ctx["st_obs"]
    sx, sy = float(ctx["x"][0]), float(ctx["x"][1])
    gx, gy = float(ctx["goal_xy"][0]), float(ctx["goal_xy"][1])
    rx = ry = kx = ky = None

    for rr in (0.22, 0.18, 0.14):
        planner = dw.AStarPlanner(obs[:, 0].tolist(), obs[:, 1].tolist(), resolution=0.08, rr=rr)
        rx, ry, kx, ky = planner.planning(sx, sy, gx, gy)
        if not planner.last_plan_failed:
            print(f"A* OK com rr={rr:.2f}")
            break
        print(f"A* falhou com rr={rr:.2f}")
    else:
        print("A* falhou geral, linha reta.")
        rx, ry, kx, ky = [sx, gx], [sy, gy], [sx, gx], [sy, gy]

    ctx["g_path"] = list(zip(rx, ry))
    ctx["k_pts"] = list(zip(kx, ky))
    ctx["p_idx"] = 0


def get_target(x):
    path = ctx.get("g_path", [ctx["goal_xy"]])
    while ctx["p_idx"] < len(path) - 1:
        t = path[ctx["p_idx"]]
        if math.hypot(t[0] - x[0], t[1] - x[1]) >= 0.40:
            break
        ctx["p_idx"] += 1
    idx = min(ctx["p_idx"] + 2, len(path) - 1)
    return path[idx]


def move_robot(u):
    v, w = float(u[0]), float(u[1])
    dt = dwa.dt
    r_w, base = 0.0375, 0.15

    x = ctx["x"].copy()
    x[2] = norm_ang(x[2] + w * dt)
    x[0] += v * math.cos(x[2]) * dt
    x[1] += v * math.sin(x[2]) * dt
    x[3], x[4] = v, w

    if check_collision(x):
        x = ctx["x"].copy()
        x[2] = norm_ang(x[2] + w * dt)
        x[3], x[4] = 0.0, w
        v = 0.0

    v_f, w_f = -v, -w
    wr = (2.0 * v_f + w_f * base) / (2.0 * r_w)
    wl = (2.0 * v_f - w_f * base) / (2.0 * r_w)
    wr = max(min(wr, 20.0), -20.0)
    wl = max(min(wl, 20.0), -20.0)

    sim.setJointTargetVelocity(ctx["m_r"], wr)
    sim.setJointTargetVelocity(ctx["m_l"], wl)
    sim.setObjectPosition(ctx["rb"], -1, [x[0], x[1], ctx["rb_z"]])

    offset = ctx.get("h_off", 0.0)
    flip = 0.0 if SINAL_FWD >= 0 else math.pi
    yaw_m = norm_ang(x[2] - offset - flip)

    sim.setObjectOrientation(ctx["rb"], -1, [ctx["rb_r"], ctx["rb_p"], yaw_m])
    try:
        sim.resetDynamicObject(ctx["rb"])
    except Exception:
        pass

    sim.step()
    return x


def fetch_static_obs():
    skip = [
        (float(ctx["x"][0]), float(ctx["x"][1]), dwa.robot_radius + 0.15),
        (float(ctx["goal_xy"][0]), float(ctx["goal_xy"][1]), dwa.robot_radius + 0.10),
    ]
    try:
        floor = get_obj("/Floor")
        pts, grid, inf = mo.construir_obstaculos_por_visao(sim, floor, excluir=skip)
        ctx["grid"], ctx["grid_inf"] = grid, inf
        if len(pts) >= 4:
            return pts
    except Exception:
        print("Usando fallback Bounding Box...")
    return get_static_obs()


def _thread_map(x0, x1, y0, y1, obs, path, k_pts, rx, ry, gx, gy):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Planejamento Estático - A*", fontsize=12, fontweight='bold')

    for ax, t in zip([ax1, ax2], ["Rota Completa", "Pontos-Chave"]):
        ax.set_title(t)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.axis("equal")
        ax.set_xlim(x0 - 0.1, x1 + 0.1)
        ax.set_ylim(y0 - 0.1, y1 + 0.1)
        if len(obs) > 0:
            ax.scatter(obs[:, 0], obs[:, 1], s=4, c="black", marker="s")
        ax.plot(rx, ry, "go", markersize=9)
        ax.plot(gx, gy, "ro", markersize=9)

    if len(path) > 0:
        ax1.plot(path[:, 0], path[:, 1], "b-", linewidth=2)
    if len(k_pts) > 0:
        ax2.plot(k_pts[:, 0], k_pts[:, 1], "y-x", markersize=8, linewidth=1.5)

    plt.tight_layout()
    plt.show()


def exibir_mapa_inicial():
    # Só tenta gerar o gráfico se as variáveis estáticas foram inicializadas
    if "st_obs" not in ctx:
        print("[Aviso] Não foi possível abrir o mapa pois os obstáculos não foram calculados.")
        return

    floor = get_obj("/Floor")
    f_pos = sim.getObjectPosition(floor, -1)
    x0 = f_pos[0] + sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_x)
    x1 = f_pos[0] + sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_x)
    y0 = f_pos[1] + sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_min_y)
    y1 = f_pos[1] + sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_y)

    print("\n---> ABRINDO O MAPA INICIAL <---")
    print("Analise as rotas geradas pelo A*. FECHE A JANELA DO GRÁFICO para iniciar a simulação no CoppeliaSim.\n")
    
    # Executa a função diretamente (sem thread) para pausar o código aqui
    _thread_map(
        x0, x1, y0, y1,
        np.array(ctx.get("st_obs", [])),
        np.array(ctx.get("g_path", [])),
        np.array(ctx.get("k_pts", [])),
        ctx["x"][0], ctx["x"][1],
        ctx["goal_xy"][0], ctx["goal_xy"][1]
    )


def init():
    ctx["m_r"] = get_obj("/MOTOR_DIREITO")
    ctx["m_l"] = get_obj("/MOTOR_ESQUERDO")
    ctx["rb"] = sim.getObjectParent(ctx["m_r"])
    ctx["gl"] = get_obj("/Alvo")

    ctx["ss"] = [
        get_obj("/SENSOR_MEIO"),
        get_obj("/SENSOR_DIAG_DIREITO"),
        get_obj("/SENSOR_DIAG_ESQUERDO"),
        get_obj("/SENSOR_DIREITO"),
        get_obj("/SENSOR_ESQUERDO"),
    ]

    r_pos = sim.getObjectPosition(ctx["rb"], -1)
    r_ori = sim.getObjectOrientation(ctx["rb"], -1)
    ctx["rb_z"], ctx["rb_r"], ctx["rb_p"] = r_pos[2], r_ori[0], r_ori[1]

    calc_h_offset()
    ctx["x"] = get_rb_state()

    g_pos = sim.getObjectPosition(ctx["gl"], -1)
    ctx["goal_xy"] = [g_pos[0], g_pos[1]]
    ctx["st_obs"] = fetch_static_obs()
    ctx["steps"] = 0

    # Calcula a rota global do A*
    plan_global()


def replan():
    plan_global()


def recovery_cmd(x, t):
    ang = math.atan2(t[1] - x[1], t[0] - x[0])
    turn = norm_ang(ang - x[2])
    w = max(min(1.2 * turn, dwa.max_yaw_rate), -dwa.max_yaw_rate)
    fase = ctx.get("_rec", 0)

    if fase > 22:
        return [0.07, 0.3 * w]
    if abs(turn) > 0.25:
        return [0.0, -0.8 if turn >= 0.0 else 0.8]
    return [-0.10, 0.5 * w]


def loop():
    if ctx["gl"] is not None:
        g_pos = sim.getObjectPosition(ctx["gl"], -1)
        ctx["goal_xy"] = [g_pos[0], g_pos[1]]

    x = ctx["x"]
    obs = get_all_obs(x)
    tgt = get_target(x)

    ref = ctx.get("_stk_ref")
    if ref is None or math.hypot(x[0] - ref[0], x[1] - ref[1]) > 0.05:
        ctx["_stk_ref"] = (float(x[0]), float(x[1]))
        ctx["_stk_steps"] = 0
    else:
        ctx["_stk_steps"] = ctx.get("_stk_steps", 0) + 1

    if ctx.get("_rec", 0) > 0:
        u = recovery_cmd(x, tgt)
        ctx["_rec"] -= 1
    else:
        u, _ = dwa.plan(x[0:3], x[3], x[4], tgt, obs)
        if ctx.get("_stk_steps", 0) >= 30:
            print("Preso -> replanejando...")
            replan()
            ctx["_rec"] = 35
            ctx["_stk_steps"] = 0

    ctx["x"] = move_robot(u)
    ctx["steps"] += 1

    if math.hypot(ctx["x"][0] - ctx["goal_xy"][0], ctx["x"][1] - ctx["goal_xy"][1]) <= 0.20:
        print("ALVO ATINGIDO!")
        return True
    return False


if __name__ == "__main__":
    print("Conectando ao CoppeliaSim...")
    
    # 1. Inicializa as variáveis e calcula a rota estática ANTES de dar Play
    init()
    
    # 2. Mostra o mapa na tela. O código vai travar aqui até você fechar a janela.
    exibir_mapa_inicial()

    # 3. Assim que você fechar a janela, a simulação REALMENTE começa no CoppeliaSim
    print("Iniciando a simulação física...")
    sim.setStepping(True)
    sim.startSimulation()

    try:
        while not loop():
            pass
    except KeyboardInterrupt:
        print("Interrompido pelo usuário.")
    finally:
        # Para os motores do robô
        if "m_r" in ctx and "m_l" in ctx:
            sim.setJointTargetVelocity(ctx["m_r"], 0.0)
            sim.setJointTargetVelocity(ctx["m_l"], 0.0)
        
        # Para a simulação física
        sim.stopSimulation()
        print("Simulação encerrada e finalizada.")