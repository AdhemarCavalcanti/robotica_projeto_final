import math
import numpy as np


# ==========================================================================
#  CONFIGURACAO
# ==========================================================================
class ConfigGrade:
    nome_sensor = "/camera_grade_ocupacao"
    res = 256
    altura = 3.0
    margem = 0.2
    alt_min_obs = 0.05
    flip_x = False
    flip_y = True


# ==========================================================================
#  CRIACAO / OBTENCAO DO SENSOR DE VISAO
# ==========================================================================
def criar_ou_obter_sensor(sim, cfg, centro_xy, tam_cena):
    try:
        sensor = sim.getObject(cfg.nome_sensor)
        _configurar_sensor(sim, sensor, cfg, centro_xy, tam_cena)
        return sensor
    except Exception:
        pass

    int_params = [cfg.res, cfg.res, 0, 0]
    float_params = [0.01, cfg.altura + 1.0, tam_cena, 0.05, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
    sensor = sim.createVisionSensor(1, int_params, float_params)
    sim.setObjectAlias(sensor, cfg.nome_sensor.strip("/"))
    _configurar_sensor(sim, sensor, cfg, centro_xy, tam_cena)
    return sensor


def _configurar_sensor(sim, sensor, cfg, centro_xy, tam_cena):
    cx, cy = centro_xy
    try:
        sim.setObjectInt32Param(sensor, sim.visionintparam_perspective_operation, 0)
        sim.setObjectInt32Param(sensor, sim.visionintparam_resolution_x, cfg.res)
        sim.setObjectInt32Param(sensor, sim.visionintparam_resolution_y, cfg.res)
        sim.setObjectFloatParam(sensor, sim.visionfloatparam_ortho_size, tam_cena)
        sim.setObjectFloatParam(sensor, sim.visionfloatparam_near_clipping, 0.01)
        sim.setObjectFloatParam(sensor, sim.visionfloatparam_far_clipping, cfg.altura + 1.0)
    except Exception as exc:
        print("[mapa_ocupacao] Erro nos parametros do sensor:", exc)

    sim.setObjectPosition(sensor, -1, [cx, cy, cfg.altura])
    sim.setObjectOrientation(sensor, -1, [math.pi, 0.0, 0.0])


# ==========================================================================
#  EXTENSAO DO CHAO
# ==========================================================================
def extensao_do_chao(sim, floor_handle):
    pos = sim.getObjectPosition(floor_handle, -1)
    min_x = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_min_x)
    max_x = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_max_x)
    min_y = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_min_y)
    max_y = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_max_y)

    centro = (pos[0], pos[1])
    tamanho = max(max_x - min_x, max_y - min_y)
    return centro, tamanho


# ==========================================================================
#  CAPTURA DA GRADE DE OCUPACAO
# ==========================================================================
def capturar_grade(sim, sensor, cfg):
    sim.handleVisionSensor(sensor)
    near = 0.01
    far = cfg.altura + 1.0

    buf, res = sim.getVisionSensorDepth(sensor, 1)
    depth = np.array(sim.unpackFloatTable(buf), dtype=float)
    nx, ny = int(res[0]), int(res[1])
    depth = depth.reshape((ny, nx))

    dist_m = near + depth * (far - near)
    limiar = cfg.altura - cfg.alt_min_obs
    return (dist_m < limiar), (nx, ny)


def capturar_grade_por_intensidade(sim, sensor, cfg, limiar=0.5, obs_escuro=True):
    sim.handleVisionSensor(sensor)
    img, res = sim.getVisionSensorImg(sensor)
    nx, ny = int(res[0]), int(res[1])
    arr = np.frombuffer(img, dtype=np.uint8).reshape((ny, nx, 3)).astype(float) / 255.0
    cinza = arr.mean(axis=2)
    if obs_escuro:
        return cinza < limiar, (nx, ny)
    return cinza > limiar, (nx, ny)


# ==========================================================================
#  CONVERSAO GRADE -> PONTOS NO MUNDO
# ==========================================================================
def grade_para_pontos(grade, cfg, centro_xy, tam_cena, excluir=None):
    ny, nx = grade.shape
    cx, cy = centro_xy
    passo = tam_cena / nx

    pontos = []
    ocupadas = np.argwhere(grade)
    for r_idx, c_idx in ocupadas:
        c = (nx - 1 - c_idx) if cfg.flip_x else c_idx
        r = (ny - 1 - r_idx) if cfg.flip_y else r_idx
        wx = cx + (c + 0.5 - nx / 2.0) * passo
        wy = cy + (r + 0.5 - ny / 2.0) * passo
        pontos.append([wx, wy])

    pontos = np.array(pontos, dtype=float) if pontos else np.empty((0, 2), dtype=float)

    if excluir and len(pontos) > 0:
        for ex, ey, er in excluir:
            d = np.hypot(pontos[:, 0] - ex, pontos[:, 1] - ey)
            pontos = pontos[d > er]

    return pontos


# ==========================================================================
#  API DE ALTO NIVEL
# ==========================================================================
def construir_obstaculos_por_visao(sim, floor_handle, cfg=None, excluir=None):
    if cfg is None:
        cfg = ConfigGrade()

    centro, tamanho = extensao_do_chao(sim, floor_handle)
    tamanho += cfg.margem
    sensor = criar_ou_obter_sensor(sim, cfg, centro, tamanho)

    try:
        grade, _ = capturar_grade(sim, sensor, cfg)
    except Exception as exc:
        print("[mapa_ocupacao] Profundidade falhou (", exc, "); usando intensidade.")
        grade, _ = capturar_grade_por_intensidade(sim, sensor, cfg)

    pontos = grade_para_pontos(grade, cfg, centro, tamanho, excluir=excluir)
    info = {
        "centro": centro,
        "tamanho": tamanho,
        "passo": tamanho / grade.shape[1],
        "sensor": sensor,
    }
    return pontos, grade, info


# ==========================================================================
#  EXECUCAO ISOLADA (DEPURACAO)
# ==========================================================================
def _ascii_preview(grade, cols=60):
    ny, nx = grade.shape
    passo = max(1, nx // cols)
    linhas = []
    for r in range(0, ny, passo):
        linha = "".join("#" if grade[r, c] else "." for c in range(0, nx, passo))
        linhas.append(linha)
    print("\n".join(linhas))


if __name__ == "__main__":
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    print("Conectando...")
    client = RemoteAPIClient()
    sim = client.require("sim")
    cfg = ConfigGrade()

    sim.setStepping(True)
    sim.startSimulation()
    try:
        floor = sim.getObject("/Floor")
        pontos, grade, info = construir_obstaculos_por_visao(sim, floor, cfg)

        ocupadas = int(np.count_nonzero(grade))
        total = grade.size
        print(f"Res grade   : {grade.shape[1]} x {grade.shape[0]}")
        print(f"Centro      : {info['centro']}")
        print(f"Tam coberto : {info['tamanho']:.2f} m")
        print(f"Celula      : {info['passo']*100:.1f} cm")
        print(f"Ocupadas    : {ocupadas}/{total} ({100.0*ocupadas/total:.1f}%)")
        print(f"Pontos      : {len(pontos)}")
        print("\nPrevia ASCII:")
        _ascii_preview(grade)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 6))
            plt.imshow(grade, cmap="Greys", origin="lower")
            plt.title("Grade de Ocupacao")
            plt.tight_layout()
            plt.savefig("grade_ocupacao.png", dpi=120)
            print("\nSalvo em grade_ocupacao.png")
        except Exception as exc:
            print("(matplotlib indisponivel:", exc, ")")

    finally:
        sim.stopSimulation()
        print("Finalizado.")