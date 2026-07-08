import math
import numpy as np


# ==========================================================================
#  CONFIGURAÇÃO DA GRADE DE OCUPAÇÃO
# ==========================================================================
class GridMapConfig:
    sensor_name = "/camera_grade_ocupacao"
    resolution = 256
    sensor_height = 4.0
    boundary_margin = 0.2
    min_obstacle_height = 0.05
    flip_x = False
    flip_y = True


# ==========================================================================
#  GERENCIAMENTO E INSTANCIAÇÃO DO SENSOR DE VISÃO
# ==========================================================================
def create_or_acquire_sensor(sim, config, center_xy, scene_size):
    """Localiza o sensor de visão na cena ou instancia um dinamicamente se ausente."""
    try:
        sensor = sim.getObject(config.sensor_name)
        _configure_vision_sensor(sim, sensor, config, center_xy, scene_size)
        return sensor
    except Exception:
        pass

    int_parameters = [config.resolution, config.resolution, 0, 0]
    float_parameters = [
        0.01, 
        config.sensor_height + 1.0, 
        scene_size, 
        0.05, 0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    
    sensor = sim.createVisionSensor(1, int_parameters, float_parameters)
    sim.setObjectAlias(sensor, config.sensor_name.strip("/"))
    _configure_vision_sensor(sim, sensor, config, center_xy, scene_size)
    return sensor


def _configure_vision_sensor(sim, sensor, config, center_xy, scene_size):
    """Aplica propriedades ortográficas e coordenadas de orientação ao sensor."""
    cx, cy = center_xy
    try:
        sim.setObjectInt32Param(sensor, sim.visionintparam_perspective_operation, 0)
        sim.setObjectInt32Param(sensor, sim.visionintparam_resolution_x, config.resolution)
        sim.setObjectInt32Param(sensor, sim.visionintparam_resolution_y, config.resolution)
        sim.setObjectFloatParam(sensor, sim.visionfloatparam_ortho_size, scene_size)
        sim.setObjectFloatParam(sensor, sim.visionfloatparam_near_clipping, 0.01)
        sim.setObjectFloatParam(sensor, sim.visionfloatparam_far_clipping, config.sensor_height + 1.0)
    except Exception as err:
        print(f"[mapa_ocupacao] Erro ao injetar parâmetros no sensor: {err}")

    sim.setObjectPosition(sensor, -1, [cx, cy, config.sensor_height])
    sim.setObjectOrientation(sensor, -1, [math.pi, 0.0, 0.0])


# ==========================================================================
#  EXTRAÇÃO DE DIMENSÕES DA SCENE (FLOOR)
# ==========================================================================
def extract_floor_bounds(sim, floor_handle):
    """Calcula o centro e a maior dimensão do plano delimitador do chão."""
    position = sim.getObjectPosition(floor_handle, -1)
    min_x = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_min_x)
    max_x = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_max_x)
    min_y = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_min_y)
    max_y = sim.getObjectFloatParam(floor_handle, sim.objfloatparam_objbbox_max_y)

    center = (position[0], position[1])
    max_span = max(max_x - min_x, max_y - min_y)
    return center, max_span


# ==========================================================================
#  CAPTURA E PROCESSAMENTO DA MATRIZ DE VISÃO
# ==========================================================================
def capture_occupancy_grid_by_depth(sim, sensor, config):
    """Analisa o buffer de profundidade e segmenta objetos acima da cota do chão."""
    sim.handleVisionSensor(sensor)
    near_clip = 0.01
    far_clip = config.sensor_height + 1.0

    raw_buffer, resolution = sim.getVisionSensorDepth(sensor, 1)
    depth_data = np.array(sim.unpackFloatTable(raw_buffer), dtype=float)
    nx, ny = int(resolution[0]), int(resolution[1])
    depth_matrix = depth_data.reshape((ny, nx))

    # Converte o buffer normalizado para distância linear em metros
    metric_distances = near_clip + depth_matrix * (far_clip - near_clip)
    obstacle_threshold = config.sensor_height - config.min_obstacle_height
    
    return (metric_distances < obstacle_threshold), (nx, ny)


def capture_occupancy_grid_by_intensity(sim, sensor, config, threshold=0.5, dark_is_obstacle=True):
    """Alternativa de segmentação baseada na intensidade de cor (RGB para Escala de Cinza)."""
    sim.handleVisionSensor(sensor)
    raw_img, resolution = sim.getVisionSensorImg(sensor)
    nx, ny = int(resolution[0]), int(resolution[1])
    
    rgb_array = np.frombuffer(raw_img, dtype=np.uint8).reshape((ny, nx, 3)).astype(float) / 255.0
    grayscale_matrix = rgb_array.mean(axis=2)
    
    if dark_is_obstacle:
        return grayscale_matrix < threshold, (nx, ny)
    return grayscale_matrix > threshold, (nx, ny)


# ==========================================================================
#  MAPEAMENTO VETORIZADO: GRADE -> MUNDO REAL (NUMPY OPERATORS)
# ==========================================================================
def convert_grid_to_world_points(grid, config, center_xy, scene_size, exclusion_zones=None):
    """Transforma índices booleanos da matriz em coordenadas métricas mundiais (2D)."""
    ny, nx = grid.shape
    cx, cy = center_xy
    pixel_step = scene_size / nx

    # Extrai os índices das células ocupadas
    occupied_indices = np.argwhere(grid)
    if occupied_indices.size == 0:
        return np.empty((0, 2), dtype=float)

    # Vetorização completa: Substitui o loop 'for' por álgebra matricial sequencial
    row_indices = occupied_indices[:, 0]
    col_indices = occupied_indices[:, 1]

    if config.flip_x:
        col_indices = (nx - 1) - col_indices
    if config.flip_y:
        row_indices = (ny - 1) - row_indices

    # Mapeamento linear direto do espaço discretizado para o espaço contínuo métrico
    world_x = cx + (col_indices + 0.5 - nx / 2.0) * pixel_step
    world_y = cy + (row_indices + 0.5 - ny / 2.0) * pixel_step
    world_points = np.column_stack((world_x, world_y))

    # Aplica zonas de filtragem circular (ex: ao redor do próprio robô e do alvo)
    if exclusion_zones and len(world_points) > 0:
        for ex, ey, radius in exclusion_zones:
            distances = np.hypot(world_points[:, 0] - ex, world_points[:, 1] - ey)
            world_points = world_points[distances > radius]

    return world_points


# ==========================================================================
#  FACHADA PRINCIPAL (HIGH-LEVEL API)
# ==========================================================================
def construir_obstaculos_por_visao(sim, floor_handle, cfg=None, excluir=None):
    """Ponto de entrada unificado para processar e gerar a nuvem de obstáculos estáticos."""
    config = cfg if cfg is not None else GridMapConfig()

    center, size = extract_floor_bounds(sim, floor_handle)
    size += config.boundary_margin
    sensor = create_or_acquire_sensor(sim, config, center, size)

    try:
        grid, _ = capture_occupancy_grid_by_depth(sim, sensor, config)
    except Exception as err:
        print(f"[mapa_ocupacao] Buffer de profundidade falhou ({err}); chaveando para intensidade de imagem.")
        grid, _ = capture_occupancy_grid_by_intensity(sim, sensor, config)

    world_obstacles = convert_grid_to_world_points(grid, config, center, size, exclusion_zones=excluir)
    
    metadata = {
        "centro": center,
        "tamanho": size,
        "passo": size / grid.shape[1],
        "sensor": sensor,
    }
    return world_obstacles, grid, metadata


# ==========================================================================
#  SUBSISTEMA DE DEPURAÇÃO E PRÉ-VISUALIZAÇÃO TERMINAL
# ==========================================================================
def _display_ascii_preview(grid, columns=60):
    """Renderiza uma versão compacta em texto da matriz de ocupação diretamente no console."""
    ny, nx = grid.shape
    step = max(1, nx // columns)
    preview_rows = []
    
    for r in range(0, ny, step):
        row_string = "".join("#" if grid[r, c] else "." for c in range(0, nx, step))
        preview_rows.append(row_string)
        
    print("\n".join(preview_rows))


if __name__ == "__main__":
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    print("Estabelecendo conexão remota...")
    client = RemoteAPIClient()
    sim = client.require("sim")
    app_config = GridMapConfig()

    sim.setStepping(True)
    sim.startSimulation()
    
    try:
        floor_obj = sim.getObject("/Floor")
        obstacles, occupancy_grid, info = construir_obstaculos_por_visao(sim, floor_obj, app_config)

        total_cells = occupancy_grid.size
        filled_cells = int(np.count_nonzero(occupancy_grid))
        
        print(f"Resolução da Grade: {occupancy_grid.shape[1]} x {occupancy_grid.shape[0]}")
        print(f"Coordenada Centro : {info['centro']}")
        print(f"Área de Cobertura : {info['tamanho']:.2f} m")
        print(f"Dimensão Célula   : {info['passo']*100:.1f} cm")
        print(f"Densidade Ocupada : {filled_cells}/{total_cells} ({100.0 * filled_cells / total_cells:.1f}%)")
        print(f"Total de Pontos 2D: {len(obstacles)}")
        
        print("\nPrévia Gráfica em ASCII:")
        _display_ascii_preview(occupancy_grid)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 6))
            plt.imshow(occupancy_grid, cmap="Greys", origin="lower")
            plt.title("Grade de Ocupacao - Output")
            plt.tight_layout()
            plt.savefig("grade_ocupacao.png", dpi=120)
            print("\nMapa de depuração exportado com sucesso para 'grade_ocupacao.png'")
        except Exception as plot_err:
            print(f"\n[Aviso] Matplotlib indisponível para renderização em disco: {plot_err}")

    finally:
        sim.stopSimulation()
        print("Sessão finalizada e desligada com segurança.")