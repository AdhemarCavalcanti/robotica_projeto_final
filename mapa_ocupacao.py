import math
import numpy as np


class OccupancyGridGenerator:
    """Gerenciador orientado a objetos para criação de grades de ocupação

    e mapeamento de obstáculos usando sensores de visão no CoppeliaSim.
    """

    def __init__(self, sim, config=None):
        self.sim = sim
        # Configurações padrão encapsuladas
        self.config = config or {
            "sensor_name": "/camera_grade_ocupacao",
            "resolution": 256,
            "sensor_height": 4.0,
            "boundary_margin": 0.2,
            "min_obstacle_height": 0.05,
            "flip_x": False,
            "flip_y": True,
        }
        self.sensor_handle = None

    def _extract_floor_geometry(self, floor_handle):
        """Calcula o centro e a dimensão máxima delimitadora do chão."""
        pos = self.sim.getObjectPosition(floor_handle, -1)
        min_x = self.sim.getObjectFloatParam(
            floor_handle, self.sim.objfloatparam_objbbox_min_x
        )
        max_x = self.sim.getObjectFloatParam(
            floor_handle, self.sim.objfloatparam_objbbox_max_x
        )
        min_y = self.sim.getObjectFloatParam(
            floor_handle, self.sim.objfloatparam_objbbox_min_y
        )
        max_y = self.sim.getObjectFloatParam(
            floor_handle, self.sim.objfloatparam_objbbox_max_y
        )

        center = (pos[0], pos[1])
        span = max(max_x - min_x, max_y - min_y)
        return center, span

    def _init_sensor(self, center_xy, scene_size):
        """Garante a existência e a configuração correta do Vision Sensor."""
        name = self.config["sensor_name"]
        res = self.config["resolution"]
        z_pos = self.config["sensor_height"]
        far_clip = z_pos + 1.0

        # Tenta obter sensor existente, senão cria um novo
        try:
            self.sensor_handle = self.sim.getObject(name)
        except Exception:
            int_params = [res, res, 0, 0]
            float_params = [
                0.01,
                far_clip,
                scene_size,
                0.05,
                0.05,
                0.02,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            self.sensor_handle = self.sim.createVisionSensor(
                1, int_params, float_params
            )
            self.sim.setObjectAlias(self.sensor_handle, name.strip("/"))

        # Aplica parâmetros ortográficos e transformação espacial
        try:
            self.sim.setObjectInt32Param(
                self.sensor_handle,
                self.sim.visionintparam_perspective_operation,
                0,
            )
            self.sim.setObjectInt32Param(
                self.sensor_handle, self.sim.visionintparam_resolution_x, res
            )
            self.sim.setObjectInt32Param(
                self.sensor_handle, self.sim.visionintparam_resolution_y, res
            )
            self.sim.setObjectFloatParam(
                self.sensor_handle, self.sim.visionfloatparam_ortho_size, scene_size
            )
            self.sim.setObjectFloatParam(
                self.sensor_handle, self.sim.visionfloatparam_near_clipping, 0.01
            )
            self.sim.setObjectFloatParam(
                self.sensor_handle, self.sim.visionfloatparam_far_clipping, far_clip
            )
        except Exception as e:
            print(f"[OccupancyGrid] Falha ao configurar propriedades: {e}")

        self.sim.setObjectPosition(
            self.sensor_handle, -1, [center_xy[0], center_xy[1], z_pos]
        )
        self.sim.setObjectOrientation(self.sensor_handle, -1, [math.pi, 0.0, 0.0])

    def _get_occupancy_matrix(self):
        """Captura os dados do sensor priorizando profundidade, com fallback para intensidade."""
        self.sim.handleVisionSensor(self.sensor_handle)
        res = self.config["resolution"]

        try:
            # Abordagem 1: Buffer de Profundidade
            raw_depth, _ = self.sim.getVisionSensorDepth(self.sensor_handle, 1)
            depth_data = np.array(
                self.sim.unpackFloatTable(raw_depth), dtype=float
            ).reshape((res, res))

            near = 0.01
            far = self.config["sensor_height"] + 1.0
            metric_depth = near + depth_data * (far - near)

            threshold = self.config["sensor_height"] - self.config["min_obstacle_height"]
            return metric_depth < threshold

        except Exception as err:
            print(
                f"[OccupancyGrid] Erro no buffer de profundidade ({err}). Mudando para Intensidade RGB..."
            )
            # Abordagem 2: Fallback via Imagem RGB (Grayscale)
            raw_img, _ = self.sim.getVisionSensorImg(self.sensor_handle)
            rgb = (
                np.frombuffer(raw_img, dtype=np.uint8)
                .reshape((res, res, 3))
                .astype(float)
                / 255.0
            )
            grayscale = rgb.mean(axis=2)
            return grayscale < 0.5  # Considera escuro como obstáculo

    def _map_grid_to_world(self, grid, center_xy, scene_size, exclusion_zones):
        """Abordagem via Meshgrid: Converte a matriz booleana para coordenadas reais."""
        res = self.config["resolution"]
        cx, cy = center_xy
        pixel_step = scene_size / res

        # Gerar vetores coordenados lineares para os eixos
        col_indices = np.arange(res)
        row_indices = np.arange(res)

        if self.config["flip_x"]:
            col_indices = (res - 1) - col_indices
        if self.config["flip_y"]:
            row_indices = (res - 1) - row_indices

        # Mapeamento linear dos vetores coordenados
        x_coords = cx + (col_indices + 0.5 - res / 2.0) * pixel_step
        y_coords = cy + (row_indices + 0.5 - res / 2.0) * pixel_step

        # Construir matriz de coordenadas (Meshgrid)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Filtrar as coordenadas onde o grid indica obstáculo (Máscara booleana direta)
        world_points = np.column_stack((X[grid], Y[grid]))

        # Aplicar zonas de exclusão cilíndricas/circulares
        if exclusion_zones and len(world_points) > 0:
            for ex, ey, radius in exclusion_zones:
                distances = np.hypot(world_points[:, 0] - ex, world_points[:, 1] - ey)
                world_points = world_points[distances > radius]

        return world_points

    def generate(self, floor_handle, exclusion_zones=None):
        """Fluxo principal para geração e retorno dos dados da grade de ocupação."""
        center, size = self._extract_floor_geometry(floor_handle)
        size += self.config["boundary_margin"]

        self._init_sensor(center, size)
        grid = self._get_occupancy_matrix()
        points = self._map_grid_to_world(grid, center, size, exclusion_zones)

        metadata = {
            "centro": center,
            "tamanho": size,
            "passo": size / res if (res := self.config["resolution"]) else 0,
            "sensor": self.sensor_handle,
        }
        return points, grid, metadata


# ==========================================================================
#  PRE-VISUALIZAÇÃO TERMINAL (STRING COMPACTA)
# ==========================================================================
def render_ascii_art(grid, max_width=60):
    res = grid.shape[0]
    step = max(1, res // max_width)
    sub_grid = grid[::step, ::step]
    print("\n".join("".join("#" if cell else "." for cell in row) for row in sub_grid))


# ==========================================================================
#  BLOCO DE EXECUÇÃO PRINCIPAL
# ==========================================================================
if __name__ == "__main__":
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    print("Conectando ao CoppeliaSim via ZMQ Remote API...")
    client = RemoteAPIClient()
    sim = client.require("sim")

    sim.setStepping(True)
    sim.startSimulation()

    try:
        floor_obj = sim.getObject("/Floor")

        # Instanciação da nova arquitetura
        generator = OccupancyGridGenerator(sim)
        obstacles, occupancy_grid, info = generator.generate(floor_obj)

        # Estatísticas e Métricas
        filled = int(np.sum(occupancy_grid))
        total = occupancy_grid.size

        print(f"Resolução da Grade: {occupancy_grid.shape[1]} x {occupancy_grid.shape[0]}")
        print(f"Coordenada Centro : {info['centro']}")
        print(f"Área de Cobertura : {info['tamanho']:.2f} m")
        print(f"Dimensão Célula   : {info['passo']*100:.1f} cm")
        print(f"Densidade Ocupada : {filled}/{total} ({100.0 * filled / total:.1f}%)")
        print(f"Total de Pontos 2D: {len(obstacles)}")

        print("\nVisualização ASCII:")
        render_ascii_art(occupancy_grid)

        # Exportação gráfica opcional
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 6))
            plt.imshow(occupancy_grid, cmap="Greys", origin="lower")
            plt.title("Grade de Ocupacao - Nova Abordagem")
            plt.tight_layout()
            plt.savefig("grade_ocupacao.png", dpi=120)
            print("\nMapa salvo com sucesso em 'grade_ocupacao.png'")
        except Exception as e:
            print(f"\n[Aviso] Pulando exportação do Matplotlib: {e}")

    finally:
        sim.stopSimulation()
        print("Simulação encerrada com sucesso.")