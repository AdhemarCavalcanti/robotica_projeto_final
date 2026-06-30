import sys
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import time

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient # type: ignore
except ImportError:
    print("\n[ERRO]: Não foi possível encontrar 'coppeliasim_zmqremoteapi_client'.")
    sys.exit(1)

class ImprovedAStarPlanner:
    def __init__(self, resolution=256, world_size=5.0):
        self.resolution = resolution
        self.world_size = world_size
        self.scale = resolution / world_size
        self.grid_map = np.zeros((resolution, resolution), dtype=np.uint8)

    def world_to_grid(self, pos):
        offset = self.world_size / 2.0
        col = int((pos[0] + offset) * self.scale)
        lin = int((pos[1] + offset) * self.scale)
        
        col = (self.resolution - 1) - col
        lin = (self.resolution - 1) - lin
        
        return max(0, min(self.resolution - 1, col)), max(0, min(self.resolution - 1, lin))

    def find_nearest_free_node(self, node):
        x, y = node
        if self.grid_map[y, x] != 255:
            return node
            
        queue = [(x, y)]
        visited = {(x, y)}
        
        while queue:
            cx, cy = queue.pop(0)
            if self.grid_map[cy, cx] != 255:
                return (cx, cy)
                
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.resolution and 0 <= ny < self.resolution:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return node

    def _euclidean_distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def _get_neighbors(self, node):
        x, y = node
        neighbors = []
        directions = [
            (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))
        ]
        for dx, dy, cost in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.resolution and 0 <= ny < self.resolution:
                if self.grid_map[ny, nx] != 255: 
                    neighbors.append(((nx, ny), cost))
        return neighbors

    def has_line_of_sight(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        cx, cy = x1, y1
        while (cx, cy) != (x2, y2):
            if self.grid_map[cy, cx] == 255:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy
        return self.grid_map[y2, x2] != 255

    def find_path(self, start, goal):
        D = self._euclidean_distance(start, goal)
        open_list = []
        heapq.heappush(open_list, (0.0, 0, start))
        
        g_score = {start: 0.0}
        came_from = {start: None}
        closed_set = set()
        counter = 0

        print(f"-> [A*] Executando busca de {start} até {goal}...")
        while open_list:
            _, _, current = heapq.heappop(open_list)
            
            if current == goal:
                print("-> [A*] Alvo alcançado! Gerando caminho bruto...")
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            closed_set.add(current)

            for neighbor, move_cost in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    
                    d = self._euclidean_distance(neighbor, goal)
                    omega = 1.0 + (d / D) if D > 0 else 1.0
                    f = tentative_g + (omega * d)
                    
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_list, (f, counter, neighbor))
                    
        raise RuntimeError("Não foi possível encontrar um caminho válido.")

    def remove_redundant_nodes(self, raw_path):
        if len(raw_path) <= 2:
            return raw_path
            
        colinear = [raw_path[0]]
        for i in range(1, len(raw_path) - 1):
            x1, y1 = raw_path[i-1]
            x2, y2 = raw_path[i]
            x3, y3 = raw_path[i+1]
            if (y2 - y1) * (x3 - x2) != (y3 - y2) * (x2 - x1):
                colinear.append(raw_path[i])
        colinear.append(raw_path[-1])

        key_points = [colinear[0]]
        curr = 0
        while curr < len(colinear) - 1:
            next_p = curr + 1
            for j in range(len(colinear) - 1, curr, -1):
                if self.has_line_of_sight(colinear[curr], colinear[j]):
                    next_p = j
                    break
            key_points.append(colinear[next_p])
            curr = next_p
        return key_points

    def bessel_smoothing(self, key_points, num_points=15):
        if len(key_points) < 2:
            return key_points
            
        smoothed = []
        for idx in range(len(key_points) - 1):
            p_start = np.array(key_points[idx])
            p_end = np.array(key_points[idx+1])
            
            for step in range(num_points):
                t = step / (num_points - 1)
                p_t = (1 - t) * p_start + t * p_end
                
                if step == 0 and len(smoothed) > 0:
                    continue
                smoothed.append((float(p_t[0]), float(p_t[1])))
                
        return smoothed


def main():
    print("Conectando ao CoppeliaSim via API Remota...")
    try:
        client = RemoteAPIClient()
        sim = client.getObject('sim')
        
        sensor_handle = sim.getObject('/SENSOR_MAPEAMENTO') 
        cuboid_handle = sim.getObject('/Cuboid')
        alvo_handle = sim.getObject('/Alvo')

        print("\n--- CAPTURANDO SCENE ENVIRONMENT MAP ---")
        planner = ImprovedAStarPlanner(resolution=256, world_size=5.0)
        
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.05)
            
        client.setStepping(True)
        sim.startSimulation()
        
        for _ in range(15):
            client.step()
            time.sleep(0.02)

        retorno_bruto = sim.getVisionSensorDepthBuffer(sensor_handle)
        pos_robo = sim.getObjectPosition(cuboid_handle, -1)
        pos_alvo = sim.getObjectPosition(alvo_handle, -1)
        
        client.setStepping(False)
        sim.stopSimulation()

    except Exception as e:
        print(f"[ERRO DE CONEXÃO/API]: {e}")
        sys.exit(1)

    # --- PROCESSAMENTO DO BUFFER ---
    try:
        depth_buffer = None
        res_x, res_y = 256, 256 

        if isinstance(retorno_bruto, (list, tuple)):
            if len(retorno_bruto) > 0 and isinstance(retorno_bruto[0], (int, float)):
                depth_buffer = retorno_bruto
            else:
                for item in retorno_bruto:
                    if isinstance(item, dict):
                        depth_buffer = item.get('buffer', item.get('image', None))
                        res_x, res_y = item.get('resolution', [256, 256])
                        break
                    elif isinstance(item, (bytes, bytearray, list)):
                        depth_buffer = item
        elif isinstance(retorno_bruto, dict):
            depth_buffer = retorno_bruto.get('buffer', retorno_bruto.get('image', None))
            res_x, res_y = retorno_bruto.get('resolution', [256, 256])
        else:
            depth_buffer = retorno_bruto

        if depth_buffer is None:
            print("[ERRO]: Buffer de imagem nulo.")
            return

        depth_data = np.array(depth_buffer, dtype=np.float32).flatten()
        if depth_data.size != (res_x * res_y):
            lado = int(np.sqrt(depth_data.size))
            if lado * lado == depth_data.size: res_x, res_y = lado, lado

        depth_matrix = depth_data.reshape(res_y, res_x)
        
        depth_matrix = depth_matrix.T
        depth_matrix = np.flipud(depth_matrix)
        depth_matrix = np.fliplr(depth_matrix)
        
        valor_chao = np.max(depth_matrix)
        planner.grid_map = np.where(depth_matrix < (valor_chao - 0.015), 255, 0).astype(np.uint8)
        print(f"-> Mapa alinhado com sucesso. Resolução: {res_x}x{res_y}")

    except Exception as erro_mapa:
        print(f"[ERRO PROCESSAMENTO MAPA]: {erro_mapa}")
        return

    start_grid = planner.world_to_grid([pos_robo[0], pos_robo[1]])
    goal_grid = planner.world_to_grid([pos_alvo[0], pos_alvo[1]])

    start_grid = planner.find_nearest_free_node(start_grid)
    goal_grid = planner.find_nearest_free_node(goal_grid)

    planner.grid_map[start_grid[1], start_grid[0]] = 0
    planner.grid_map[goal_grid[1], goal_grid[0]] = 0

    # --- EXECUÇÃO E EXIBIÇÃO ---
    try:
        raw_path = planner.find_path(start_grid, goal_grid)
        key_points = planner.remove_redundant_nodes(raw_path)
        optimal_path = planner.bessel_smoothing(key_points)
        print("-> Fluxo A* concluído com sucesso. Gerando amostragem...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Análise Categórica - Comparativo de Algoritmos (Pontos do A* Tradicional)", fontsize=13, fontweight='bold')

        # CENA 1: Captura Limpa
        axes[0].imshow(planner.grid_map, cmap='gray_r', origin='lower')
        axes[0].plot(start_grid[0], start_grid[1], 'go', markersize=10, label='Start (Robô)')
        axes[0].plot(goal_grid[0], goal_grid[1], 'ro', markersize=10, label='Target (Alvo)')
        axes[0].set_title("1. Scene Environment Map (Capturado)")
        axes[0].legend()

        # CENA 2: A* Tradicional Renderizado por Pontos Discretos
        axes[1].imshow(planner.grid_map, cmap='gray_r', origin='lower')
        rx, ry = zip(*raw_path)
        # MODIFICAÇÃO AQUI: Trocado 'b-' por 'b.' para desenhar pontos isolados em vez de linha
        axes[1].plot(rx, ry, 'b.', markersize=4, label='Nós do A* Tradicional')
        axes[1].plot(start_grid[0], start_grid[1], 'go', markersize=8)
        axes[1].plot(goal_grid[0], goal_grid[1], 'ro', markersize=8)
        axes[1].set_title("2. A* Tradicional (Nós Explorados)")
        axes[1].legend()

        # CENA 3: A* Modificado Otimizado
        axes[2].imshow(planner.grid_map, cmap='gray_r', origin='lower')
        # MODIFICAÇÃO AQUI: Pano de fundo do A* tradicional também alterado para pontos ('b.')
        axes[2].plot(rx, ry, 'b.', markersize=2, alpha=0.5, label='A* Tradicional')
        
        # Plota os Key Points (X amarelos)
        kx, ky = zip(*key_points)
        axes[2].plot(kx, ky, 'yX', markersize=9, zorder=5, label='Key Points Assinalados')
        
        # Plota a Trajetória Otimizada (Linha contínua rosa cruzando os Key Points)
        ox, oy = zip(*optimal_path)
        axes[2].plot(ox, oy, 'm-', linewidth=3, zorder=4, label='Trajetória Modificada Corrigida')
        
        axes[2].plot(start_grid[0], start_grid[1], 'go', markersize=8)
        axes[2].plot(goal_grid[0], goal_grid[1], 'ro', markersize=8)
        axes[2].set_title("3. A* Melhorado (Vínculo aos Pontos-Chave)")
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    except Exception as erro_loop:
        print(f"\n[FALHA NO LOOP DE BUSCA]: {erro_loop}")

if __name__ == "__main__":
    main()