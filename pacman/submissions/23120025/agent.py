"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
"""

import sys
from pathlib import Path
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random
import itertools
from heapq import heappush, heappop

class PacmanAgent(BasePacmanAgent):
    """
    Pacman Agent using Minimax with heat map heuristic
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_positions = []
        self.counter = itertools.count()
    
    # -------------------------------
    # Utilities
    # -------------------------------
    def _is_valid_position(self, pos, map_state):
        row, col = pos
        h, w = map_state.shape
        return 0 <= row < h and 0 <= col < w and map_state[row, col] == 0

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0]+dr, pos[1]+dc)

    def _get_neighbors(self, pos, map_state):
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])
    
    # -------------------------------
    # Predictive Heat Map
    # -------------------------------
    def predictive_heat_map(self, map_state, ghost_pos, depth_limit=5, decay=0.8):
        heat_map = np.zeros_like(map_state, dtype=float)
        height, width = map_state.shape

        # Compute escape routes for all cells
        escape_routes = np.zeros_like(map_state, dtype=int)
        for r in range(height):
            for c in range(width):
                if map_state[r,c]==0:
                    escape_routes[r,c] = self.count_escape_routes((r,c), map_state, max_depth=3)

        # BFS from ghost position
        queue = deque([(ghost_pos, 0)])
        visited = set([ghost_pos])
        while queue:
            pos, step = queue.popleft()
            if step > depth_limit:
                continue

            neighbors = [n for n, _ in self._get_neighbors(pos, map_state)]
            if not neighbors:
                continue
            max_escape = max([escape_routes[n] for n in neighbors])
            best_neighbors = [n for n in neighbors if escape_routes[n] == max_escape]

            for n in best_neighbors:
                heat_map[n] += decay ** step
                if n not in visited:
                    visited.add(n)
                    queue.append((n, step+1))
        return heat_map

    def count_escape_routes(self, pos, map_state, max_depth=3):
        visited = set([pos])
        queue = deque([(pos, 0)])
        count = 0
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for n, _ in self._get_neighbors(current, map_state):
                count += 1
                if n not in visited:
                    visited.add(n)
                    queue.append((n, depth+1))
        return count

    # -------------------------------
    # Minimax with heat map heuristic
    # -------------------------------
    def minimax(self, map_state, pacman_pos, ghost_pos, depth, is_pacman_turn,
                alpha=-float('inf'), beta=float('inf')):
        
        # Leaf node heuristic: use predictive heat map
        if depth == 0 or pacman_pos == ghost_pos:
            heat_map = self.predictive_heat_map(map_state, ghost_pos, depth_limit=3)
            score = heat_map[pacman_pos] - self._manhattan_distance(pacman_pos, ghost_pos)
            return score, None

        if is_pacman_turn:
            max_eval = -float('inf')
            best_move = Move.STAY
            for next_pos, move in self._get_neighbors(pacman_pos, map_state):
                eval_score, _ = self.minimax(map_state, next_pos, ghost_pos,
                                             depth-1, False, alpha, beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = Move.STAY
            for next_pos, move in self._get_neighbors(ghost_pos, map_state):
                eval_score, _ = self.minimax(map_state, pacman_pos, next_pos,
                                             depth-1, True, alpha, beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    # -------------------------------
    # Main step
    # -------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        self.previous_positions.append(my_position)
        if len(self.previous_positions) > 10:
            self.previous_positions.pop(0)

        distance = self._manhattan_distance(my_position, enemy_position)
        # Far away: use A* (optional)
        if distance > 6:
            path = self.astar(my_position, enemy_position, map_state)
            if path and path[0] != Move.STAY:
                next_pos = self._apply_move(my_position, path[0])
                if self.previous_positions.count(next_pos) < 3:
                    return path[0]

        # Close: use Minimax with heat map heuristic
        _, best_move = self.minimax(map_state, my_position, enemy_position,
                                    depth=3, is_pacman_turn=True)
        return best_move if best_move else Move.STAY

    # -------------------------------
    # Optional: A* as fallback
    # -------------------------------
    def astar(self, start, goal, map_state):
        def heuristic(pos):
            return self._manhattan_distance(pos, goal)
        frontier = [(0, next(self.counter), start, [])]
        visited = set()
        while frontier:
            f, _, current, path = heappop(frontier)
            if current == goal:
                return path
            if current in visited:
                continue
            visited.add(current)
            for next_pos, move in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    new_path = path + [move]
                    heappush(frontier, (len(new_path) + heuristic(next_pos), next(self.counter), next_pos, new_path))
        return [Move.STAY]


from collections import deque # Vẫn cần cho Minimax (nếu bạn muốn tối ưu)


class GhostAgent(BaseGhostAgent):
    """
    Ghost Agent v3.7 (Final) - 100% Minimax "Thuần túy"
    
    CHIẾN LƯỢC:
    - Vứt bỏ hoàn toàn logic Hybrid (if xa/gần). Đây là lỗ hổng
      bị Pacman "Interceptor" lợi dụng để "lùa" (herd).
    - Chiến lược duy nhất: 100% MINIMAX mọi lúc.
    
    "BỘ NÃO" (HEURISTIC):
    - Dùng heuristic "thuần túy" của bản vSimple gốc:
      Mục tiêu duy nhất là "TỐI ĐA HÓA KHOẢNG CÁCH".
    - Minimax "nhìn" 6-8 bước sẽ tự động "thấy" ngõ cụt
      là nước đi "thua" (vì distance sẽ về 0).
      
    "BIẾT LUẬT":
    - Minimax đã được nâng cấp để hiểu:
        1. self.pacman_speed
        2. self.max_steps (để "câu giờ")
        3. self.capture_threshold
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Smart Ghost v3.7 (Pure Minimax)"

        # === ‼️ CÀI ĐẶT LUẬT CHƠI (BẮT BUỘC PHẢI KHỚP VỚI ARENA.PY) ===
        # NẾU BẠN THUA, 90% LÀ DO SAI SÓT Ở ĐÂY.
        
        self.pacman_speed = 1      # Sửa thành 2 nếu Pacman chạy 2 bước
        self.max_steps = 200       # Sửa nếu bạn chạy --max-steps
        self.capture_threshold = 2 # Sửa thành 2 nếu Pacman bắt ở dist < 2
        
        # ==========================================================

        # === Hằng số chiến lược ===
        # Vì ta dùng mọi lúc, hãy dùng depth cao nhất có thể
        # mà không bị timeout (1.0s)
        self.MINIMAX_DEPTH = 8  # Thử 6, nếu chậm thì 4, nếu nhanh thì 8
        # === THÊM VÀO ĐÂY (4 DÒNG) ===
        self._map_height = 0
        self._map_width = 0
        self._junction_nodes = None    # Set các ngã rẽ
        self._junction_degrees = {}  # "Bộ não" học map (lưu độ an toàn)
    # ===================================================================
    # ========== CÁC HÀM CƠ BẢN (Giữ nguyên) ==========
    # ===================================================================
    
    def _is_valid_position(self, pos, map_state):
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
    
    def _apply_move(self, pos, move):
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)
    
    def _get_neighbors(self, pos, map_state):
        """Lấy 4 ô xung quanh (cho Ghost)."""
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors
    
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # === THÊM VÀO ĐÂY (HÀM SỐ 1) ===
    def _build_junction_graph(self, map_state):
        """
        "Học map": Tìm tất cả ngã rẽ VÀ lưu "độ an toàn" (degree)
        của chúng.
        """
        h, w = map_state.shape
        self._map_height, self._map_width = h, w # Cần lưu lại
        
        junctions = set()
        degrees = {}
        for r in range(h):
            for c in range(w):
                pos = (r, c)
                # Dùng hàm _is_valid_position (đã có)
                if self._is_valid_position(pos, map_state): 
                    # Dùng _get_neighbors (đã có) để tính degree
                    degree = len(self._get_neighbors(pos, map_state))
                    
                    # Ngã rẽ = không phải đường thẳng (degree != 2)
                    if degree != 2:
                        junctions.add(pos)
                        degrees[pos] = degree
                    
        self._junction_nodes = junctions
        self._junction_degrees = degrees # Đã "học" độ an toàn

    # === THÊM VÀO ĐÂY (HÀM SỐ 2) ===
    def _count_adjacent_walls(self, pos: tuple, map_state: np.ndarray) -> int:
            """
            Đếm số tường xung quanh (phiên bản ĐÃ SỬA LỖI).
            """
            wall_count = 0
            
            # Dùng _get_neighbors để lấy các ô hợp lệ xung quanh
            # và _is_wall để kiểm tra các ô không hợp lệ
            
            for move_dir in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                neighbor_pos = self._apply_move(pos, move_dir)
                
                # Nếu ô bên cạnh là tường (hoặc ngoài map)
                if self._is_wall(neighbor_pos, map_state):
                    wall_count += 1
                    
            return wall_count
    def _is_wall(self, pos: tuple, map_state: np.ndarray) -> bool:
            r, c = pos
            # Giả định _map_height và _map_width đã được set
            if not (0 <= r < self._map_height and 0 <= c < self._map_width):
                return True  # Ngoài bản đồ là tường
            return map_state[r, c] != 0
    # ===================================================================
    # ========== BỘ NÃO CHÍNH: 100% MINIMAX (ĐÃ BIẾT LUẬT) ==========
    # ===================================================================
    
    def minimax(self, map_state, pacman_pos, ghost_pos, depth, is_ghost_turn, 
                step_number, alpha=-float('inf'), beta=float('inf')):
        """
        Minimax "thuần túy" (chỉ quan tâm distance) nhưng "BIẾT LUẬT".
        """
        
        # === ĐIỀU KIỆN DỪNG (ĐÃ BIẾT LUẬT) ===
        distance = self._manhattan_distance(pacman_pos, ghost_pos)

        # 1. Bị bắt? (Dùng luật capture_threshold)
        if distance < self.capture_threshold:
            return -float('inf'), None  # Thua (rất tệ)

        # 2. Hết giờ? (Dùng luật max_steps)
        estimated_current_game_turn = step_number + (self.MINIMAX_DEPTH - depth) // 2
        if estimated_current_game_turn >= self.max_steps:
            return float('inf'), None  # Thắng (rất tốt)

        # 3. Hết độ sâu tìm kiếm? Đánh giá bằng Heuristic "Học Map"
        if depth == 0:
            # === BỘ NÃO MỚI ===
            
            # 1. "Học map": Vị trí này an toàn hay là bẫy?
            # 2 = hành lang bình thường (mặc định).
            degree = self._junction_degrees.get(ghost_pos, 2)
            
            # 2. LUẬT MỚI: CỰC KỲ GHÉT NGÕ CỤT (Bẫy của Pacman)
            if degree <= 1:
                return -100000.0, None # (Rất tệ! Gần như thua)
                
            # 3. Heuristic cũ (vẫn hữu ích)
            adjacent_wall_count = self._count_adjacent_walls(ghost_pos, map_state)

            # 4. Công thức điểm MỚI (kết hợp)
            # (Khoảng cách) + (Độ an toàn của ngã rẽ) - (Tường)
            score = (distance * 10.0) + (degree * 5.0) - (adjacent_wall_count * 2.0)
            
            return score, None
            # === HẾT BỘ NÃO MỚI ===
        
        # === LƯỢT CỦA GHOST (MAX) ===
        if is_ghost_turn:
            max_eval = -float('inf')
            best_move = Move.STAY
            
            # Tối ưu: Sắp xếp nước đi, ưu tiên nước đi "có vẻ"
            # xa Pacman nhất (Greedy)
            moves = self._get_neighbors(ghost_pos, map_state)
            moves.sort(key=lambda m: self._manhattan_distance(m[0], pacman_pos), reverse=True)

            for next_pos, move in moves:
                eval_score, _ = self.minimax(
                    map_state, pacman_pos, next_pos,
                    depth - 1, False, step_number + 1, alpha, beta
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        
        # === LƯỢT CỦA PACMAN (MIN) (Đã biết pacman_speed) ===
        else:
            min_eval = float('inf')
            
            # Tối ưu: Sắp xếp nước đi, ưu tiên nước "có vẻ"
            # gần Ghost nhất (Greedy)
            moves = self._get_pacman_simulated_moves(pacman_pos, map_state)
            moves.sort(key=lambda m: self._manhattan_distance(m[0], ghost_pos))

            for next_pos, _ in moves:
                eval_score, _ = self.minimax(
                    map_state, next_pos, ghost_pos,
                    depth - 1, True, step_number + 1, alpha, beta
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            if min_eval == float('inf'): # Fallback nếu Pacman bị kẹt
                 eval_score, _ = self.minimax(
                    map_state, pacman_pos, ghost_pos,
                    depth - 1, True, step_number + 1, alpha, beta
                )
                 min_eval = eval_score

            return min_eval, None

    # ===================================================================
    # ========== CÁC HÀM HỖ TRỢ CHO MINIMAX (BIẾT LUẬT) ==========
    # ===================================================================

    def _get_pacman_simulated_moves(self, pacman_pos, map_state):
        """Mô phỏng các nước đi của Pacman (với pacman_speed)."""
        possible_moves = []
        for initial_move_dir in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            final_pos = self._simulate_single_pacman_dash(
                pacman_pos, initial_move_dir, map_state, self.pacman_speed
            )
            if final_pos != pacman_pos:
                possible_moves.append((final_pos, initial_move_dir))
        
        if not possible_moves:
            possible_moves.append((pacman_pos, Move.STAY))
        return possible_moves

    def _simulate_single_pacman_dash(self, start_pos, move_direction, map_state, speed):
        """Mô phỏng cú "lướt" N bước của Pacman."""
        current_pos = start_pos
        dr, dc = move_direction.value
        for _ in range(speed):
            next_pos = (current_pos[0] + dr, current_pos[1] + dc)
            if not self._is_valid_position(next_pos, map_state):
                break 
            current_pos = next_pos
            # (Giữ đơn giản: Pacman có thể lướt qua ngã rẽ)
            # (Nếu muốn nó dừng ở ngã rẽ, thêm_check_junction ở đây)
        return current_pos

    # ===================================================================
    # ========== MAIN DECISION (100% MINIMAX) ==========
    # ===================================================================
    
    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Chiến lược "Tối thượng" v3.7:
        - Không có "if xa/gần".
        - Chỉ dùng 100% Minimax "Biết Luật".
        """
        # "Học map" 1 lần duy nhất khi game bắt đầu
        if self._junction_nodes is None:
            self._build_junction_graph(map_state)       
        _, best_move = self.minimax(
            map_state, enemy_position, my_position,
            depth=self.MINIMAX_DEPTH, 
            is_ghost_turn=True,
            step_number=step_number 
        )
        return best_move if best_move else Move.STAY