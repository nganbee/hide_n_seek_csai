import sys
from pathlib import Path
import time
import heapq 
from collections import deque 
import numpy as np
import random

# --- Import Interface và Môi trường ---
try:
    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from agent_interface import PacmanAgent as BasePacmanAgent
    from agent_interface import GhostAgent as BaseGhostAgent
    from environment import Move
except ImportError:
    print("Không thể import interface, vui lòng đảm bảo cấu trúc thư mục là chính xác.")
    class BasePacmanAgent:
        def __init__(self, **kwargs): pass
    class BaseGhostAgent:
        def __init__(self, **kwargs): pass

    class Move:
        UP = (-1, 0)
        DOWN = (1, 0)
        LEFT = (0, -1)
        RIGHT = (0, 1)
        STAY = (0, 0)

# --- Helper cho PacmanAgent ---
DELTAS_TO_MOVE = {
    (-1, 0): Move.UP,
    (1, 0): Move.DOWN,
    (0, -1): Move.LEFT,
    (0, 1): Move.RIGHT,
    (0, 0): Move.STAY
}

# ==================================================================
#                       PACMAN AGENT (SEEKER)
# ==================================================================
class PacmanAgent(BasePacmanAgent):
    """
    Pacman agent (Seek Agent) sử dụng thuật toán A* để tìm đường ngắn nhất.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A_Star_Seeker"

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
    
    def _heuristic(self, pos1: tuple, pos2: tuple) -> int:
        """Tính toán heuristic: Khoảng cách Manhattan."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_move_from_positions(self, current_pos: tuple, next_pos: tuple) -> Move:
        """Chuyển đổi từ vị trí hiện tại và vị trí tiếp theo sang Move."""
        d_row = next_pos[0] - current_pos[0]
        d_col = next_pos[1] - current_pos[1]
        return DELTAS_TO_MOVE.get((d_row, d_col), Move.STAY)
    
    def _astar_search(self, map_state: np.ndarray, start: tuple, goal: tuple) -> list:
        """
        Thuật toán A* để tìm đường đi ngắn nhất.
        Trả về list các tọa độ [(r1, c1), (r2, c2), ...]
        """
        # Priority Queue: (f_cost, position)
        open_set = [(0 + self._heuristic(start, goal), start)] 
        g_score = {start: 0} # Chi phí thực tế từ start
        came_from = {} # Lưu vị trí liền trước

        while open_set:
            f_cost, current_pos = heapq.heappop(open_set) 
            
            if current_pos == goal:
                # Xây dựng lại đường đi
                path = []
                while current_pos != start:
                    path.append(current_pos)
                    current_pos = came_from[current_pos]
                path.append(start)
                path.reverse()
                return path

            current_g = g_score.get(current_pos, float('inf'))

            # Khám phá 4 hướng di chuyển (Không xét STAY trong A*)
            for move_delta in [Move.UP.value, Move.DOWN.value, Move.LEFT.value, Move.RIGHT.value]:
                delta_row, delta_col = move_delta
                neighbor_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)

                if self._is_valid_position(neighbor_pos, map_state):
                    # Chi phí mỗi bước đi là 1
                    tentative_g_score = current_g + 1 

                    if tentative_g_score < g_score.get(neighbor_pos, float('inf')):
                        # Đây là đường đi tốt hơn
                        came_from[neighbor_pos] = current_pos
                        g_score[neighbor_pos] = tentative_g_score
                        
                        h_cost = self._heuristic(neighbor_pos, goal)
                        f_cost = tentative_g_score + h_cost
                        
                        heapq.heappush(open_set, (f_cost, neighbor_pos))
                        
        return []

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        # 1. Chạy thuật toán A* để tìm đường đi ngắn nhất đến enemy_position
        path = self._astar_search(map_state, my_position, enemy_position)
        
        # 2. Nếu tìm thấy đường đi, trả về bước đi đầu tiên
        if path and len(path) > 1:
            next_pos = path[1]
            return self._get_move_from_positions(my_position, next_pos)
        
        # 3. Nếu không tìm thấy đường đi hoặc đường đi đã bị chặn, STAY
        return Move.STAY


# ==================================================================
#                       GHOST AGENT (HIDER)
# ==================================================================
class GhostAgent(BaseGhostAgent):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.TIME_LIMIT = 0.9 
        self.start_time = 0
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _get_valid_moves(self, pos: tuple, map_state: np.ndarray) -> list:
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            if move == Move.STAY:
                valid_moves.append((Move.STAY, pos))
                continue
                
            delta_row, delta_col = move.value
            new_pos = (pos[0] + delta_row, pos[1] + delta_col)
            if self._is_valid_position(new_pos, map_state):
                valid_moves.append((move, new_pos))
        return valid_moves

    def _get_bfs_distance(self, map_state: np.ndarray, start: tuple, goal: tuple) -> int:
        if start == goal:
            return 0
            
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            current, dist = queue.popleft()
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                neighbor = (current[0] + delta_row, current[1] + delta_col)

                if neighbor == goal:
                    return dist + 1
                    
                if self._is_valid_position(neighbor, map_state) and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    
        return float('inf') 

    def _get_simulated_seek_move(self, map_state: np.ndarray, my_pos: tuple, enemy_pos: tuple) -> tuple:
        best_dist = float('inf')
        best_new_pos = enemy_pos 

        for move in [Move.STAY, Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            new_pos = (enemy_pos[0] + delta_row, enemy_pos[1] + delta_col)

            if not self._is_valid_position(new_pos, map_state):
                continue

            dist = self._get_bfs_distance(map_state, new_pos, my_pos)
            
            if dist < best_dist:
                best_dist = dist
                best_new_pos = new_pos
                
        return best_new_pos

    def _evaluate_state(self, map_state: np.ndarray, my_pos: tuple, enemy_pos: tuple) -> float:
        if my_pos == enemy_pos:
            return -float('inf') 
        
        distance = self._get_bfs_distance(map_state, my_pos, enemy_pos)
        if distance == float('inf'):
            distance = 1000 
        
        freedom = len(self._get_valid_moves(my_pos, map_state))
        
        if freedom <= 2:
            freedom_penalty = -50
        else:
            freedom_penalty = freedom * 2
            
        return (distance * 10) + freedom_penalty

    def _minimax_recursive(self, map_state: np.ndarray, my_pos: tuple, enemy_pos: tuple, 
                           depth: int, alpha: float, beta: float, is_maximizing_player: bool) -> float:
        
        if (time.time() - self.start_time) > self.TIME_LIMIT:
            raise TimeoutError() 

        if depth == 0 or my_pos == enemy_pos:
            return self._evaluate_state(map_state, my_pos, enemy_pos)

        if is_maximizing_player:
            max_eval = -float('inf')
            for move, new_my_pos in self._get_valid_moves(my_pos, map_state):
                eval = self._minimax_recursive(map_state, new_my_pos, enemy_pos, 
                                             depth - 1, alpha, beta, False) 
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break 
            return max_eval
        
        else:
            min_eval = float('inf')
            
            new_enemy_pos = self._get_simulated_seek_move(map_state, my_pos, enemy_pos)
            
            eval = self._minimax_recursive(map_state, my_pos, new_enemy_pos, 
                                         depth - 1, alpha, beta, True) 
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            
            return min_eval

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        self.start_time = time.time()
        best_move = Move.STAY
        
        depth = 2
        
        while True:
            try:
                current_best_move, current_best_score = self._find_best_move_at_depth(
                    map_state, my_position, enemy_position, depth
                )
                
                best_move = current_best_move
                
                depth += 2 
                
                if (time.time() - self.start_time) > self.TIME_LIMIT:
                    break 
                    
            except TimeoutError:
                break 
            except Exception as e:
                break
                
        return best_move

    def _find_best_move_at_depth(self, map_state, my_pos, enemy_pos, depth):
        best_score = -float('inf')
        best_move = Move.STAY
        alpha = -float('inf')
        beta = float('inf')

        for move, new_my_pos in self._get_valid_moves(my_pos, map_state):
            
            score = self._minimax_recursive(map_state, new_my_pos, enemy_pos, 
                                          depth - 1, alpha, beta, False)
            
            if score > best_score:
                best_score = score
                best_move = move
                
            alpha = max(alpha, best_score)
            
        return best_move, best_score