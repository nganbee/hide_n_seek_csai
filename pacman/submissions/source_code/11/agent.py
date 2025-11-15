"""
Agent File 2: Greedy Best-First Pacman + Minimax Ghost
- Pacman: Sử dụng Greedy Best-First Search (nhanh hơn A* nhưng không đảm bảo tối ưu)
- Ghost: Sử dụng Minimax với Alpha-Beta Pruning để dự đoán và tránh Pacman
"""

import sys
from pathlib import Path
import heapq
import numpy as np
import random

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


class PacmanAgent(BasePacmanAgent):
    """
    Pacman Agent sử dụng Greedy Best-First Search.
    Chỉ dựa vào heuristic h(n) để chọn nước đi, không quan tâm đến cost g(n).
    Nhanh hơn A* nhưng không đảm bảo đường đi ngắn nhất.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Greedy Best-First Pacman"
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Tìm đường đi bằng Greedy Best-First Search.
        """
        path = self._greedy_search(map_state, my_position, enemy_position)
        
        if path and len(path) > 1:
            next_pos = path[1]
            return self._get_move_direction(my_position, next_pos)
        
        return self._get_random_valid_move(map_state, my_position)
    
    def _greedy_search(self, map_state: np.ndarray, start: tuple, goal: tuple) -> list:
        """
        Greedy Best-First Search: chỉ dùng heuristic h(n).
        """
        counter = 0
        # Priority queue: (h_score, counter, position, path)
        frontier = [(0, counter, start, [start])]
        explored = set()
        
        while frontier:
            h_score, _, current, path = heapq.heappop(frontier)
            
            if current == goal:
                return path
            
            if current in explored:
                continue
            
            explored.add(current)
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                neighbor = (current[0] + delta_row, current[1] + delta_col)
                
                if neighbor not in explored and self._is_valid_position(neighbor, map_state):
                    # Chỉ dùng heuristic, không quan tâm cost thực tế
                    h_score = self._manhattan_distance(neighbor, goal)
                    
                    counter += 1
                    new_path = path + [neighbor]
                    heapq.heappush(frontier, (h_score, counter, neighbor, new_path))
        
        return None
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Tính Manhattan distance giữa 2 điểm."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_move_direction(self, current: tuple, next_pos: tuple) -> Move:
        """Chuyển đổi từ 2 vị trí thành Move direction."""
        delta_row = next_pos[0] - current[0]
        delta_col = next_pos[1] - current[1]
        
        if delta_row == -1:
            return Move.UP
        elif delta_row == 1:
            return Move.DOWN
        elif delta_col == -1:
            return Move.LEFT
        elif delta_col == 1:
            return Move.RIGHT
        
        return Move.STAY
    
    def _get_random_valid_move(self, map_state: np.ndarray, position: tuple) -> Move:
        """Lấy một nước di chuyển hợp lệ ngẫu nhiên."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (position[0] + delta_row, position[1] + delta_col)
            
            if self._is_valid_position(new_pos, map_state):
                return move
        
        return Move.STAY
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Kiểm tra vị trí có hợp lệ không."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost Agent sử dụng Minimax Algorithm với Alpha-Beta Pruning.
    - Ghost (MAX player): maximize khoảng cách với Pacman
    - Pacman (MIN player): minimize khoảng cách với Ghost
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Minimax Ghost"
        self.max_depth = 4  # Độ sâu tìm kiếm
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Chọn nước đi tối ưu bằng Minimax với Alpha-Beta Pruning.
        """
        best_move = self._minimax_decision(map_state, my_position, enemy_position)
        
        if best_move:
            return best_move
        
        return self._get_random_valid_move(map_state, my_position)
    
    def _minimax_decision(self, map_state: np.ndarray, 
                          ghost_pos: tuple, 
                          pacman_pos: tuple) -> Move:
        """
        Quyết định nước đi tốt nhất cho Ghost bằng Minimax.
        """
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        # Thử tất cả các nước đi hợp lệ
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            new_pos = self._get_new_position(ghost_pos, move)
            
            if not self._is_valid_position(new_pos, map_state):
                continue
            
            # Tính điểm cho nước đi này (Ghost = MAX player)
            score = self._minimax(map_state, new_pos, pacman_pos, 
                                 self.max_depth - 1, alpha, beta, False)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
        
        return best_move
    
    def _minimax(self, map_state: np.ndarray, 
                 ghost_pos: tuple, 
                 pacman_pos: tuple, 
                 depth: int, 
                 alpha: float, 
                 beta: float, 
                 is_ghost_turn: bool) -> float:
        """
        Minimax với Alpha-Beta Pruning.
        
        Args:
            is_ghost_turn: True nếu lượt Ghost (MAX), False nếu lượt Pacman (MIN)
            alpha: giá trị tốt nhất cho MAX player
            beta: giá trị tốt nhất cho MIN player
        """
        # Base case: đạt độ sâu max hoặc game over
        if depth == 0 or ghost_pos == pacman_pos:
            return self._evaluate_state(ghost_pos, pacman_pos, map_state)
        
        if is_ghost_turn:
            # Ghost (MAX player) - muốn maximize score
            max_score = float('-inf')
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                new_ghost_pos = self._get_new_position(ghost_pos, move)
                
                if not self._is_valid_position(new_ghost_pos, map_state):
                    continue
                
                score = self._minimax(map_state, new_ghost_pos, pacman_pos, 
                                     depth - 1, alpha, beta, False)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                
                # Alpha-Beta Pruning
                if beta <= alpha:
                    break
            
            return max_score if max_score != float('-inf') else 0
        else:
            # Pacman (MIN player) - muốn minimize score
            min_score = float('inf')
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                new_pacman_pos = self._get_new_position(pacman_pos, move)
                
                if not self._is_valid_position(new_pacman_pos, map_state):
                    continue
                
                score = self._minimax(map_state, ghost_pos, new_pacman_pos, 
                                     depth - 1, alpha, beta, True)
                min_score = min(min_score, score)
                beta = min(beta, score)
                
                # Alpha-Beta Pruning
                if beta <= alpha:
                    break
            
            return min_score if min_score != float('inf') else 0
    
    def _evaluate_state(self, ghost_pos: tuple, pacman_pos: tuple, map_state: np.ndarray) -> float:
        """
        Hàm đánh giá trạng thái.
        - Score cao = Ghost xa Pacman (tốt cho Ghost)
        - Score thấp = Ghost gần Pacman (xấu cho Ghost)
        """
        # Manhattan distance
        distance = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        
        # Penalty nếu quá gần Pacman
        if distance <= 1:
            return -1000
        
        # Bonus cho việc ở xa Pacman
        score = distance * 10
        
        # Bonus nếu ở gần góc/cạnh (khó bị bắt hơn)
        height, width = map_state.shape
        if ghost_pos[0] in [0, height-1] or ghost_pos[1] in [0, width-1]:
            score += 5
        
        return score
    
    def _get_new_position(self, position: tuple, move: Move) -> tuple:
        """Tính vị trí mới sau khi di chuyển."""
        delta_row, delta_col = move.value
        return (position[0] + delta_row, position[1] + delta_col)
    
    def _get_random_valid_move(self, map_state: np.ndarray, position: tuple) -> Move:
        """Lấy một nước di chuyển hợp lệ ngẫu nhiên."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        random.shuffle(all_moves)
        
        for move in all_moves:
            new_pos = self._get_new_position(position, move)
            
            if self._is_valid_position(new_pos, map_state):
                return move
        
        return Move.STAY
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Kiểm tra vị trí có hợp lệ không."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0