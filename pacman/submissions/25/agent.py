import sys
from pathlib import Path
from collections import deque
from heapq import heappush, heappop
import itertools

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """
    Pacman Agent - Sử dụng A* + Minimax Hybrid
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_positions = []
        self.counter = itertools.count()  # Tạo counter unique
    
    def _is_valid_position(self, pos, map_state):
        """Kiểm tra vị trí hợp lệ"""
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0
    
    def _apply_move(self, pos, move):
        """Áp dụng nước đi"""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)
    
    def _get_neighbors(self, pos, map_state):
        """Lấy các ô láng giềng hợp lệ"""
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors
    
    def _manhattan_distance(self, pos1, pos2):
        """Khoảng cách Manhattan"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # ========== A* ALGORITHM (FIXED) ==========
    def astar(self, start, goal, map_state):
        """
        A* - Tìm đường ngắn nhất với heuristic
        FIX: Thêm counter để tránh so sánh Move objects
        """
        def heuristic(pos):
            return self._manhattan_distance(pos, goal)
        
        counter = itertools.count()  # Counter để tránh so sánh tuple
        frontier = [(0, next(counter), start, [])]  # (f_cost, counter, pos, path)
        visited = set()
        
        while frontier:
            f_cost, _, current_pos, path = heappop(frontier)
            
            if current_pos == goal:
                return path
            
            if current_pos in visited:
                continue
            
            visited.add(current_pos)
            
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    new_path = path + [move]
                    new_g_cost = len(new_path)
                    new_h_cost = heuristic(next_pos)
                    new_f_cost = new_g_cost + new_h_cost
                    heappush(frontier, (new_f_cost, next(counter), next_pos, new_path))
        
        return [Move.STAY]
    
    # ========== MINIMAX ALGORITHM ==========
    def minimax(self, map_state, pacman_pos, ghost_pos, depth, is_pacman_turn, alpha=-float('inf'), beta=float('inf')):
        """
        Minimax với Alpha-Beta Pruning
        """
        # Điều kiện dừng
        if depth == 0 or pacman_pos == ghost_pos:
            distance = self._manhattan_distance(pacman_pos, ghost_pos)
            return -distance, None
        
        if is_pacman_turn:
            max_eval = -float('inf')
            best_move = Move.STAY
            
            for next_pos, move in self._get_neighbors(pacman_pos, map_state):
                eval_score, _ = self.minimax(
                    map_state, next_pos, ghost_pos,
                    depth - 1, False, alpha, beta
                )
                
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
                eval_score, _ = self.minimax(
                    map_state, pacman_pos, next_pos,
                    depth - 1, True, alpha, beta
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    # ========== MAIN DECISION ==========
    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Hybrid Strategy:
        - Xa: Dùng A*
        - Gần: Dùng Minimax
        """
        distance = self._manhattan_distance(my_position, enemy_position)
        
        # Tránh lặp vô hạn
        self.previous_positions.append(my_position)
        if len(self.previous_positions) > 10:
            self.previous_positions.pop(0)
        
        # Xa: Dùng A* để tiếp cận
        if distance > 6:
            path = self.astar(my_position, enemy_position, map_state)
            if path and path[0] != Move.STAY:
                next_pos = self._apply_move(my_position, path[0])
                # Kiểm tra xem có bị lặp không
                if self.previous_positions.count(next_pos) < 3:
                    return path[0]
        
        # Gần: Dùng Minimax để chặn đường
        _, best_move = self.minimax(
            map_state, my_position, enemy_position,
            depth=3, is_pacman_turn=True
        )
        return best_move if best_move else Move.STAY


class GhostAgent(BaseGhostAgent):
    """
    Ghost Agent - Sử dụng Escape Route Analysis + Minimax
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
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
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors
    
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    # ========== ESCAPE ROUTE ANALYSIS ==========
    def count_escape_routes(self, pos, map_state, max_depth=4):
        """
        Đếm số lối thoát trong bán kính max_depth
        """
        visited = set([pos])
        queue = deque([(pos, 0)])
        escape_count = 0
        
        while queue:
            current_pos, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            neighbors = self._get_neighbors(current_pos, map_state)
            escape_count += len(neighbors)
            
            for next_pos, _ in neighbors:
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, depth + 1))
        
        return escape_count
    
    def smart_escape_move(self, my_position, enemy_position, map_state):
        """
        Chạy về vùng có nhiều lối thoát
        """
        best_move = Move.STAY
        best_score = -float('inf')
        
        for next_pos, move in self._get_neighbors(my_position, map_state):
            distance = self._manhattan_distance(next_pos, enemy_position)
            escape_routes = self.count_escape_routes(next_pos, map_state)
            
            # Công thức: 40% khoảng cách + 60% lối thoát
            score = distance * 0.4 + escape_routes * 0.6
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    # ========== MINIMAX FOR GHOST ==========
    def minimax(self, map_state, pacman_pos, ghost_pos, depth, is_ghost_turn, alpha=-float('inf'), beta=float('inf')):
        """
        Minimax từ góc nhìn Ghost
        """
        if depth == 0 or pacman_pos == ghost_pos:
            distance = self._manhattan_distance(pacman_pos, ghost_pos)
            return distance, None
        
        if is_ghost_turn:
            max_eval = -float('inf')
            best_move = Move.STAY
            
            for next_pos, move in self._get_neighbors(ghost_pos, map_state):
                eval_score, _ = self.minimax(
                    map_state, pacman_pos, next_pos,
                    depth - 1, False, alpha, beta
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        
        else:
            min_eval = float('inf')
            
            for next_pos, move in self._get_neighbors(pacman_pos, map_state):
                eval_score, _ = self.minimax(
                    map_state, next_pos, ghost_pos,
                    depth - 1, True, alpha, beta
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, None
    
    # ========== MAIN DECISION ==========
    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Hybrid Strategy:
        - Gần Pacman: Minimax
        - Xa Pacman: Escape Route
        """
        distance = self._manhattan_distance(my_position, enemy_position)
        
        # Gần: Dùng Minimax để trốn tối ưu
        if distance < 6:
            _, best_move = self.minimax(
                map_state, enemy_position, my_position,
                depth=4, is_ghost_turn=True
            )
            return best_move if best_move else Move.STAY
        
        # Xa: Chạy về vùng an toàn
        return self.smart_escape_move(my_position, enemy_position, map_state)