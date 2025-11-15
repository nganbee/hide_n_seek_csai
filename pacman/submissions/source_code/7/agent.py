"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- You MUST return a Move enum value from step()
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
"""

import sys
from pathlib import Path
import random
# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
from collections import deque
from heapq import heappop, heappush


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent với thuật toán A* (đã sửa lỗi tie-breaking)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A_Star_Pacman_Fixed"
        self.current_path = []
        self.last_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        # Logic Path Caching (giữ nguyên)
        if not self.current_path or enemy_position != self.last_enemy_pos:
            self.current_path = self.astar(my_position, enemy_position, map_state)
            self.last_enemy_pos = enemy_position
        
        if self.current_path:
            next_move = self.current_path.pop(0)
            return next_move
        
        return Move.STAY

    # ===== HÀM ASTAR ĐÃ SỬA LỖI =====
    
    def astar(self, start, goal, map_state):
        """
        Tìm đường đi tối ưu từ start đến goal bằng A*
        ĐÃ SỬA LỖI TIE-BREAKING CỦA PYTHON 3
        """
        
        def heuristic(pos):
            """Heuristic là khoảng cách Manhattan."""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # SỬA 1: Thêm counter
        unique_counter = 0 
        
        # Hàng đợi ưu tiên: (f_cost, unique_id, position, path)
        # SỬA 2: Thêm counter vào frontier
        frontier = [(heuristic(start), unique_counter, start, [])] 
        visited = set()
        
        while frontier:
            # SỬA 3: Lấy counter ra (dùng _ để bỏ qua)
            f_cost_total, _, current_pos, path = heappop(frontier)
            
            # (Phần còn lại giữ nguyên logic)
            if current_pos == goal:
                return path 
            
            if current_pos in visited:
                continue
            
            visited.add(current_pos)
            
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    new_path = path + [move]
                    g_cost = len(new_path)
                    h_cost = heuristic(next_pos)
                    f_cost = g_cost + h_cost
                    
                    # SỬA 4: Thêm counter vào heappush
                    heappush(frontier, (f_cost, unique_counter, next_pos, new_path))
                    unique_counter += 1 # Tăng counter
        
        # Không tìm thấy đường
        return [Move.STAY]

    # ===== CÁC HÀM HELPER (Giữ nguyên) =====
    
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
    

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furx`thest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        
        self.strategy = "MINIMAX"
        
        # Search depth for Minimax/Expectimax. 
     
        self.SEARCH_DEPTH = 3
        
        # Cache for Minimax/Expectimax to speed up calculation
        self.cache = {}
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Pacman's current (row, col)
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        
        # --- Strategy Router ---
        # Clear cache at the start of a new move
        self.cache = {}
        
     
        best_score = -float('inf')
        best_move = Move.STAY
            
        # Try all 5 possible moves (incl. STAY)
        for move in self._get_all_moves(my_position, map_state, include_stay=True):
            next_my_pos = self._apply_move(my_position, move)
                
            # Call minimax for Pacman's turn (MIN)
            score, _ = self._minimax(map_state, next_my_pos, enemy_position, self.SEARCH_DEPTH, False)
                
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

        

    # --- ALGORITHM: MINIMAX ---
    
    def _minimax(self, map_state, my_pos, enemy_pos, depth, is_max_turn):
        """
        Recursive Minimax function.
        - is_max_turn = True: Ghost's turn (Maximize score)
        - is_max_turn = False: Pacman's turn (Minimize score)
        """
        # Use cache (memoization) to speed up
        state = (my_pos, enemy_pos, depth, is_max_turn)
        if state in self.cache:
            return self.cache[state]

        # Base case: Reached depth limit or game over
        if depth == 0 or my_pos == enemy_pos:
            score = self._evaluate_heuristic(my_pos, enemy_pos, map_state)
            return score, Move.STAY
        
        if is_max_turn: # Ghost's turn (MAX)
            best_score = -float('inf')
            best_move = Move.STAY
            
            for move in self._get_all_moves(my_pos, map_state, include_stay=True):
                next_my_pos = self._apply_move(my_pos, move)
                # Call for Pacman's turn (MIN)
                score, _ = self._minimax(map_state, next_my_pos, enemy_pos, depth, False)
                if score > best_score:
                    best_score = score
                    best_move = move
            
            self.cache[state] = (best_score, best_move)
            return best_score, best_move
        
        else: # Pacman's turn (MIN)
            best_score = float('inf')
            best_move = Move.STAY
            
            for move in self._get_all_moves(enemy_pos, map_state, include_stay=True):
                next_enemy_pos = self._apply_move(enemy_pos, move)
                # Call for Ghost's turn (MAX), decrement depth
                score, _ = self._minimax(map_state, my_pos, next_enemy_pos, depth - 1, True)
                if score < best_score:
                    best_score = score
                    best_move = move
            
            self.cache[state] = (best_score, best_move)
            return best_score, best_move



    # --- GHOST HELPER METHODS ---
    
    def _evaluate_heuristic(self, my_pos, enemy_pos, map_state):
        """
        Heuristic function for Minimax/Expectimax.
        Ghost wants to MAXIMIZE this score.
        """
        if my_pos == enemy_pos:
            return -1000000 # Very bad score if caught
        
        # Primary heuristic: distance to Pacman (maximize this)
        distance = self._manhattan_distance(my_pos, enemy_pos)
        
        # Secondary heuristic: avoid walls (minimize this)
        wall_count = 0
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
             if not self._is_valid_position(self._apply_move(my_pos, m), map_state):
                   wall_count += 1
        
        # We want high distance and low wall count
        return distance - (wall_count * 0.5)

    def _manhattan_distance(self, pos1, pos2):
        """Calculates Manhattan distance."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        """Helper: Applies a move to a position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_all_moves(self, pos: tuple, map_state: np.ndarray, include_stay: bool = True):
        """Returns a list of valid Move enums from a position."""
        moves = []
        if include_stay:
            moves.append(Move.STAY)
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(pos, move, map_state):
                moves.append(move)
        return moves
    
    # Helper methods (from template)
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0