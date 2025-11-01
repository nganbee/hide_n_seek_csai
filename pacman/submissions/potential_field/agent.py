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

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random
import math


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        # - self.name = "Your Agent Name"
        pass
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Ghost's current (row, col)
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        
        # Example: Simple greedy approach (replace with your algorithm)
        row_diff = enemy_position[0] - my_position[0]
        col_diff = enemy_position[1] - my_position[1]
        
        # Try to move towards ghost
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
        
        # Check if move is valid
        if self._is_valid_move(my_position, move, map_state):
            return move
        
        # If not valid, try other moves
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                return move
        
        return Move.STAY
    
    # Helper methods (you can add more)
    
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


class GhostAgent(BaseGhostAgent):
    """
    Smart Ghost v2.1 – Fixed Potential Field
    • Không đứng yên khi bị đuổi gần
    • Không chạy ngược về phía Pacman
    • Né vuông góc + wall hugging + openness bias
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Smart Ghost v2.1 (Safe Evasion)"
        self.K = 200.0
        self.wall_penalty = 3.0
        self.escape_bias = 1.6
        self.momentum_weight = 2.5
        self.random_eps = 0.08
        self.safe_distance = 4  # critical radius
        self.last_move = Move.STAY
        self.last_pacman_pos = None

    def step(self, map_state, my_pos, pac_pos, step_number):
        if my_pos == pac_pos:
            return Move.STAY

        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        best_move = Move.STAY
        best_score = float("inf")
        fallback_move = Move.STAY

        predicted_pac = self._predict_pacman(pac_pos)
        dist_now = self._euclid_dist(my_pos, pac_pos)

        for move in moves:
            dr, dc = move.value
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            if not self._valid(new_pos, map_state):
                continue

            new_dist = self._euclid_dist(new_pos, pac_pos)
            # ------------------ danger filter ------------------
            # Nếu Pacman gần và bước này làm khoảng cách giảm → bỏ
            if dist_now <= self.safe_distance and new_dist < dist_now - 0.1:
                continue

            # Base potential field
            pot = self._repulsive_potential(new_pos, predicted_pac)
            wall_p = self._adjacent_wall_count(new_pos, map_state) * self.wall_penalty
            openness = self._flood_fill_space(map_state, new_pos)
            escape_bonus = -openness * self.escape_bias

            # Momentum (ưu tiên hướng đang đi)
            momentum_bonus = -self.momentum_weight if move == self.last_move else 0

            # Random noise giảm dần khi gần Pacman
            jitter_strength = self.random_eps * self.K * (1 if dist_now > self.safe_distance else 0.3)
            jitter = random.uniform(-1, 1) * jitter_strength

            score = pot + wall_p + escape_bonus + momentum_bonus + jitter

            # Track best
            if score < best_score:
                best_score = score
                best_move = move

            # Fallback: chọn hướng xa Pacman nhất
            if new_dist > self._euclid_dist(my_pos, self._apply_move(my_pos, fallback_move)):
                fallback_move = move

        # Nếu không có hướng hợp lệ → fallback
        final_move = best_move if best_move != Move.STAY else fallback_move

        # Nếu vẫn STAY → đi tiếp hướng cũ để tránh đứng yên
        if final_move == Move.STAY:
            final_move = self.last_move

        self.last_move = final_move
        self.last_pacman_pos = pac_pos
        return final_move

    # ------------------------------------------------------------
    def _predict_pacman(self, pac_pos):
        if not self.last_pacman_pos:
            return pac_pos
        dr = pac_pos[0] - self.last_pacman_pos[0]
        dc = pac_pos[1] - self.last_pacman_pos[1]
        return (pac_pos[0] + dr, pac_pos[1] + dc)

    def _repulsive_potential(self, pos, pac_pos):
        dist = self._euclid_dist(pos, pac_pos)
        if dist <= 0:
            return float("inf")
        return self.K / (dist ** 1.4)

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _valid(self, pos, map_state):
        h, w = map_state.shape
        r, c = pos
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    def _adjacent_wall_count(self, pos, map_state):
        count = 0
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        h, w = map_state.shape
        for dr, dc in dirs:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < h and 0 <= nc < w) or map_state[nr, nc] != 0:
                count += 1
        return count

    def _flood_fill_space(self, map_state, start):
        """Đếm số ô trống xung quanh để ưu tiên vùng mở."""
        h, w = map_state.shape
        q = [start]
        visited = {start}
        count = 0
        limit = 8
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q and count < limit:
            r, c = q.pop(0)
            count += 1
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] == 0 and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return count

    def _euclid_dist(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)