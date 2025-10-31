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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Potential Field Ghost"
        self.K = 120.0
        self.wall_penalty = 4.0
        self.random_eps = 0.12
        self.corridor_penalty = 6.0

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if my_position == enemy_position:
            return Move.STAY

        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        best_move = Move.STAY
        best_score = float('inf')

        for move in moves:
            dr, dc = move.value
            new_pos = (my_position[0] + dr, my_position[1] + dc)
            if not self._valid(new_pos, map_state):
                continue

            pot = self._repulsive_potential(new_pos, enemy_position)
            wall_p = self._adjacent_wall_count(new_pos, map_state) * self.wall_penalty
            free_neighbors = self._free_neighbor_count(new_pos, map_state)
            corridor_pen = self.corridor_penalty if free_neighbors <= 1 else 0.0
            jitter = random.uniform(-1, 1) * self.random_eps * self.K

            score = pot + wall_p + corridor_pen + jitter

            # tie-breaker: prefer farther from Pacman
            if score < best_score - 1e-6:
                best_score = score
                best_move = move
            elif abs(score - best_score) <= 1e-6:
                cur_dist = self._euclid_dist(my_position, enemy_position)
                new_dist = self._euclid_dist(new_pos, enemy_position)
                if new_dist > cur_dist:
                    best_move = move
        return best_move

    # -------------- helpers ---------------
    def _repulsive_potential(self, pos, pacman_pos):
        dist = self._euclid_dist(pos, pacman_pos)
        if dist <= 0:
            return float("inf")
        return self.K / dist

    def _valid(self, pos, map_state):
        h, w = map_state.shape
        r, c = pos
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    def _adjacent_wall_count(self, pos, map_state):
        count = 0
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        h, w = map_state.shape
        for dr, dc in dirs:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < h and 0 <= nc < w) or map_state[nr, nc] != 0:
                count += 1
        return count

    def _free_neighbor_count(self, pos, map_state):
        count = 0
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        h, w = map_state.shape
        for dr, dc in dirs:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] == 0:
                count += 1
        return count

    def _euclid_dist(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5