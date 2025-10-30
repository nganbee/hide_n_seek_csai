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
    
    def dfs(self, start, goal, map_state):
        """
        Find a path from start to goal using Depth-First Search (DFS).
    
        Returns:
            List of Move enums representing the path, or [Move.STAY] if no path.
        """
        # Stack stores (position, path_to_reach_it)
        stack = [(start, [])]
        visited = {start}
    
        while stack:
            current_pos, path = stack.pop()
        
            # Found the goal!
            if current_pos == goal:
                return path
        
            # Explore neighbors (order affects path shape)
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    stack.append((next_pos, path + [move]))
    
        # No path found
        return [Move.STAY]

    def step(self, map_state, my_position, enemy_position, step_number):
        path = self.dfs(my_position, enemy_position, map_state)
        if path:
            return path[0]  # Return first move in path
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

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        """Apply a move to a position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        """Get all valid neighboring positions."""
        neighbors = []
    
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
    
        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        pass
    
    def minimax(self, state, depth, is_maximizing_player, map_state):
        """
        Minimax algorithm for Ghost (minimizing player).
    
        Args:
            state: (my_pos, enemy_pos)
            depth: Search depth remaining
            is_maximizing_player: True if Pacman's turn, False if Ghost's turn
            map_state: The maze map
        
        Returns:
            (best_score, best_move)
        """
        my_pos, enemy_pos = state
    
        # Base case: reached depth limit or caught
        if depth == 0 or my_pos == enemy_pos:
            # Return negative distance (Ghost wants to maximize distance)
            return -self._manhattan_distance(my_pos, enemy_pos), Move.STAY
    
        if is_maximizing_player:  # Pacman's turn (wants to minimize distance)
            best_score = float('-inf')
            best_move = Move.STAY
        
            for next_pos, move in self._get_neighbors(enemy_pos, map_state):
                new_state = (my_pos, next_pos)
                score, _ = self.minimax(new_state, depth-1, False, map_state)
                if score > best_score:
                    best_score = score
                    best_move = move
        
            return best_score, best_move
    
        else:  # Ghost's turn (wants to maximize distance)
            best_score = float('inf')
            best_move = Move.STAY
        
            for next_pos, move in self._get_neighbors(my_pos, map_state):
                new_state = (next_pos, enemy_pos)
                score, _ = self.minimax(new_state, depth-1, True, map_state)
                if score < best_score:
                    best_score = score
                    best_move = move
        
            return best_score, best_move

    def step(self, map_state, my_position, enemy_position, step_number):
        _, best_move = self.minimax(
            (my_position, enemy_position),
            depth=3,  # Search 3 moves ahead
            is_maximizing_player=False,
            map_state=map_state
        )
        return best_move
    
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

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        """Apply a move to a position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        """Get all valid neighboring positions."""
        neighbors = []
    
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
    
        return neighbors
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])