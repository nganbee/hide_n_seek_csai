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
        self.name = "BFS pacman"

    def step(self, map_state, my_position, enemy_position, step_number):
        path = self.bfs(my_position, enemy_position, map_state)
        if path:
            return path[0]  # Return first move in path
        return Move.STAY

    # Helper methods (you can add more)
    def bfs(self, start, goal, map_state):
        """
        Find shortest path from start to goal using BFS.

        Returns:
            List of Move enums representing the path, or [Move.STAY] if no path
        """
        from collections import deque

        # Queue stores (position, path_to_reach_it)
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current_pos, path = queue.popleft()

            # Found the goal!
            if current_pos == goal:
                return path

            # Explore neighbors
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))

        # No path found
        return [Move.STAY]
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos, map_state):
        """Check if position is valid (not wall, within bounds)."""
        row, col = pos
        height, width = map_state.shape

        # Check bounds
        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        # Check not a wall
        return map_state[row, col] == 0

    def _apply_move(self, pos, move):
        """Apply a move to a position, return new position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos, map_state):
        """Get all valid neighboring positions and their moves."""
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
        self.name = 'Greedy Ghost'

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
        # Tìm nước đi trốn thoát tốt nhất
        return self.find_escape_move(my_position, enemy_position, map_state)

    def find_escape_move(self, my_position, enemy_position, map_state):
        """
        Find move that maximizes distance from enemy.

        Returns:
            Move enum
        """
        best_move = Move.STAY
        best_distance = self._manhattan_distance(my_position, enemy_position)

        for next_pos, move in self._get_neighbors(my_position, map_state):
            distance = self._manhattan_distance(next_pos, enemy_position)
            if distance > best_distance:
                best_distance = distance
                best_move = move

        return random.choice(best_move)
    # Helper methods (you can add more)

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos, map_state):
        """Check if position is valid (not wall, within bounds)."""
        row, col = pos
        height, width = map_state.shape

        # Check bounds
        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        # Check not a wall
        return map_state[row, col] == 0

    def _apply_move(self, pos, move):
        """Apply a move to a position, return new position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos, map_state):
        """Get all valid neighboring positions and their moves."""
        neighbors = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))

        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
