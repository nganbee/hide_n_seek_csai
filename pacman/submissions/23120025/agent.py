"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
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
    Example Pacman agent using a simple greedy strategy.
    Students should implement their own search algorithms here.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Pacman agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "Example Greedy Pacman"
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Use BFS to find the shortest path to the ghost and take the
        first step along that path. Falls back to a random valid
        move if no path is found.
        """
        # If already at the same position, stay
        if my_position == enemy_position:
            return Move.STAY

        first_move = self._bfs_first_move(map_state, my_position, enemy_position)
        if first_move is not None:
            return first_move

        # Fallback: pick any valid move at random
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            if self._is_valid_position(new_pos, map_state):
                return move

        return Move.STAY

    def _bfs_first_move(self, map_state: np.ndarray, start: tuple, goal: tuple):
        """
        Run BFS on the grid (map_state) from start to goal.
        Returns the first Move to take along the shortest path,
        or None if no path exists.
        """
        from collections import deque

        # Quick bounds check
        h, w = map_state.shape

        def valid(pos):
            r, c = pos
            if r < 0 or r >= h or c < 0 or c >= w:
                return False
            return map_state[r, c] == 0

        # BFS structures
        q = deque()
        q.append(start)
        came_from = {start: None}

        # Neighbors mapping to Moves (exclude STAY)
        neighbors = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

        while q:
            current = q.popleft()
            if current == goal:
                break
            for move in neighbors:
                dr, dc = move.value
                nxt = (current[0] + dr, current[1] + dc)
                if nxt in came_from:
                    continue
                if not valid(nxt):
                    continue
                came_from[nxt] = (current, move)
                q.append(nxt)

        # If goal wasn't reached
        if goal not in came_from:
            return None

        # Reconstruct path: walk back from goal to start to find first move
        node = goal
        last_move = None
        while came_from[node] is not None:
            prev, move = came_from[node]
            last_move = move
            node = prev

        # last_move is the move that led from start to the next node
        return last_move
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Example Ghost agent using a simple evasive strategy.
    Students should implement their own search algorithms here.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Ghost agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "Example Evasive Ghost"
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Simple evasive strategy: move away from Pacman.
        
        Students should implement better search algorithms like:
        - BFS to find furthest point
        - A* to plan escape route
        - Minimax for adversarial search
        - etc.
        """
        # Calculate direction away from Pacman
        row_diff = my_position[0] - enemy_position[0]
        col_diff = my_position[1] - enemy_position[1]
        
        # List of possible moves in order of preference
        moves = []
        
        # Prioritize vertical movement away from Pacman
        if row_diff > 0:
            moves.append(Move.DOWN)
        elif row_diff < 0:
            moves.append(Move.UP)
        
        # Prioritize horizontal movement away from Pacman
        if col_diff > 0:
            moves.append(Move.RIGHT)
        elif col_diff < 0:
            moves.append(Move.LEFT)
        
        # Try each move in order
        for move in moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            # Check if move is valid
            if self._is_valid_position(new_pos, map_state):
                return move
        
        # If no preferred move is valid, try any valid move
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            if self._is_valid_position(new_pos, map_state):
                return move
        
        # If no move is valid, stay
        return Move.STAY
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

