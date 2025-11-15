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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = []
        self.name = "Example Greedy Pacman"
        self.old_position = None
    
    def bfs(self, start, goal, map_state):
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

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        manhattan_dist = self._manhattan_distance(my_position, enemy_position)

        if manhattan_dist <= 1:
            return Move.STAY  # Avoid moving into the ghost
        elif manhattan_dist <= 5:
            self.path = self.bfs(my_position, enemy_position, map_state)
        else:
            if self.old_position == None:
                self.old_position = my_position
            elif self._manhattan_distance(enemy_position, self.old_position) > 5:
                self.path = self.bfs(my_position, enemy_position, map_state)
                self.old_position = enemy_position

        if self.path:
            move = self.path[0]
            self.path.pop(0)           
            return move  # Return first move in path
        
        return Move.STAY  # Placeholder implementation
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Example Evasive Ghost"
        self.visited = []
        self.not_visit = []


    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        valid_pos = self._get_neighbors(my_position, map_state)
        valid_pos.append((my_position, Move.STAY))
        best_move = None

        for pos in valid_pos:
            manhatan_dist = self._manhattan_distance(pos[0], enemy_position)
            if best_move == None or best_move[0] < manhatan_dist:
                best_move = [manhatan_dist, pos[0], pos[1]]

        return best_move[2]
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
 
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
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
