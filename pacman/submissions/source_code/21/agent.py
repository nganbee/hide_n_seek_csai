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

from collections import deque
import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import random
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
        super().__init__(**kwargs)
        self.name = "BFS Pacman"
    
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
        path = self._bfs(my_position, enemy_position, map_state)
        if path and len(path) > 1:
            next_pos = path[1]
            delta_row = next_pos[0] - my_position[0]
            delta_col = next_pos[1] - my_position[1]
            for move in Move:
                if move.value == (delta_row, delta_col):
                    return move
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)  
            if self._is_valid_position(new_pos, map_state):
                return move
        return Move.STAY
    
    def _bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        if start == goal:
            return [start]
        queue = deque([start])
        visited = set([start])
        parent = {start: None}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        found = False
        while queue:
            pos = queue.popleft()
            for dr, dc in directions:
                new_pos = (pos[0] + dr, pos[1] + dc)
                if (self._is_valid_position(new_pos, map_state) and
                    new_pos not in visited):
                    visited.add(new_pos)
                    parent[new_pos] = pos
                    queue.append(new_pos)
                    if new_pos == goal:
                        found = True
                        break
            if found:
                break
        if not found:
            return []
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape 
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS Ghost"
    
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
        dists = self._compute_distances_from(enemy_position, map_state)
        candidates = []
        directions = {
            Move.UP: (-1, 0),
            Move.DOWN: (1, 0),
            Move.LEFT: (0, -1),
            Move.RIGHT: (0, 1)
        }
        for move, (dr, dc) in directions.items():
            new_pos = (my_position[0] + dr, my_position[1] + dc)
            if self._is_valid_position(new_pos, map_state):
                dist = dists.get(new_pos, 0) 
                candidates.append((dist, move))
        if candidates:
            max_dist = max(c[0] for c in candidates)
            max_moves = [c[1] for c in candidates if c[0] == max_dist]
            return random.choice(max_moves) 
        return Move.STAY
    
    def _compute_distances_from(self, start: tuple, map_state: np.ndarray) -> dict:
        queue = deque([start])
        visited = set([start])
        dist = {start: 0}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            pos = queue.popleft()
            for dr, dc in directions:
                new_pos = (pos[0] + dr, pos[1] + dc)
                if (self._is_valid_position(new_pos, map_state) and
                    new_pos not in visited):
                    visited.add(new_pos)
                    dist[new_pos] = dist[pos] + 1
                    queue.append(new_pos)
        return dist

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0