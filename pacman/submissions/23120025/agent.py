"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
"""

import sys
from pathlib import Path
from collections import deque

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
        self.name = "BFS Pacman"
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Using BFS (Breadth-First Search)
        """
        if my_position == enemy_position:
            return Move.STAY

        first_move = self._bfs_first_move(map_state, my_position, enemy_position)
        if first_move is not None:
            return first_move

        # fallback: random valid move
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for move in all_moves:
            dr, dc = move.value
            new_pos = (my_position[0] + dr, my_position[1] + dc)
            if self._is_valid_position(new_pos, map_state):
                return move
        return Move.STAY
    
    def _bfs_first_move(self, map_state, start, goal):
        h, w = map_state.shape

        def valid(pos):
            r, c = pos
            return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

        q = deque([start])
        came_from = {start: None}
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for move in moves:
                dr, dc = move.value
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt not in came_from and valid(nxt):
                    came_from[nxt] = (cur, move)
                    q.append(nxt)

        if goal not in came_from:
            return None

        # Reconstruct path to find first move
        cur = goal
        while came_from[cur][0] != start:
            cur = came_from[cur][0]
        return came_from[cur][1]
    
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
        self.name = "BFS Escape Ghost"
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Using BFS to find furthest point
        """
        if my_position == enemy_position:
            return Move.STAY

        target = self._find_farthest_point(map_state, my_position, enemy_position)
        if target is None:
            return Move.STAY

        first_move = self._bfs_first_move(map_state, my_position, target)
        if first_move is not None:
            return first_move

        return Move.STAY
    
    def _find_farthest_point(self, map_state, start, pacman_pos):
        h, w = map_state.shape

        def valid(pos):
            r, c = pos
            return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

        q = deque([pacman_pos])
        dist = {pacman_pos: 0}

        while q:
            cur = q.popleft()
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nxt = (cur[0] + dr, cur[1] + dc)
                if valid(nxt) and nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)

        farthest = None
        max_dist = -1
        for pos, d in dist.items():
            if valid(pos) and d > max_dist:
                max_dist = d
                farthest = pos
        return farthest

    def _bfs_first_move(self, map_state, start, goal):
        h, w = map_state.shape

        def valid(pos):
            r, c = pos
            return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

        q = deque([start])
        came_from = {start: None}
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for move in moves:
                dr, dc = move.value
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt not in came_from and valid(nxt):
                    came_from[nxt] = (cur, move)
                    q.append(nxt)

        if goal not in came_from:
            return None

        node = goal
        last_move = None
        while came_from[node] is not None:
            prev, move = came_from[node]
            last_move = move
            node = prev
        return last_move
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
