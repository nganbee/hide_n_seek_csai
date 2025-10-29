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
        self.name = "A star Pacman"
    
    def astar(self, start, goal, map_state):
        from heapq import heappush, heappop
        
        def heuristic(pos):
            # Manhattan distance
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        prior_queue = [(0, start, [])]
        visited = set()
        
        while prior_queue:
            f_cost, current_pos, path = heappop(prior_queue)
            
            if current_pos == goal:
                return path
            
            if current_pos in visited:
                continue    # bỏ qua các nơi đã đi đến
            
            visited.add(current_pos)
            
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    new_path = path + [move]
                    g_cost = len(new_path)
                    h_cost = heuristic(next_pos)
                    f_cost = g_cost + h_cost
                    heappush(prior_queue, (f_cost, next_pos, new_path))
                    
        return [Move.STAY]
            
            
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        path = self.astar(my_position, enemy_position, map_state)
        if path:
            return path[0]
        
        return Move.STAY
    
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
        
        # Example: Simple evasive approach (replace with your algorithm)
        row_diff = my_position[0] - enemy_position[0]
        col_diff = my_position[1] - enemy_position[1]
        
        # Try to move away from Pacman
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
