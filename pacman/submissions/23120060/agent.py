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
        self.ghost_prev_pos = None
    
    def astar(self, start, goal, map_state):
        from heapq import heappush, heappop
        
        W = 1.5   # using Weightened A*
        count = 0       
        g_costs = {start : 0}

        def heuristic(pos):
            # Manhattan distance
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        h_cost_start = heuristic(start)
        f_cost_start = g_costs[start] + (W * h_cost_start)
        
        # using count for avoiding heapq not comparing "path" -> cause error
        prior_queue = [(f_cost_start, count, start, [])] 
        count += 1
        visited = set()
        
        while prior_queue:
            f_cost, _,  current_pos, path = heappop(prior_queue)
            
            if current_pos == goal:
                return path
            
            if current_pos in visited:
                continue    # pass the visited position
            
            visited.add(current_pos)
            
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                new_g_cost = g_costs[current_pos] + 1
                
                if new_g_cost < g_costs.get(next_pos, float('inf')):
                    g_costs[next_pos] = new_g_cost
                    
                    h_cost = heuristic(next_pos)
                    new_f_cost = new_g_cost + (W * h_cost)
                    new_path = path + [move]
                    heappush(prior_queue, (new_f_cost, count, next_pos, new_path))
                    count += 1
                    
        return [Move.STAY]
            
            
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        # predict the pos of ghost
        ghost_velocity = (0,0)
        if self.ghost_prev_pos is not None:
            ghost_velocity = (enemy_position[0] - self.ghost_prev_pos[0],
                              enemy_position[1] - self.ghost_prev_pos[1])
            
        
        PREDICTION_STEPS = 5
        predicted_goal = enemy_position
        current_velocity = ghost_velocity
        
        for _ in range(PREDICTION_STEPS):
            next_pos = (predicted_goal[0] + current_velocity[0],
                        predicted_goal[1] + current_velocity[1])
            
            if not self._is_valid_position(next_pos, map_state):
                break
            
            else:
                predicted_goal = next_pos
        
        
        path = self.astar(my_position, predicted_goal, map_state)
        
        # update the previous ghost position for the next step
        self.ghost_prev_pos = enemy_position
        
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
