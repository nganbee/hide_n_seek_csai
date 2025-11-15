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
import time

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
    
    Uses A* search algorithm to find optimal path to the ghost.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A* Pacman Agent"
        self.current_path = []
        self.last_enemy_pos = None
        self.path_step = 0
        # Cache for valid positions to avoid repeated checks
        self.valid_positions_cache = {}
        
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move using A* search.
        """
        start_time = time.time()
        
        # Clear cache if map changed (shouldn't happen but just in case)
        if len(self.valid_positions_cache) == 0:
            self._precompute_valid_positions(map_state)
        
        # More aggressive replanning conditions for better tracking
        should_replan = (
            not self.current_path or
            self.last_enemy_pos is None or
            enemy_position != self.last_enemy_pos or  # Replan every time enemy moves
            self.path_step >= len(self.current_path)
        )
        
        if should_replan:
            # Use A* to find path to enemy
            self.current_path = self._astar(my_position, enemy_position, map_state)
            self.last_enemy_pos = enemy_position
            self.path_step = 0
        
        # Follow current path
        if self.current_path and self.path_step < len(self.current_path):
            next_move = self.current_path[self.path_step]
            self.path_step += 1
            
            # Verify move is still valid
            if self._is_valid_move_fast(my_position, next_move):
                return next_move
        
        # Fallback: try greedy move
        return self._greedy_move(my_position, enemy_position, map_state)
    
    def _precompute_valid_positions(self, map_state: np.ndarray):
        """Pre-compute all valid positions for faster lookup."""
        height, width = map_state.shape
        self.valid_positions_cache = {}
        
        for row in range(height):
            for col in range(width):
                if map_state[row, col] == 0:  # Not a wall
                    self.valid_positions_cache[(row, col)] = True
    
    def _astar(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """
        A* search algorithm implementation.
        """
        start_time = time.time()
        
        if start == goal:
            return []
        
        # Use binary heap simulation with lists for better performance
        # Format: [f_cost, g_cost, h_cost, position, parent_pos]
        frontier = [[0, 0, self._manhattan_distance(start, goal), start, None]]
        
        # Track best g_cost for each position to avoid exploring worse paths
        g_costs = {start: 0}
        came_from = {}
        closed_set = set()
        
        # Pre-calculate move deltas for speed
        move_deltas = {
            Move.UP: (-1, 0),
            Move.DOWN: (1, 0), 
            Move.LEFT: (0, -1),
            Move.RIGHT: (0, 1)
        }
        
        nodes_explored = 0
        max_nodes = 2000  # Limit nodes to prevent timeout
        
        while frontier and nodes_explored < max_nodes:
            # Check timeout more frequently
            if nodes_explored % 100 == 0 and time.time() - start_time > 0.5:
                break
            
            nodes_explored += 1
            
            # Find and remove node with lowest f_cost
            min_idx = 0
            min_f = frontier[0][0]
            for i in range(1, len(frontier)):
                if frontier[i][0] < min_f:
                    min_f = frontier[i][0]
                    min_idx = i
            
            current_node = frontier.pop(min_idx)
            f_cost, g_cost, h_cost, current_pos, parent_pos = current_node
            
            # Skip if we've found a better path to this position
            if current_pos in closed_set:
                continue
                
            if current_pos in g_costs and g_costs[current_pos] < g_cost:
                continue
            
            closed_set.add(current_pos)
            
            # Found goal - reconstruct path
            if current_pos == goal:
                return self._reconstruct_path(came_from, current_pos, start, move_deltas)
            
            # Explore neighbors
            current_row, current_col = current_pos
            
            for move, (delta_row, delta_col) in move_deltas.items():
                next_row = current_row + delta_row
                next_col = current_col + delta_col
                next_pos = (next_row, next_col)
                
                # Fast validity check using cache
                if not self._is_valid_position_fast(next_pos):
                    continue
                
                if next_pos in closed_set:
                    continue
                
                tentative_g = g_cost + 1
                
                # Skip if we already found a better path to this position
                if next_pos in g_costs and g_costs[next_pos] <= tentative_g:
                    continue
                
                # This is the best path to next_pos so far
                g_costs[next_pos] = tentative_g
                came_from[next_pos] = (current_pos, move)
                
                h_cost = self._manhattan_distance(next_pos, goal)
                f_cost = tentative_g + h_cost
                
                frontier.append([f_cost, tentative_g, h_cost, next_pos, current_pos])
        
        return []  # No path found
    
    def _reconstruct_path(self, came_from: dict, current: tuple, start: tuple, move_deltas: dict) -> list:
        """Path reconstruction."""
        path = []
        
        while current != start:
            if current not in came_from:
                break
            parent_pos, move = came_from[current]
            path.append(move)
            current = parent_pos
        
        path.reverse()
        return path
    
    def _is_valid_position_fast(self, pos: tuple) -> bool:
        """Fast position validity check using cache."""
        return pos in self.valid_positions_cache
    
    def _is_valid_move_fast(self, pos: tuple, move: Move) -> bool:
        """Fast move validity check."""
        delta_row, delta_col = move.value
        next_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position_fast(next_pos)
    
    def _greedy_move(self, my_position: tuple, enemy_position: tuple, map_state: np.ndarray) -> Move:
        """Fallback greedy move towards enemy."""
        row_diff = enemy_position[0] - my_position[0]
        col_diff = enemy_position[1] - my_position[1]
        
        # Prioritize moves by distance reduction
        moves_with_priority = []
        
        if row_diff != 0:
            move = Move.DOWN if row_diff > 0 else Move.UP
            if self._is_valid_move_fast(my_position, move):
                moves_with_priority.append((abs(row_diff), move))
        
        if col_diff != 0:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
            if self._is_valid_move_fast(my_position, move):
                moves_with_priority.append((abs(col_diff), move))
        
        # Sort by priority (larger difference first)
        moves_with_priority.sort(reverse=True)
        
        for priority, move in moves_with_priority:
            return move
        
        # Try any valid move if no direct move available
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move_fast(my_position, move):
                return move
        
        return Move.STAY
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        """Apply a move to a position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        new_pos = self._apply_move(pos, move)
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
    Ghost (Hider) Agent - Goal: Avoid being caught
    Algorithms: BFS 
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        
        #BFS from pacman position to get distance map
        distance_map = self._bfs_full_map(enemy_position, map_state)
        best_move = Move.STAY

        max_distance = distance_map[my_position[0], my_position[1]]

        #evaluate all possible move
        possible_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]

        for move in possible_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)

            if self._is_valid_position(new_pos, map_state):
                new_distance = distance_map[new_pos[0], new_pos[1]]

                if new_distance == -1:
                    new_distance = float('inf')

                if new_distance > max_distance:
                    max_distance = new_distance
                    best_move = move
        #return move
        return best_move
    
    # Helper methods (you can add more)
    
    def _bfs_full_map(self, start_pos:tuple, map_state:np.ndarray) -> np.ndarray:
        """
        find min distance to all position
        - grid[r][c] = distance from start_pos to (r, c)
        - grid[r][c] = -1 if (r, c) is wall or unreachable
        """
        height, width = map_state.shape
        distances = np.full(map_state.shape, -1, dtype=int)
        queue = []

        if self._is_valid_position(start_pos, map_state):
            distances[start_pos[0], start_pos[1]] = 0
            queue.append(start_pos) #add to tail

        while queue:
            current_pos = queue.pop(0) #get head
            current_dist = distances[current_pos[0], current_pos[1]]

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)

                #valid position and not visited
                if self._is_valid_position(next_pos, map_state) and distances[next_pos[0], next_pos[1]] == -1:
                    distances[next_pos[0], next_pos[1]] = current_dist + 1
                    queue.append(next_pos)
        return distances

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
