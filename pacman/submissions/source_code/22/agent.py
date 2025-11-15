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
import heapq
import time

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from collections import deque
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """
    Enhanced Pacman Agent with:
    - Advanced A* with better heuristic
    - Ghost movement prediction
    - Corridor optimization
    - Dead-end avoidance
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = []
        self.name = "Enhanced A* Hunter"
        self.last_enemy_pos = None
        self.enemy_velocity = (0, 0)
        self.dead_ends = set()
        self.map_analyzed = False
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Enhanced decision making with ghost prediction.
        """
        # Analyze map on first step
        if not self.map_analyzed:
            self._analyze_map(map_state)
            self.map_analyzed = True
        if self.last_enemy_pos:
            self.enemy_velocity = (
                enemy_position[0] - self.last_enemy_pos[0],
                enemy_position[1] - self.last_enemy_pos[1]
            )
        self.last_enemy_pos = enemy_position
        predicted_pos = self._predict_enemy_position(enemy_position, map_state)
        distance = abs(my_position[0] - enemy_position[0]) + abs(my_position[1] - enemy_position[1])
        
        if distance <= 3:
            target = enemy_position
        else:
            target = predicted_pos
        
        if self.path and len(self.path) > 0:
            if self._should_recalculate_path(enemy_position, target):
                self.path = []
        
        if not self.path:
            self.path = self._advanced_a_star(my_position, target, map_state, enemy_position)
        
        if self.path and len(self.path) > 0:
            next_pos = self.path[0]
            if self._is_valid_position(next_pos, map_state):
                move = self._get_move_to_position(my_position, next_pos)
                if move and self._is_valid_move(my_position, move, map_state):
                    self.path.pop(0)
                    return move
    
        best_move = self._get_best_tactical_move(my_position, enemy_position, map_state)
        if best_move:
            return best_move
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                return move
        
        return Move.STAY
    
    def _analyze_map(self, map_state: np.ndarray):
        """Analyze map to find dead ends and important features."""
        height, width = map_state.shape
        
        for row in range(height):
            for col in range(width):
                if map_state[row, col] == 0:  # Empty cell
                    neighbors = 0
                    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                        if self._is_valid_move((row, col), move, map_state):
                            neighbors += 1
                    
                    if neighbors == 1:
                        self.dead_ends.add((row, col))
    
    def _predict_enemy_position(self, enemy_pos: tuple, map_state: np.ndarray) -> tuple:
        """Predict where ghost will move based on velocity and map."""
        if self.enemy_velocity == (0, 0):
            return enemy_pos
        
        predicted = enemy_pos
        for _ in range(2):
            next_row = predicted[0] + self.enemy_velocity[0]
            next_col = predicted[1] + self.enemy_velocity[1]
            next_pos = (next_row, next_col)
            
            if self._is_valid_position(next_pos, map_state):
                predicted = next_pos
            else:
                break
        
        return predicted
    
    def _should_recalculate_path(self, current_enemy_pos: tuple, target: tuple) -> bool:
        """Decide if path needs recalculation."""
        if not self.path:
            return True
        
        if len(self.path) > 10:
            return True
        
        if self.last_enemy_pos and current_enemy_pos != self.last_enemy_pos:
            dist_moved = abs(current_enemy_pos[0] - self.last_enemy_pos[0]) + \
                        abs(current_enemy_pos[1] - self.last_enemy_pos[1])
            if dist_moved > 1:
                return True
        
        return False
    
    def _advanced_a_star(self, start: tuple, goal: tuple, 
                        map_state: np.ndarray, enemy_pos: tuple) -> list:
        """
        Enhanced A* with:
        - Better heuristic considering corridors
        - Dead-end penalty
        - Straight path bonus
        """
        def heuristic(pos):
            h = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            
            if pos in self.dead_ends:
                h += 20
            enemy_to_goal = abs(enemy_pos[0] - goal[0]) + abs(enemy_pos[1] - goal[1])
            if enemy_to_goal < 5:
                h += abs(pos[0] - enemy_pos[0]) + abs(pos[1] - enemy_pos[1]) * 0.5
            
            return h
        
        def get_direction_bonus(current, next_pos, prev_pos):
            """Bonus for continuing in same direction."""
            if prev_pos is None:
                return 0
            
            prev_dir = (current[0] - prev_pos[0], current[1] - prev_pos[1])
            next_dir = (next_pos[0] - current[0], next_pos[1] - current[1])
            
            if prev_dir == next_dir:
                return -0.5  # Bonus (reduce cost)
            return 0.5  # Penalty for turning
        
        counter = 0
        pq = [(heuristic(start), counter, 0, start, [], None)]
        visited = {}
        
        while pq:
            f, _, g, current, path, prev_pos = heapq.heappop(pq)
            
            if current in visited and visited[current] <= g:
                continue
            
            visited[current] = g
            
            if current == goal:
                return path
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(current, move, map_state):
                    delta_row, delta_col = move.value
                    neighbor = (current[0] + delta_row, current[1] + delta_col)
                    
                    if neighbor not in visited or visited[neighbor] > g + 1:
                        new_path = path + [neighbor]
                        new_g = g + 1 + get_direction_bonus(current, neighbor, prev_pos)
                        new_f = new_g + heuristic(neighbor)
                        
                        counter += 1
                        heapq.heappush(pq, (new_f, counter, new_g, neighbor, new_path, current))
        
        return []
    
    def _get_best_tactical_move(self, my_pos: tuple, enemy_pos: tuple, 
                                map_state: np.ndarray) -> Move:
        """Choose best move when no path available."""
        best_move = None
        best_score = float('-inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move, map_state):
                delta_row, delta_col = move.value
                new_pos = (my_pos[0] + delta_row, my_pos[1] + delta_col)
                
                # 1. Distance to enemy (lower is better)
                dist = abs(new_pos[0] - enemy_pos[0]) + abs(new_pos[1] - enemy_pos[1])
                score = -dist
                
                # 2. Avoid dead ends
                if new_pos in self.dead_ends:
                    score -= 50
                
                # 3. Prefer open spaces
                neighbors = sum(1 for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
                              if self._is_valid_move(new_pos, m, map_state))
                score += neighbors * 2
                
                if score > best_score:
                    best_score = score
                    best_move = move
        
        return best_move
    
    def _get_move_to_position(self, current: tuple, target: tuple) -> Move:
        """Convert position difference to Move."""
        row_diff = target[0] - current[0]
        col_diff = target[1] - current[1]
        
        if row_diff == -1:
            return Move.UP
        elif row_diff == 1:
            return Move.DOWN
        elif col_diff == -1:
            return Move.LEFT
        elif col_diff == 1:
            return Move.RIGHT
        return None
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if move is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if position is valid."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0



class GhostAgent(BaseGhostAgent):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Aggressive Ghost"
        
    def step(self, map_state, my_position, enemy_position, step_number):
        manhattan_dist = self._manhattan_distance(my_position, enemy_position)

        num_exits = len(self._get_valid_non_stay_neighbors(my_position, map_state))

        if num_exits > 1 and manhattan_dist == 1:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._apply_move(my_position, move) == enemy_position:
                    return move

        
        if num_exits > 2 and manhattan_dist > 5:
            return Move.STAY


        target_pos = self.find_best_escape_target(my_position, enemy_position, map_state)
        path = self.bfs(my_position, target_pos, map_state)

        if path and path != [Move.STAY]:
            return path[0]


        neighbors = self._get_valid_non_stay_neighbors(my_position, map_state)
        if neighbors:
            return random.choice(neighbors)[1]

        return Move.STAY


    # Support Function
    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_valid_non_stay_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors

    def bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list[Move]:
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current_pos, path = queue.popleft()
            if current_pos == goal:
                return path

            for next_pos, move in self._get_valid_non_stay_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))

        return [Move.STAY]

    def find_best_escape_target(self, my_position, enemy_pos, map_state):
        queue = deque([(enemy_pos, 0)])
        visited = {enemy_pos: 0}

        while queue:
            current, dist = queue.popleft()
            for next_pos, move in self._get_valid_non_stay_neighbors(current, map_state):
                if next_pos not in visited:
                    visited[next_pos] = dist + 1
                    queue.append((next_pos, dist + 1))

        max_dist = -1
        best_positions = []

        for pos, dist in visited.items():
            if dist > max_dist:
                max_dist = dist
                best_positions = [pos]
            elif dist == max_dist:
                best_positions.append(pos)

        if not best_positions:
            return my_position

        best_target = min(best_positions, key=lambda p: self._manhattan_distance(my_position, p))
        return best_target