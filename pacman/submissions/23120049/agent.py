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
# import heapq

# class PacmanAgent(BasePacmanAgent):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.current_path = []
#         self.last_enemy_pos = None
#         self.last_distance = None
#         self.last_stall_step = None
#         self.critical_distance = 5  # threshold for A* switch

#     def dfs_limited(self, start, goal, map_state, max_depth=10):
#         stack = [(start, [], 0)]
#         visited = {start}
#         best_path = []
#         best_distance = self._manhattan_distance(start, goal)

#         while stack:
#             current_pos, path, depth = stack.pop()
#             current_distance = self._manhattan_distance(current_pos, goal)

#             if current_distance < best_distance:
#                 best_distance = current_distance
#                 best_path = path

#             if depth >= max_depth:
#                 continue

#             neighbors = self._get_neighbors(current_pos, map_state)
#             neighbors.sort(key=lambda x: self._manhattan_distance(x[0], goal))

#             for next_pos, move in neighbors:
#                 if next_pos not in visited:
#                     visited.add(next_pos)
#                     stack.append((next_pos, path + [move], depth + 1))

#         return best_path if best_path else [Move.STAY]

#     def astar_path(self, start, goal, map_state):
#         """A* Search for shortest path using Manhattan heuristic."""
#         open_heap = []
#         heapq.heappush(open_heap, (0, start, []))
#         g_score = {start: 0}
#         visited = set()

#         while open_heap:
#             _, current, path = heapq.heappop(open_heap)
#             if current == goal:
#                 return path

#             if current in visited:
#                 continue
#             visited.add(current)

#             for next_pos, move in self._get_neighbors(current, map_state):
#                 tentative_g = g_score[current] + 1
#                 if tentative_g < g_score.get(next_pos, float('inf')):
#                     g_score[next_pos] = tentative_g
#                     f_score = tentative_g + self._manhattan_distance(next_pos, goal)
#                     heapq.heappush(open_heap, (f_score, next_pos, path + [move]))

#         return [Move.STAY]

#     def step(self, map_state, my_position, enemy_position, step_number):
#         dist = self._manhattan_distance(my_position, enemy_position)

#         # --- Adjacent stall logic ---
#         if dist == 1:
#             if (self.last_distance is not None 
#                 and dist >= self.last_distance
#                 and self.last_stall_step != step_number):
#                 self.last_stall_step = step_number
#                 self.current_path = []
#                 self.last_distance = dist
#                 return Move.STAY

#         # --- Algorithm switching logic ---
#         if (not self.current_path or
#             self.last_enemy_pos is None or
#             self._manhattan_distance(enemy_position, self.last_enemy_pos) > 3):

#             if dist <= self.critical_distance:
#                 # Close range: A* for optimal capture
#                 self.current_path = self.astar_path(my_position, enemy_position, map_state)
#             else:
#                 # Far range: limited DFS for exploratory pursuit
#                 self.current_path = self.dfs_limited(my_position, enemy_position, map_state)

#             self.last_enemy_pos = enemy_position

#         # --- Move execution ---
#         move = self.current_path.pop(0) if self.current_path else Move.STAY
#         self.last_distance = dist
#         return move

# import heapq
# from math import copysign

# class PacmanAgent(BasePacmanAgent):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.current_path = []
#         self.last_enemy_pos = None
#         self.last_distance = None
#         self.last_stall_step = None
#         self.critical_distance = 5  # switch to A* when close
#         self.last_enemy_dir = (0, 0)
#         self.los_threshold = 5      # how far Pacman can "see"

#     # ---------------- DFS Limited ----------------
#     def dfs_limited(self, start, goal, map_state, max_depth=10):
#         stack = [(start, [], 0)]
#         visited = {start}
#         best_path = []
#         best_distance = self._manhattan_distance(start, goal)

#         while stack:
#             current_pos, path, depth = stack.pop()
#             current_distance = self._manhattan_distance(current_pos, goal)

#             if current_distance < best_distance:
#                 best_distance = current_distance
#                 best_path = path

#             if depth >= max_depth:
#                 continue

#             neighbors = self._get_neighbors(current_pos, map_state)
#             neighbors.sort(key=lambda x: self._manhattan_distance(x[0], goal))

#             for next_pos, move in neighbors:
#                 if next_pos not in visited:
#                     visited.add(next_pos)
#                     stack.append((next_pos, path + [move], depth + 1))

#         return best_path if best_path else [Move.STAY]

#     # ---------------- A* Search ----------------
#     def astar_path(self, start, goal, map_state):
#         open_heap = []
#         heapq.heappush(open_heap, (0, start, []))
#         g_score = {start: 0}
#         visited = set()

#         while open_heap:
#             _, current, path = heapq.heappop(open_heap)
#             if current == goal:
#                 return path

#             if current in visited:
#                 continue
#             visited.add(current)

#             for next_pos, move in self._get_neighbors(current, map_state):
#                 tentative_g = g_score[current] + 1
#                 if tentative_g < g_score.get(next_pos, float('inf')):
#                     g_score[next_pos] = tentative_g
#                     f_score = tentative_g + self._manhattan_distance(next_pos, goal)
#                     heapq.heappush(open_heap, (f_score, next_pos, path + [move]))

#         return [Move.STAY]

#     # ---------------- LOS Check ----------------
#     def has_line_of_sight(self, start, goal, map_state):
#         """Return True if Pacman can 'see' the ghost (no wall in between)."""
#         x0, y0 = start
#         x1, y1 = goal
#         dx = abs(x1 - x0)
#         dy = abs(y1 - y0)
#         sx = 1 if x0 < x1 else -1
#         sy = 1 if y0 < y1 else -1
#         err = dx - dy

#         while (x0, y0) != (x1, y1):
#             if map_state[y0][x0] == '#':  # wall encountered
#                 return False
#             e2 = 2 * err
#             if e2 > -dy:
#                 err -= dy
#                 x0 += sx
#             if e2 < dx:
#                 err += dx
#                 y0 += sy
#         return True

#     # ---------------- Predictive Intercept ----------------
#     def predict_intercept(self, enemy_pos, enemy_dir, map_state, steps_ahead=3):
#         """Predict ghost's future position or nearest junction in that direction."""
#         x, y = enemy_pos
#         dx, dy = enemy_dir
#         for _ in range(steps_ahead):
#             nx, ny = x + dx, y + dy
#             if not (0 <= ny < len(map_state) and 0 <= nx < len(map_state[0])):
#                 break
#             if map_state[ny][nx] == '#':  # hit a wall, stop
#                 break
#             x, y = nx, ny
#         return (x, y)

#     # ---------------- Step ----------------
#     def step(self, map_state, my_position, enemy_position, step_number):
#         dist = self._manhattan_distance(my_position, enemy_position)

#         # Compute direction ghost moved last step
#         if self.last_enemy_pos:
#             self.last_enemy_dir = (
#                 enemy_position[0] - self.last_enemy_pos[0],
#                 enemy_position[1] - self.last_enemy_pos[1]
#             )

#         # --- Stall logic for 1-tile chase ---
#         if dist == 1:
#             if (self.last_distance is not None 
#                 and dist >= self.last_distance
#                 and self.last_stall_step != step_number):
#                 self.last_stall_step = step_number
#                 self.current_path = []
#                 self.last_distance = dist
#                 return Move.STAY

#         # --- Determine if Pacman can "see" ghost ---
#         los = self.has_line_of_sight(my_position, enemy_position, map_state)

#         # --- Path planning ---
#         if (not self.current_path or
#             self.last_enemy_pos is None or
#             self._manhattan_distance(enemy_position, self.last_enemy_pos) > 3):

#             if los and dist <= self.critical_distance:
#                 # Visible & close → direct A*
#                 target = enemy_position
#                 self.current_path = self.astar_path(my_position, target, map_state)

#             elif not los and dist > self.critical_distance:
#                 # Lost sight → Predict intercept point
#                 predicted = self.predict_intercept(
#                     enemy_position, self.last_enemy_dir, map_state, steps_ahead=3
#                 )
#                 self.current_path = self.astar_path(my_position, predicted, map_state)

#             else:
#                 # Default fallback → DFS exploration
#                 self.current_path = self.dfs_limited(my_position, enemy_position, map_state)

#             self.last_enemy_pos = enemy_position

#         # --- Execute move ---
#         move = self.current_path.pop(0) if self.current_path else Move.STAY
#         self.last_distance = dist
#         return move


import heapq
import collections
import math
import random

# Assumes Move enum exists with .UP, .DOWN, .LEFT, .RIGHT, .STAY and that each has .value = (dr, dc)
# Assumes BasePacmanAgent defines _get_neighbors, _apply_move, _is_valid_position, _manhattan_distance, etc.


import itertools
from heapq import heappush, heappop
from collections import deque

class PacmanAgent(BasePacmanAgent):
    """
    Pacman Agent using Minimax with heat map heuristic
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_positions = []
        self.counter = itertools.count()
    
    # -------------------------------
    # Utilities
    # -------------------------------
    def _is_valid_position(self, pos, map_state):
        row, col = pos
        h, w = map_state.shape
        return 0 <= row < h and 0 <= col < w and map_state[row, col] == 0

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0]+dr, pos[1]+dc)

    def _get_neighbors(self, pos, map_state):
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])
    
    # -------------------------------
    # Predictive Heat Map
    # -------------------------------
    def predictive_heat_map(self, map_state, ghost_pos, depth_limit=5, decay=0.8):
        heat_map = np.zeros_like(map_state, dtype=float)
        height, width = map_state.shape

        # Compute escape routes for all cells
        escape_routes = np.zeros_like(map_state, dtype=int)
        for r in range(height):
            for c in range(width):
                if map_state[r,c]==0:
                    escape_routes[r,c] = self.count_escape_routes((r,c), map_state, max_depth=3)

        # BFS from ghost position
        queue = deque([(ghost_pos, 0)])
        visited = set([ghost_pos])
        while queue:
            pos, step = queue.popleft()
            if step > depth_limit:
                continue

            neighbors = [n for n, _ in self._get_neighbors(pos, map_state)]
            if not neighbors:
                continue
            max_escape = max([escape_routes[n] for n in neighbors])
            best_neighbors = [n for n in neighbors if escape_routes[n] == max_escape]

            for n in best_neighbors:
                heat_map[n] += decay ** step
                if n not in visited:
                    visited.add(n)
                    queue.append((n, step+1))
        return heat_map

    def count_escape_routes(self, pos, map_state, max_depth=3):
        visited = set([pos])
        queue = deque([(pos, 0)])
        count = 0
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for n, _ in self._get_neighbors(current, map_state):
                count += 1
                if n not in visited:
                    visited.add(n)
                    queue.append((n, depth+1))
        return count

    # -------------------------------
    # Minimax with heat map heuristic
    # -------------------------------
    def minimax(self, map_state, pacman_pos, ghost_pos, depth, is_pacman_turn,
                alpha=-float('inf'), beta=float('inf')):
        
        # Leaf node heuristic: use predictive heat map
        if depth == 0 or pacman_pos == ghost_pos:
            heat_map = self.predictive_heat_map(map_state, ghost_pos, depth_limit=3)
            score = heat_map[pacman_pos] - self._manhattan_distance(pacman_pos, ghost_pos)
            return score, None

        if is_pacman_turn:
            max_eval = -float('inf')
            best_move = Move.STAY
            for next_pos, move in self._get_neighbors(pacman_pos, map_state):
                eval_score, _ = self.minimax(map_state, next_pos, ghost_pos,
                                             depth-1, False, alpha, beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = Move.STAY
            for next_pos, move in self._get_neighbors(ghost_pos, map_state):
                eval_score, _ = self.minimax(map_state, pacman_pos, next_pos,
                                             depth-1, True, alpha, beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    # -------------------------------
    # Main step
    # -------------------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        self.previous_positions.append(my_position)
        if len(self.previous_positions) > 10:
            self.previous_positions.pop(0)

        distance = self._manhattan_distance(my_position, enemy_position)
        # Far away: use A* (optional)
        if distance > 6:
            path = self.astar(my_position, enemy_position, map_state)
            if path and path[0] != Move.STAY:
                next_pos = self._apply_move(my_position, path[0])
                if self.previous_positions.count(next_pos) < 3:
                    return path[0]

        # Close: use Minimax with heat map heuristic
        _, best_move = self.minimax(map_state, my_position, enemy_position,
                                    depth=3, is_pacman_turn=True)
        return best_move if best_move else Move.STAY

    # -------------------------------
    # Optional: A* as fallback
    # -------------------------------
    def astar(self, start, goal, map_state):
        def heuristic(pos):
            return self._manhattan_distance(pos, goal)
        frontier = [(0, next(self.counter), start, [])]
        visited = set()
        while frontier:
            f, _, current, path = heappop(frontier)
            if current == goal:
                return path
            if current in visited:
                continue
            visited.add(current)
            for next_pos, move in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    new_path = path + [move]
                    heappush(frontier, (len(new_path) + heuristic(next_pos), next(self.counter), next_pos, new_path))
        return [Move.STAY]

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
    
    def minimax(self, my_pos, enemy_pos, depth, map_state):
        """
        Simplified minimax for simultaneous-move scenario.
        Ghost tries to maximize distance, assuming Pacman moves toward it.
        """
        if depth == 0 or my_pos == enemy_pos:
            return -self._manhattan_distance(my_pos, enemy_pos), Move.STAY

        best_score = float('-inf')
        best_move = Move.STAY

        for next_pos, move in self._get_neighbors(my_pos, map_state):
            # Predict Pacman's next move: move that minimizes distance to ghost
            pacman_moves = self._get_neighbors(enemy_pos, map_state)
            if pacman_moves:
                pacman_next = min(
                    pacman_moves,
                    key=lambda nm: self._manhattan_distance(nm[0], next_pos)
                )[0]
            else:
                pacman_next = enemy_pos

            score, _ = self.minimax(next_pos, pacman_next, depth - 1, map_state)

            if score > best_score:
                best_score = score
                best_move = move

        return best_score, best_move


    def step(self, map_state, my_position, enemy_position, step_number):
        _, best_move = self.minimax(
            my_position, enemy_position,
            depth=3,
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