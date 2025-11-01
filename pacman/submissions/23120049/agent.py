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

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # path is list of Move enums for upcoming steps
        self.current_path = []
        self.last_enemy_pos = None
        self.last_distance = None
        self.last_stall_step = None
        self.last_enemy_dir = (0, 0)

        # Critical parameters
        self.critical_distance = 5      # <= this -> critical chasing
        self.intercept_steps_ahead = 2  # how many tiles ahead to predict for A*
        self.guard_ticks = 3            # how many ticks to guard an intercept node

        # Junction graph caches (built on first use)
        self._junction_nodes = None            # set of node tile coords (r,c)
        self._node_edges = None                # dict: node -> list of (neighbor_node, length, corridor_tiles)
        self._tile_to_node = None              # dict tile -> node (tile may map to itself)
        self._node_guard_timeout = {}          # node -> remaining guard ticks when we arrive
        self._last_graph_map_signature = None  # to detect if map changed (optional)

        # small cache for LOS per step to avoid repeated Bresenham calls
        self._los_cache = {"step": None, "value": None, "key": None}

    # ------------------ Utility: is_wall ------------------
    def _is_wall(self, pos, map_state):
        """
        Robust wall check: supports map_state with 0/1 or '.'/'#' or numpy arrays.
        pos = (r, c)
        """
        r, c = pos
        try:
            cell = map_state[r, c]
        except Exception:
            try:
                cell = map_state[r][c]
            except Exception:
                return True
        # interpret
        if cell == 1 or cell == True:
            return True
        if isinstance(cell, str):
            return cell == '#' or cell == '1'
        # numpy might have dtype 'int' or 'uint8'
        try:
            if int(cell) == 1:
                return True
        except Exception:
            pass
        return False

    # ------------------ LOS: Bresenham with cache ------------------
    def has_line_of_sight(self, a_pos, b_pos, map_state, step_number=None):
        """
        Bresenham line between a_pos and b_pos (tile coords). Returns True if no wall in between.
        Adds a tiny cache per step when step_number provided.
        """
        key = (a_pos, b_pos)
        if step_number is not None:
            cached_key = self._los_cache.get("key")
            if self._los_cache.get("step") == step_number and cached_key == key:
                return self._los_cache.get("value")

        x0, y0 = a_pos[1], a_pos[0]  # convert (row,col)->(x,y) for algorithm convenience
        x1, y1 = b_pos[1], b_pos[0]
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cx, cy = x0, y0
        visible = True
        while True:
            # convert back to (row, col)
            rr, cc = cy, cx
            if self._is_wall((rr, cc), map_state):
                visible = False
                break
            if cx == x1 and cy == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

        if step_number is not None:
            self._los_cache["step"] = step_number
            self._los_cache["key"] = key
            self._los_cache["value"] = visible

        return visible

    # ------------------ DFS limited (keeps original heuristic) ------------------
    def dfs_limited(self, start, goal, map_state, max_depth=10, prefer_dir=None):
        """
        Original DFS-limited with optional prefer_dir (dx,dy) to bias neighbor ordering.
        start, goal are (row,col)
        Returns list of Move enums or [Move.STAY]
        """
        stack = [(start, [], 0)]
        visited = {start}
        best_path = []
        best_distance = self._manhattan_distance(start, goal)

        while stack:
            current_pos, path, depth = stack.pop()
            current_distance = self._manhattan_distance(current_pos, goal)

            if current_distance < best_distance:
                best_distance = current_distance
                best_path = path

            if depth >= max_depth:
                continue

            neighbors = self._get_neighbors(current_pos, map_state)

            if prefer_dir:
                # prefer moves aligned with prefer_dir (dot product)
                def score(n):
                    npos, _ = n
                    dr = npos[0] - current_pos[0]
                    dc = npos[1] - current_pos[1]
                    return - (dr * prefer_dir[0] + dc * prefer_dir[1])
                neighbors.sort(key=score)
            else:
                neighbors.sort(key=lambda x: self._manhattan_distance(x[0], goal))

            for next_pos, move in neighbors:
                if next_pos not in visited:
                    visited.add(next_pos)
                    stack.append((next_pos, path + [move], depth + 1))

        return best_path if best_path else [Move.STAY]

    # ------------------ A* (tile-level) ------------------
    def astar_path(self, start, goal, map_state):
        """
        Tile-level A* returning list of Move enums. start/goal are (row,col).
        """
        open_heap = []
        heapq.heappush(open_heap, (0, start, []))
        g_score = {start: 0}
        visited = set()

        while open_heap:
            _, current, path = heapq.heappop(open_heap)
            if current == goal:
                return path

            if current in visited:
                continue
            visited.add(current)

            for next_pos, move in self._get_neighbors(current, map_state):
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(next_pos, float('inf')):
                    g_score[next_pos] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(next_pos, goal)
                    heapq.heappush(open_heap, (f_score, next_pos, path + [move]))

        return [Move.STAY]

    # ------------------ Junction graph builder ------------------
    def _build_junction_graph_if_needed(self, map_state):
        """
        Build junction nodes and corridor edges if not built yet.
        Node definition: any free tile whose walk-degree != 2 (dead-end or junction) OR endpoints of long corridors.
        Edges connect nodes along straight corridors; we store corridor tile lists for path expansion.
        """
        # rebuild every time map changes might be heavy; assume static map in this environment
        if self._junction_nodes is not None:
            return

        h = len(map_state)
        w = len(map_state[0]) if h > 0 else 0
        nodes = set()
        # scan all tiles, choose nodes where degree != 2
        for r in range(h):
            for c in range(w):
                if self._is_wall((r, c), map_state):
                    continue
                # compute degree (valid neighbor count)
                deg = 0
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and not self._is_wall((nr, nc), map_state):
                        deg += 1
                if deg != 2:
                    nodes.add((r, c))

        # also ensure endpoints of corridors of length > 1 are nodes (already covered by deg != 2),
        # but to be safe, include immediate neighbor nodes.
        edges = {}       # node -> list of (neighbor_node, length, corridor_tiles_in_order_including_end)
        tile_to_node = {}

        for node in nodes:
            edges[node] = []

        # for each node, walk in 4 directions to find next node
        for node in list(nodes):
            r, c = node
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                path = []
                cr, cc = r, c
                length = 0
                while True:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        break
                    if self._is_wall((nr, nc), map_state):
                        break
                    length += 1
                    path.append((nr, nc))
                    cr, cc = nr, nc
                    if (nr, nc) in nodes:
                        # found neighbor node at end of corridor
                        neighbor = (nr, nc)
                        # corridor tiles from node->neighbor (excluding starting node, including neighbor)
                        corridor_tiles = list(path)
                        # record edge (avoid duplicates by only adding if neighbor not already recorded)
                        # but we'll append; duplicates will be handled by pathfinding tolerance
                        edges[node].append((neighbor, length, corridor_tiles))
                        break
                    # continue walking

        # Build tile_to_node map: each node maps to itself; corridor tiles map to nearest node endpoint for reconstruction
        tile_to_node = {}
        for node, neighs in edges.items():
            tile_to_node[node] = node
            for neighbor, length, corridor_tiles in neighs:
                # map corridor tiles to the two endpoints as needed; we map each tile to the closer endpoint for quick lookup
                for idx, tile in enumerate(corridor_tiles):
                    # distance to node is idx+1, to neighbor is length - idx
                    dist_to_node = idx + 1
                    dist_to_neighbor = length - idx
                    tile_to_node[tile] = node if dist_to_node <= dist_to_neighbor else neighbor

        self._junction_nodes = nodes
        self._node_edges = edges
        self._tile_to_node = tile_to_node

    # ------------------ Walk from tile along direction until next junction node ------------------
    def _walk_to_next_node(self, start_tile, direction, map_state):
        """
        Walk from start_tile in integer direction (dr,dc) until hitting a node in junction graph or wall.
        Returns the node tile if found, or None.
        """
        self._build_junction_graph_if_needed(map_state)
        if not direction or (direction[0] == 0 and direction[1] == 0):
            return None
        r, c = start_tile
        dr, dc = direction
        h = len(map_state); w = len(map_state[0]) if h>0 else 0
        cr, cc = r, c
        while True:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                return None
            if self._is_wall((nr, nc), map_state):
                return None
            cr, cc = nr, nc
            if (cr, cc) in self._junction_nodes:
                return (cr, cc)
        # unreachable

    # ------------------ Node-level dijkstra ------------------
    def _dijkstra_nodes(self, start_node, goal_node):
        """
        Dijkstra on node graph, returns list of nodes from start_node to goal_node inclusive.
        Uses self._node_edges (neighbor_node, length, corridor_tiles)
        """
        if start_node == goal_node:
            return [start_node]

        pq = []
        heapq.heappush(pq, (0, start_node, [start_node]))
        best_cost = {start_node: 0}
        visited = set()
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == goal_node:
                return path
            if node in visited:
                continue
            visited.add(node)
            for (nbr, length, corridor_tiles) in self._node_edges.get(node, []):
                ncost = cost + length
                if ncost < best_cost.get(nbr, float('inf')):
                    best_cost[nbr] = ncost
                    heapq.heappush(pq, (ncost, nbr, path + [nbr]))
        return None

    # ------------------ Convert node path to tile-level Move list ------------------
    def _nodes_to_tile_moves(self, start_tile, node_path, map_state):
        """
        Convert node_path (list of nodes) into sequence of Move enums from start_tile to reach node_path[-1].
        We expand along corridors using stored corridor tiles in self._node_edges.
        start_tile might be not exactly at node_path[0], so we first A* from start_tile to node_path[0] if needed.
        """
        moves = []

        # helper to get Move from delta
        delta_to_move = {
            (-1,0): Move.UP,
            (1,0): Move.DOWN,
            (0,-1): Move.LEFT,
            (0,1): Move.RIGHT
        }

        # if start not at first node, compute tile-level A* to that node
        first_node = node_path[0]
        if start_tile != first_node:
            path_to_first = self.astar_path(start_tile, first_node, map_state)
            if path_to_first and path_to_first != [Move.STAY]:
                moves.extend(path_to_first)
            else:
                # fallback: if cannot reach, return empty
                return []

        # expand node-by-node
        for i in range(len(node_path)-1):
            a = node_path[i]
            b = node_path[i+1]
            # find corridor tiles from a->b in edges
            corridor = None
            for nbr, length, corridor_tiles in self._node_edges.get(a, []):
                if nbr == b:
                    corridor = corridor_tiles
                    break
            if corridor is None:
                # maybe path stored reverse; try from b->a and reverse
                for nbr, length, corridor_tiles in self._node_edges.get(b, []):
                    if nbr == a:
                        corridor = list(reversed(corridor_tiles))
                        break
            if corridor is None:
                # last resort: A* between a and b
                moves_between = self.astar_path(a, b, map_state)
                if moves_between and moves_between != [Move.STAY]:
                    moves.extend(moves_between)
                else:
                    return []  # failed
            else:
                # corridor is list of tiles from a's adjacent to endpoint b inclusive
                cur = a
                for tile in corridor:
                    dr = tile[0] - cur[0]
                    dc = tile[1] - cur[1]
                    mv = delta_to_move.get((dr,dc))
                    if mv is None:
                        # something odd; fallback to A*
                        moves_between = self.astar_path(cur, tile, map_state)
                        if moves_between and moves_between != [Move.STAY]:
                            moves.extend(moves_between)
                        else:
                            return []
                        cur = tile
                    else:
                        moves.append(mv)
                        cur = tile
        return moves

    # ------------------ Predict enemy tile (k steps ahead) ------------------
    def _predict_enemy_tile(self, enemy_pos, map_state, k=2):
        """
        Predict enemy tile k steps ahead using last_enemy_dir; fallback to enemy_pos if blocked or unknown.
        """
        if not self.last_enemy_pos:
            return enemy_pos
        dx = enemy_pos[0] - self.last_enemy_pos[0]
        dy = enemy_pos[1] - self.last_enemy_pos[1]
        if dx == 0 and dy == 0:
            dx, dy = self.last_enemy_dir
        # try k..1
        h = len(map_state); w = len(map_state[0]) if h>0 else 0
        for step_ahead in range(k, 0, -1):
            pr = enemy_pos[0] + dx * step_ahead
            pc = enemy_pos[1] + dy * step_ahead
            if 0 <= pr < h and 0 <= pc < w and not self._is_wall((pr, pc), map_state):
                return (pr, pc)
        return enemy_pos

    # ------------------ Main step() ------------------
    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Main decision function:
        - If far: use dfs_limited (original heuristic).
        - If within critical_distance:
            - If LOS: use predictive A* (chase predicted enemy tile)
            - Else: compute predicted junction node in enemy direction, route to that node via node graph and guard.
        """
        # update enemy direction history
        if self.last_enemy_pos:
            self.last_enemy_dir = (
                enemy_position[0] - self.last_enemy_pos[0],
                enemy_position[1] - self.last_enemy_pos[1]
            )

        dist = self._manhattan_distance(my_position, enemy_position)

        # stall logic when adjacent (avoid swap)
        if dist == 1:
            if (self.last_distance is not None and dist >= self.last_distance and self.last_stall_step != step_number):
                self.last_stall_step = step_number
                self.current_path = []
                self.last_distance = dist
                self.last_enemy_pos = enemy_position
                return Move.STAY

        # if need rebuild graph (lazy)
        self._build_junction_graph_if_needed(map_state)

        # LOS check (cached per step)
        los = self.has_line_of_sight(my_position, enemy_position, map_state, step_number=step_number)

        replan_needed = (not self.current_path or
                         self.last_enemy_pos is None or
                         self._manhattan_distance(enemy_position, self.last_enemy_pos) > 3)

        # If in critical range -> special behavior
        if dist <= self.critical_distance:
            if replan_needed:
                if los:
                    # Predict enemy a bit ahead and A* there (predictive A*)
                    predicted_tile = self._predict_enemy_tile(enemy_position, map_state, k=self.intercept_steps_ahead)
                    self.current_path = self.astar_path(my_position, predicted_tile, map_state)
                else:
                    # LOST SIGHT but close: try to intercept at next junction node
                    predicted_node = self._walk_to_next_node(enemy_position, self.last_enemy_dir, map_state)
                    # fallback: if cannot find node or predicted node not available, do predictive DFS bias
                    if predicted_node is None:
                        # fallback to predictive DFS (prefer direction)
                        prefer_dir = self.last_enemy_dir if self.last_enemy_dir != (0,0) else None
                        self.current_path = self.dfs_limited(my_position, enemy_position, map_state, max_depth=10, prefer_dir=prefer_dir)
                    else:
                        # compute node path
                        pac_node = self._tile_to_node.get(my_position, None)
                        if pac_node is None:
                            # map pacman's tile to nearest node
                            pac_node = self._tile_to_node.get(my_position, None)
                        # If pac_node still None, pick nearest node by simple BFS to node set
                        if pac_node is None:
                            # BFS until reach any node
                            from collections import deque
                            q = deque([(my_position, [])])
                            vis = {my_position}
                            found_node = None
                            while q and found_node is None:
                                cur, pth = q.popleft()
                                if cur in self._junction_nodes:
                                    found_node = cur
                                    break
                                for nxt, _ in self._get_neighbors(cur, map_state):
                                    if nxt not in vis:
                                        vis.add(nxt)
                                        q.append((nxt, pth + [nxt]))
                            pac_node = found_node if found_node is not None else predicted_node

                        node_path = self._dijkstra_nodes(pac_node, predicted_node)
                        if not node_path:
                            # if dijkstra failed, fallback to predictive DFS
                            prefer_dir = self.last_enemy_dir if self.last_enemy_dir != (0,0) else None
                            self.current_path = self.dfs_limited(my_position, enemy_position, map_state, max_depth=10, prefer_dir=prefer_dir)
                        else:
                            # convert node path to tile-level moves
                            tile_moves = self._nodes_to_tile_moves(my_position, node_path, map_state)
                            if tile_moves:
                                self.current_path = tile_moves
                                # set guard timeout for predicted_node (when we arrive, we may guard)
                                self._node_guard_timeout[predicted_node] = self.guard_ticks
                            else:
                                # fallback to predictive DFS
                                prefer_dir = self.last_enemy_dir if self.last_enemy_dir != (0,0) else None
                                self.current_path = self.dfs_limited(my_position, enemy_position, map_state, max_depth=10, prefer_dir=prefer_dir)
                self.last_enemy_pos = enemy_position
        else:
            # Not critical: normal DFS-limited behavior (keep original heuristic)
            if replan_needed:
                self.current_path = self.dfs_limited(my_position, enemy_position, map_state)
                self.last_enemy_pos = enemy_position

        # If we have guard timeout at current tile's node and we are at that node, guard (stay) until timeout expires
        cur_tile = my_position
        cur_node = self._tile_to_node.get(cur_tile) if self._tile_to_node is not None else None
        if cur_node and self._node_guard_timeout.get(cur_node, 0) > 0:
            # decrement timeout and stay
            self._node_guard_timeout[cur_node] -= 1
            # also clear current_path to force replanning next tick
            self.current_path = []
            self.last_distance = dist
            return Move.STAY

        # Small random jitter occasionally to break symmetry
        if random.random() < 0.05 and (not self.current_path or random.random() < 0.5):
            # slight replanning jitter
            if dist <= self.critical_distance and not los:
                prefer_dir = self.last_enemy_dir if self.last_enemy_dir != (0,0) else None
                self.current_path = self.dfs_limited(my_position, enemy_position, map_state, max_depth=8, prefer_dir=prefer_dir)
            else:
                self.current_path = self.dfs_limited(my_position, enemy_position, map_state, max_depth=8)

        # Execute move
        if self.current_path:
            next_move = self.current_path.pop(0)
        else:
            next_move = Move.STAY

        self.last_distance = dist
        return next_move

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