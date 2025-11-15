"""
Group 2
23120118	Đàm Tiến Đạt
23120116	Nguyễn Việt Cường
23120162	Lê Hải Sơn
"""

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
from heapq import heappush, heappop
from collections import deque


class PacmanAgent(BasePacmanAgent):
    """Pacman (Seeker) using A* (A-star) for shortest paths."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_path = []

    # Helper functions
    def _is_valid_position(self, pos, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _apply_move(self, pos, move: Move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _get_neighbors(self, pos, map_state: np.ndarray):
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._apply_move(pos, move)
            if self._is_valid_position(nxt, map_state):
                neighbors.append((nxt, move))
        return neighbors

    def _manhattan_distance(self, a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _a_star(self, start, goal, map_state: np.ndarray):
        """Find shortest path from start to goal using A* with Manhattan heuristic.
        Returns list of Moves; returns [Move.STAY] if no path exists.
        """
        if start == goal:
            return []

        open_heap = []
        heappush(open_heap, (0, start))

        parent = {}
        g_cost = {start: 0}
        closed = set()

        while open_heap:
            _, cur = heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                # reconstruct path
                moves = []
                node = goal
                while node != start:
                    prev, mv = parent[node]
                    moves.append(mv)
                    node = prev
                moves.reverse()
                return moves
            closed.add(cur)

            for nxt, mv in self._get_neighbors(cur, map_state):
                tentative_g = g_cost[cur] + 1
                if nxt in g_cost and tentative_g >= g_cost[nxt]:
                    continue
                parent[nxt] = (cur, mv)
                g_cost[nxt] = tentative_g
                f = tentative_g + self._manhattan_distance(nxt, goal)
                heappush(open_heap, (f, nxt))

        return [Move.STAY]

    def _bfs_reachable_with_parent(self, start, map_state: np.ndarray):
        queue = deque([start])
        visited = {start}
        parent = {}
        while queue:
            cur = queue.popleft()
            for nxt, mv in self._get_neighbors(cur, map_state):
                if nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = (cur, mv)
                    queue.append(nxt)
        return visited, parent

    def _reconstruct_from_parent(self, parent, start, goal):
        if goal == start:
            return []
        if goal not in parent:
            return [Move.STAY]
        moves = []
        cur = goal
        while cur != start:
            prev, mv = parent[cur]
            moves.append(mv)
            cur = prev
        moves.reverse()
        return moves

    def step(
        self, 
        map_state: np.ndarray, 
        my_position, 
        enemy_position, 
        step_number: int
    ) -> Move:
        path = self._a_star(my_position, enemy_position, map_state)
        if path:
            mv = path[0] if path else Move.STAY
            nxt = self._apply_move(my_position, mv)
            if self._is_valid_position(nxt, map_state):
                self._last_path = path
                return mv

        reachable, parent = self._bfs_reachable_with_parent(my_position, map_state)
        if reachable:
            best_target = min(reachable, key=lambda p: self._manhattan_distance(p, enemy_position))
            path_to_target = self._reconstruct_from_parent(parent, my_position, best_target)
            if path_to_target:
                mv = path_to_target[0]
                nxt = self._apply_move(my_position, mv)
                if self._is_valid_position(nxt, map_state):
                    return mv

        return Move.STAY


class GhostAgent(BaseGhostAgent):
    """Ghost (Hider) Agent - Objective: Evade Pacman using Minimax with Alpha-Beta Pruning."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.miminax_depth = 6
        self.map_size = 21

    def _get_manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_valid_moves(
        self, 
        map_state: np.ndarray, 
        pos: tuple,
    ) -> list:
        """
        Find all valid moves from the current position.
        Returns: [(Move, next_pos)]
        """
        r, c = pos
        valid_moves = []
        
        directions = [
            (0, 0, Move.STAY), 
            (-1, 0, Move.UP), 
            (1, 0, Move.DOWN), 
            (0, -1, Move.LEFT), 
            (0, 1, Move.RIGHT)
        ]

        for dr, dc, move_dir in directions:
            nr, nc = r + dr, c + dc
            next_pos = (nr, nc)
            
            if move_dir == Move.STAY:
                valid_moves.append((Move.STAY, pos))
                continue
            
            height, width = map_state.shape
            if (
                0 <= nr < height and 
                0 <= nc < width and 
                map_state[nr, nc] == 0
            ):
                valid_moves.append((move_dir, next_pos))

        return valid_moves

    def evaluate_state(self, my_pos, enemy_pos):
        """Evaluate a state. Higher score is better for the Ghost."""
        if my_pos == enemy_pos:
            return -100000 
        
        score = self._get_manhattan_distance(my_pos, enemy_pos) * 10
        
        score += abs(my_pos[0] - self.map_size//2) + abs(my_pos[1] - self.map_size//2)
        
        return score

    def minimax(
        self, 
        map_state, 
        ghost_pos, 
        pacman_pos, 
        depth, 
        is_ghost_turn, 
        alpha=float('-inf'), 
        beta=float('inf')
    ):
        # STOP CONDITION: Reached max search depth OR ghost is caught
        if depth == 0 or ghost_pos == pacman_pos:
            return self.evaluate_state(ghost_pos, pacman_pos), Move.STAY
        
        best_move = Move.STAY

        if is_ghost_turn:
            # MAX player - Ghost
            best_value = -float('inf')
            for move, next_ghost_pos in self._get_valid_moves(map_state, ghost_pos):
                value, _ = self.minimax(
                    map_state, 
                    next_ghost_pos, 
                    pacman_pos, 
                    depth - 1,
                    False, 
                    alpha, 
                    beta
                )
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # Beta cut-off

            return best_value, best_move
        else:
            # MIN player - Pacman
            best_value = float('inf')
            best_move = Move.STAY
            for move, next_pacman_pos in self._get_valid_moves(map_state, pacman_pos):
                value, _ = self.minimax(
                    map_state, 
                    ghost_pos, 
                    next_pacman_pos, 
                    depth - 1,
                    True, 
                    alpha, 
                    beta
                )
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # Alpha cut-off
            return best_value, best_move

    def step(self, 
        map_state: np.ndarray, 
        my_position: tuple, 
        enemy_position: tuple,
        step_number: int
    ) -> Move:
        
        best_final_move = Move.STAY
        best_final_score = -float('inf')
        
        possible_moves = self._get_valid_moves(map_state, my_position)
        np.random.shuffle(possible_moves)
        
        for move, next_pos in possible_moves:
            score, _ = self.minimax(
                map_state, 
                next_pos,
                enemy_position,
                self.miminax_depth - 1, 
                is_ghost_turn=False,
                alpha=float('-inf'),
                beta=float('inf')
            )
            if score > best_final_score:
                best_final_score = score
                best_final_move = move
                
        return best_final_move