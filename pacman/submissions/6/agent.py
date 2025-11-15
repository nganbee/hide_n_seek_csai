import sys
from pathlib import Path
from queue import PriorityQueue
import math
from collections import deque, defaultdict

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))
from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random


class PacmanAgent(BasePacmanAgent):
    """
    ProgressiveAlphaBetaPacman (self-contained)
    Includes internal A*, Manhattan, and helper methods.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ProgressiveAlphaBetaPacman"
        self.search_depth = 4
        self.recent_positions = []
        self.last_distance = None
        self.stalemate_count = 0

        # Direction deltas (internal constant)
        self.dir_deltas = {
            Move.UP: (-1, 0),
            Move.DOWN: (1, 0),
            Move.LEFT: (0, -1),
            Move.RIGHT: (0, 1),
            Move.STAY: (0, 0),
        }

    def _manhattan(self, a, b):
        """Heuristic: Manhattan distance."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid(self, pos, map_state):
        """Check if position is inside map and not wall."""
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    def _apply_move(self, pos, move):
        """Apply a move to get new position."""
        dr, dc = self.dir_deltas.get(move, (0, 0))
        return (pos[0] + dr, pos[1] + dc)

    def _neighbors(self, pos, map_state):
        """Return valid neighbor positions and moves."""
        res = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._apply_move(pos, move)
            if self._is_valid(nxt, map_state):
                res.append((nxt, move))
        return res

    def _astar(self, start, goal, map_state):
        """A* pathfinding."""
        if start == goal:
            return [Move.STAY]
        frontier = PriorityQueue()
        frontier.put((self._manhattan(start, goal), start, []))
        g_cost = {start: 0}

        while not frontier.empty():
            _, cur, path = frontier.get()
            if cur == goal:
                return path
            for nxt, mv in self._neighbors(cur, map_state):
                new_g = g_cost[cur] + 1
                if nxt not in g_cost or new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    f = new_g + self._manhattan(nxt, goal)
                    frontier.put((f, nxt, path + [mv]))
        return None

    def evaluate(self, pac, ghost, map_state):
        if pac == ghost:
            return 1e6

        dist = self._manhattan(pac, ghost)
        h, w = map_state.shape

        def degree(pos):
            r, c = pos
            count = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and map_state[rr, cc] == 0:
                    count += 1
            return count

        ghost_deg = degree(ghost)
        pac_deg = degree(pac)

        score = 200 - 10 * dist + (4 - ghost_deg) * 50 - (4 - pac_deg) * 25

        # Penalize near walls
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = pac[0] + dr, pac[1] + dc
            if not (0 <= rr < h and 0 <= cc < w) or map_state[rr, cc] == 1:
                score -= 3
        return score

    def _pac_moves(self, pac, map_state):
        moves = [(pac, Move.STAY)]
        for nxt, mv in self._neighbors(pac, map_state):
            moves.append((nxt, mv))
        return moves

    def _ghost_moves(self, ghost, map_state):
        moves = [(ghost, Move.STAY)]
        for nxt, mv in self._neighbors(ghost, map_state):
            moves.append((nxt, mv))
        return moves

    def alphabeta(self, pac, ghost, map_state, depth, alpha, beta, maxing):
        if pac == ghost:
            return 1e6, Move.STAY
        if depth == 0:
            return self.evaluate(pac, ghost, map_state), Move.STAY

        if maxing:
            best_val, best_move = -math.inf, Move.STAY
            cand = self._pac_moves(pac, map_state)
            cand.sort(key=lambda x: self._manhattan(x[0], ghost))
            for new_pac, mv in cand:
                if new_pac == ghost:
                    return 1e6, mv
                val, _ = self.alphabeta(new_pac, ghost, map_state, depth - 1, alpha, beta, False)
                if val > best_val:
                    best_val, best_move = val, mv
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return best_val, best_move
        else:
            best_val = math.inf
            cand = self._ghost_moves(ghost, map_state)
            cand.sort(key=lambda x: -self._manhattan(x[0], pac))
            for new_ghost, _ in cand:
                val, _ = self.alphabeta(pac, new_ghost, map_state, depth - 1, alpha, beta, True)
                if val < best_val:
                    best_val = val
                beta = min(beta, val)
                if alpha >= beta:
                    break
            return best_val, Move.STAY

    def _ghost_best_moves(self, ghost, pac, map_state):
        best = [Move.STAY]
        best_dist = self._manhattan(ghost, pac)
        for nxt, mv in self._neighbors(ghost, map_state):
            d = self._manhattan(nxt, pac)
            if d > best_dist:
                best = [mv]
                best_dist = d
            elif d == best_dist and mv not in best:
                best.append(mv)
        return best or [Move.STAY]

    def step(self, map_state, my_pos, enemy_pos, step_num):
        dist = self._manhattan(my_pos, enemy_pos)

        # track stalemate
        if self.last_distance is not None:
            if dist == self.last_distance:
                self.stalemate_count += 1
            else:
                self.stalemate_count = 0
        self.last_distance = dist

        # A* fallback if stuck
        if self.stalemate_count >= 3:
            self.recent_positions.clear()
            self.stalemate_count = 0
            path = self._astar(my_pos, enemy_pos, map_state)
            if path:
                return path[0]

        # swap-safe when dist == 1
        if dist == 1:
            ghost_best = self._ghost_best_moves(enemy_pos, my_pos, map_state)
            for gm in ghost_best:
                if self._apply_move(enemy_pos, gm) == my_pos:
                    self._remember(my_pos)
                    return Move.STAY

            # check move into ghost
            move_into_ghost = None
            for nxt, mv in self._neighbors(my_pos, map_state):
                if nxt == enemy_pos:
                    move_into_ghost = mv
                    break

            if move_into_ghost:
                pac_new = self._apply_move(my_pos, move_into_ghost)
                ghost_best2 = self._ghost_best_moves(enemy_pos, pac_new, map_state)
                unsafe = any(self._apply_move(enemy_pos, gm) == my_pos for gm in ghost_best2)
                if not unsafe:
                    self._remember(my_pos)
                    return move_into_ghost

            # fallback stay or reposition
            if my_pos not in self.recent_positions:
                self._remember(my_pos)
                return Move.STAY
            for nxt, mv in self._neighbors(my_pos, map_state):
                if nxt not in self.recent_positions:
                    self._remember(my_pos)
                    return mv
            path = self._astar(my_pos, enemy_pos, map_state)
            if path:
                self._remember(my_pos)
                return path[0]

        # avoid loops
        if my_pos in self.recent_positions:
            for nxt, mv in self._neighbors(my_pos, map_state):
                if nxt not in self.recent_positions:
                    self._remember(my_pos)
                    return mv

        # alpha-beta main
        val, move = self.alphabeta(my_pos, enemy_pos, map_state,
                                   self.search_depth, -math.inf, math.inf, True)
        if move and move != Move.STAY:
            self._remember(my_pos)
            return move

        # fallback A*
        path = self._astar(my_pos, enemy_pos, map_state)
        if path:
            self._remember(my_pos)
            return path[0]

        return Move.STAY

    def _remember(self, pos):
        self.recent_positions.append(pos)
        if len(self.recent_positions) > 5:
            self.recent_positions.pop(0)


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - uses Minimax reasoning
    to predict Pacman's moves and maximize distance.
    """

    def __init__(self, depth=2, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        best_move, _ = self.minimax(
            map_state,
            ghost_pos=my_position,
            pacman_pos=enemy_position,
            depth=self.depth,
            maximizing=True,
        )
        return best_move or Move.STAY

    def minimax(self, map_state, ghost_pos, pacman_pos, depth, maximizing):
        """Return (best_move, score)"""
        if depth == 0:
            return None, self.evaluate_state(ghost_pos, pacman_pos)

        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        valid_moves = []
        for move in moves:
            new_pos = self.apply_move(ghost_pos if maximizing else pacman_pos, move)
            if self.is_valid(new_pos, map_state):
                valid_moves.append((move, new_pos))

        if not valid_moves:
            return None, self.evaluate_state(ghost_pos, pacman_pos)

        if maximizing:
            # Ghost’s turn — maximize distance
            max_eval = -float("inf")
            best_move = None
            for move, new_pos in valid_moves:
                _, eval_score = self.minimax(
                    map_state, new_pos, pacman_pos, depth - 1, False
                )
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            return best_move, max_eval

        else:
            # Pacman’s turn — minimize distance
            min_eval = float("inf")
            best_move = None
            for move, new_pos in valid_moves:
                _, eval_score = self.minimax(
                    map_state, ghost_pos, new_pos, depth - 1, True
                )
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
            return best_move, min_eval


    def evaluate_state(self, ghost_pos, pacman_pos):
        """Higher score if Ghost is far away from Pacman."""
        dist = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        return dist

    def apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def is_valid(self, pos, map_state):
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0