import sys
from pathlib import Path
from collections import deque
import heapq
from functools import lru_cache
from typing import List, Tuple, Dict, Optional
import random
import math

# Add src to path to import the interface (adjust path as needed)
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


Pos = Tuple[int, int]


class _SharedUtils:
    @staticmethod
    def manhattan(a: Pos, b: Pos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def in_bounds(pos: Pos, map_state: np.ndarray) -> bool:
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w

    @staticmethod
    def is_free(pos: Pos, map_state: np.ndarray) -> bool:
        return _SharedUtils.in_bounds(pos, map_state) and map_state[pos[0], pos[1]] == 0

    @staticmethod
    def get_neighbors(pos: Pos, map_state: np.ndarray, include_stay: bool = False) -> List[Pos]:
        nbrs = []
        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            dr, dc = mv.value
            np_ = (pos[0] + dr, pos[1] + dc)
            if _SharedUtils.is_free(np_, map_state):
                nbrs.append(np_)
        if include_stay and _SharedUtils.is_free(pos, map_state):
            nbrs.append(pos)
        return nbrs

    @staticmethod
    def pos_from_move(pos: Pos, move: Move) -> Pos:
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent using optimized A* with loop detection and simple stuck handling.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A* Hunter (optimized + loop-break)"
        self.path: List[Pos] = []
        self.ghost_history: List[Pos] = []
        # recent states to detect loops: store tuples (pacman_pos, ghost_pos)
        self.recent_states = deque(maxlen=8)
        # random seed for reproducibility if desired
        random.seed()

    def step(self, map_state: np.ndarray,
             my_position: Pos,
             enemy_position: Pos,
             step_number: int) -> Move:

        # record ghost history for simple heuristics
        self.ghost_history.append(enemy_position)

        # Detect repeated state and break loop early (before recomputing heavy A*)
        state = (my_position, enemy_position)
        if state in self.recent_states:
            # loop detected -> try random valid move to break symmetry
            valid_moves = [mv for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT) if self._is_valid_move(my_position, mv, map_state)]
            if valid_moves:
                return random.choice(valid_moves)
            # if none, try stay
            return Move.STAY

        # append current state
        self.recent_states.append(state)

        # Predict ghost short-horizon
        predicted = self._predict_ghost_moves(enemy_position, my_position, map_state, num_moves=3)
        target = predicted[0] if predicted else enemy_position

        # If existing path is still valid, follow it
        if self.path:
            # check next step still free
            if self.path:
                next_pos = self.path[0]
                if _SharedUtils.is_free(next_pos, map_state):
                    move = self._get_move_direction(my_position, next_pos)
                    if move and self._is_valid_move(my_position, move, map_state):
                        self.path.pop(0)
                        return move
                else:
                    # path invalidated
                    self.path = []

        # Recompute path with A*
        self.path = self._astar_search(my_position, target, map_state)

        if self.path:
            next_pos = self.path.pop(0)
            move = self._get_move_direction(my_position, next_pos)
            if move and self._is_valid_move(my_position, move, map_state):
                return move

        # If ghost is stuck (seen same position multiple times), avoid rushing in
        if len(self.ghost_history) >= 3 and len(set(self.ghost_history[-3:])) == 1:
            # pick a safe move (do not approach ghost)
            safe_moves = []
            for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                if self._is_valid_move(my_position, mv, map_state):
                    new_pos = _SharedUtils.pos_from_move(my_position, mv)
                    # prefer moves that don't reduce distance to ghost
                    if _SharedUtils.manhattan(new_pos, enemy_position) >= _SharedUtils.manhattan(my_position, enemy_position):
                        safe_moves.append(mv)
            if safe_moves:
                return random.choice(safe_moves)

        # fallback: any valid move
        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            if self._is_valid_move(my_position, mv, map_state):
                return mv

        return Move.STAY

    def _astar_search(self, start: Pos, goal: Pos, map_state: np.ndarray) -> List[Pos]:
        """
        Standard A* with g_score and came_from. Returns list excluding start.
        """
        if start == goal:
            return []

        open_heap = []
        g_score: Dict[Pos, int] = {start: 0}
        f_score_start = _SharedUtils.manhattan(start, goal)
        heapq.heappush(open_heap, (f_score_start, 0, start))
        came_from: Dict[Pos, Pos] = {}
        closed = set()
        tie = 0

        while open_heap:
            f, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                # reconstruct path
                path = []
                node = current
                while node != start:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path

            closed.add(current)
            current_g = g_score.get(current, math.inf)

            for neighbor in _SharedUtils.get_neighbors(current, map_state, include_stay=False):
                tentative_g = current_g + 1
                if tentative_g < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_neighbor = tentative_g + _SharedUtils.manhattan(neighbor, goal)
                    tie += 1
                    heapq.heappush(open_heap, (f_neighbor, tie, neighbor))

        return []

    def _predict_ghost_moves(self, ghost_pos: Pos, pacman_pos: Pos, map_state: np.ndarray, num_moves: int = 3) -> List[Pos]:
        """
        Lightweight predictor: prefer moving away and to positions with higher mobility.
        """
        predicted = []
        g_pos = ghost_pos
        p_pos = pacman_pos

        for _ in range(num_moves):
            best = g_pos
            best_score = float('-inf')
            # include stay as a possible ghost choice
            for nbr in _SharedUtils.get_neighbors(g_pos, map_state, include_stay=True):
                dist = _SharedUtils.manhattan(nbr, p_pos)
                mobility = len(_SharedUtils.get_neighbors(nbr, map_state, include_stay=False))
                score = dist * 10 + mobility * 2
                if score > best_score:
                    best_score = score
                    best = nbr
            if best == g_pos:
                predicted.append(g_pos)
                break
            predicted.append(best)
            g_pos = best
            # simulate pacman step toward ghost
            p_pos = self._simulate_pacman_move(p_pos, g_pos, map_state)
        return predicted

    def _simulate_pacman_move(self, pacman_pos: Pos, ghost_pos: Pos, map_state: np.ndarray) -> Pos:
        best = pacman_pos
        best_d = _SharedUtils.manhattan(pacman_pos, ghost_pos)
        for nbr in _SharedUtils.get_neighbors(pacman_pos, map_state, include_stay=False):
            d = _SharedUtils.manhattan(nbr, ghost_pos)
            if d < best_d:
                best_d = d
                best = nbr
        return best

    def _get_move_direction(self, from_pos: Pos, to_pos: Pos) -> Optional[Move]:
        rdiff = to_pos[0] - from_pos[0]
        cdiff = to_pos[1] - from_pos[1]
        if rdiff == -1 and cdiff == 0:
            return Move.UP
        if rdiff == 1 and cdiff == 0:
            return Move.DOWN
        if rdiff == 0 and cdiff == -1:
            return Move.LEFT
        if rdiff == 0 and cdiff == 1:
            return Move.RIGHT
        return None

    def _is_valid_move(self, pos: Pos, move: Move, map_state: np.ndarray) -> bool:
        new_pos = _SharedUtils.pos_from_move(pos, move)
        return _SharedUtils.is_free(new_pos, map_state)


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent using Minimax with alpha-beta, transposition table,
    loop-awareness and corner-avoidance in evaluation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Minimax Evader (alpha-beta + loop-break)"
        self.max_depth = 4
        self._tt: Dict[Tuple[Pos, Pos, int, bool], float] = {}
        self.recent_states = deque(maxlen=8)
        random.seed()

    def step(self, map_state: np.ndarray,
             my_position: Pos,
             enemy_position: Pos,
             step_number: int) -> Move:

        state = (my_position, enemy_position)
        # If we've seen this state recently, break loop by picking a random valid safe move:
        if state in self.recent_states:
            valid_moves = [mv for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY) if self._is_valid_move(my_position, mv, map_state)]
            if valid_moves:
                return random.choice(valid_moves)
        self.recent_states.append(state)

        best_move = self._minimax_decision(my_position, enemy_position, map_state)
        if best_move and self._is_valid_move(my_position, best_move, map_state):
            return best_move

        # fallback
        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            if self._is_valid_move(my_position, mv, map_state):
                return mv
        return Move.STAY

    def _minimax_decision(self, ghost_pos: Pos, pacman_pos: Pos, map_state: np.ndarray) -> Optional[Move]:
        alpha = float('-inf')
        beta = float('inf')
        best_score = float('-inf')
        best_move = None

        # order ghost moves by heuristic: prefer moves that increase distance from pacman and increase mobility
        moves_scored = []
        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY):
            if self._is_valid_move(ghost_pos, mv, map_state):
                new_pos = _SharedUtils.pos_from_move(ghost_pos, mv)
                # heuristic combining distance & mobility
                score = _SharedUtils.manhattan(new_pos, pacman_pos) + 0.1 * len(_SharedUtils.get_neighbors(new_pos, map_state))
                moves_scored.append((score, mv))
        moves_scored.sort(reverse=True, key=lambda x: x[0])

        for _, mv in moves_scored:
            new_ghost = _SharedUtils.pos_from_move(ghost_pos, mv)
            score = self._minimax(new_ghost, pacman_pos, map_state, self.max_depth - 1, False, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = mv
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_move

    def _minimax(self, ghost_pos: Pos, pacman_pos: Pos,
                 map_state: np.ndarray, depth: int, is_ghost_turn: bool,
                 alpha: float, beta: float) -> float:

        key = (ghost_pos, pacman_pos, depth, is_ghost_turn)
        if key in self._tt:
            return self._tt[key]

        # Terminal checks
        if ghost_pos == pacman_pos:
            return -1000.0  # caught

        if depth == 0:
            val = self._evaluate_position(ghost_pos, pacman_pos, map_state)
            self._tt[key] = val
            return val

        if is_ghost_turn:
            value = float('-inf')
            # order moves to improve pruning
            moves = _SharedUtils.get_neighbors(ghost_pos, map_state, include_stay=True)
            # sort to try far moves first
            moves.sort(key=lambda p: -_SharedUtils.manhattan(p, pacman_pos))
            for new_pos in moves:
                v = self._minimax(new_pos, pacman_pos, map_state, depth - 1, False, alpha, beta)
                value = max(value, v)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            self._tt[key] = value
            return value
        else:
            # Pacman's turn (assume pacman tries to minimize ghost's eval)
            value = float('inf')
            moves = _SharedUtils.get_neighbors(pacman_pos, map_state, include_stay=False)
            if not moves:
                moves = [pacman_pos]
            moves.sort(key=lambda p: _SharedUtils.manhattan(p, ghost_pos))  # closer first
            for new_p in moves:
                v = self._minimax(ghost_pos, new_p, map_state, depth - 1, True, alpha, beta)
                value = min(value, v)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            self._tt[key] = value
            return value

    def _evaluate_position(self, ghost_pos: Pos, pacman_pos: Pos, map_state: np.ndarray) -> float:
        """
        Evaluation: prioritize distance and mobility; penalize corner positions (avoid being trapped).
        """
        dist = _SharedUtils.manhattan(ghost_pos, pacman_pos)
        mobility = len(_SharedUtils.get_neighbors(ghost_pos, map_state, include_stay=False))

        # corner penalty: discourage staying in map corners
        h, w = map_state.shape
        corner_penalty = 0
        if ghost_pos in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
            corner_penalty = 30

        # small noise to break ties deterministically but not purely random (helps avoid perfect symmetric cycles)
        noise = random.uniform(-0.4, 0.4)

        return float(dist * 10 + mobility * 2 - corner_penalty + noise)

    def _is_valid_move(self, pos: Pos, move: Move, map_state: np.ndarray) -> bool:
        new_pos = _SharedUtils.pos_from_move(pos, move)
        return _SharedUtils.is_free(new_pos, map_state)
