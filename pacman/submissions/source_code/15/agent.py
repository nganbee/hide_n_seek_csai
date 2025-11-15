import sys
from pathlib import Path
from collections import deque
from heapq import heappush, heappop
import numpy as np
import itertools
import random
import time

# --- Đảm bảo import được src ---
src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from agent_interface import PacmanAgent as BasePacmanAgent
    from agent_interface import GhostAgent as BaseGhostAgent
    from environment import Move
except ImportError:
    print("Lỗi: Không thể import từ src/. Kiểm tra cấu trúc thư mục.")
    sys.exit(1)


# ======================================================================
#                     HÀM DÙNG CHUNG
# ======================================================================
class AgentHelpers:
    """Các hàm hỗ trợ chung."""

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return pos[0] + dr, pos[1] + dc

    def _get_neighbors(self, pos, map_state):
        for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            nxt = self._apply_move(pos, move)
            if self._is_valid_position(nxt, map_state):
                yield nxt, move

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ======================================================================
#                     PACMAN (SEEK) – Trap + Predictive A*
# ======================================================================
class PacmanAgent(BasePacmanAgent, AgentHelpers):
    """
    Pacman (Seeker) Agent - A* + Trap-based Hunting Strategy
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _is_valid_position(self, position, map_state):
        row, col = position
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _apply_move(self, position, move):
        row, col = position
        if move == Move.UP:
            return (row - 1, col)
        elif move == Move.DOWN:
            return (row + 1, col)
        elif move == Move.LEFT:
            return (row, col - 1)
        elif move == Move.RIGHT:
            return (row, col + 1)
        elif move == Move.STAY:
            return (row, col)
        else:
            return position

    def _get_neighbors(self, position, map_state):
        neighbors = []
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        for move in moves:
            new_pos = self._apply_move(position, move)
            if self._is_valid_position(new_pos, map_state):
                neighbors.append((new_pos, move))
        return neighbors

    def _mannhatan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _find_nearest_valid_position(self, target, map_state):
        best_pos = target
        best_distance = float('inf')
        height, width = map_state.shape
        for r in range(height):
            for c in range(width):
                if self._is_valid_position((r, c), map_state):
                    dist = self._mannhatan_distance((r, c), target)
                    if dist < best_distance:
                        best_distance = dist
                        best_pos = (r, c)
        return best_pos

    # =====================================================
    # SEARCH & PREDICTION
    # =====================================================
    def bfs(self, start, goal, map_state):
        from collections import deque
        if start == goal:
            return [Move.STAY]
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            current_pos, path = queue.popleft()
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos in visited:
                    continue
                new_path = path + [move]
                if next_pos == goal:
                    return new_path
                visited.add(next_pos)
                queue.append((next_pos, new_path))
        return []

    def a_star(self, start, goal, map_state):
        import heapq
        if start == goal:
            return [Move.STAY]

        def heuristic(pos):
            return self._mannhatan_distance(pos, goal)

        open_set = [(heuristic(start), 0, start, [])]
        closed_set = set()
        g_costs = {start: 0}

        while open_set:
            f_cost, g_cost, current_pos, path = heapq.heappop(open_set)
            if current_pos in closed_set:
                continue
            closed_set.add(current_pos)
            if current_pos == goal:
                return path
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos in closed_set:
                    continue
                tentative_g = g_cost + 1
                if next_pos not in g_costs or tentative_g < g_costs[next_pos]:
                    g_costs[next_pos] = tentative_g
                    new_path = path + [move]
                    f_cost = tentative_g + heuristic(next_pos)
                    heapq.heappush(open_set, (f_cost, tentative_g, next_pos, new_path))
        return []

    def predict_enemy_position(self, enemy_pos, my_pos, map_state):
        best_distance = self._mannhatan_distance(enemy_pos, my_pos)
        predicted_pos = enemy_pos
        for next_pos, move in self._get_neighbors(enemy_pos, map_state):
            distance = self._mannhatan_distance(next_pos, my_pos)
            if distance > best_distance:
                best_distance = distance
                predicted_pos = next_pos
        return predicted_pos

    def find_dead_ends(self, map_state):
        dead_ends = []
        height, width = map_state.shape
        for r in range(height):
            for c in range(width):
                if self._is_valid_position((r, c), map_state):
                    neighbors = len(self._get_neighbors((r, c), map_state))
                    if neighbors <= 2:
                        dead_ends.append((r, c))
        return dead_ends

    def find_trap_blocking_position(self, enemy_pos, trap_pos, map_state):
        escape_directions = []
        for next_pos, move in self._get_neighbors(enemy_pos, map_state):
            if self._mannhatan_distance(next_pos, trap_pos) > self._mannhatan_distance(enemy_pos, trap_pos):
                escape_directions.append(next_pos)
        if escape_directions:
            avg_row = sum(p[0] for p in escape_directions) // len(escape_directions)
            avg_col = sum(p[1] for p in escape_directions) // len(escape_directions)
            return (avg_row, avg_col)
        return enemy_pos

    def find_narrow_corridors(self, map_state):
        corridors = []
        height, width = map_state.shape
        for r in range(height):
            for c in range(width):
                if self._is_valid_position((r, c), map_state):
                    neighbors = self._get_neighbors((r, c), map_state)
                    if len(neighbors) == 3:
                        neighbor_positions = [pos for pos, move in neighbors if move != Move.STAY]
                        if len(neighbor_positions) == 2:
                            pos1, pos2 = neighbor_positions
                            if (pos1[0] == r == pos2[0]) or (pos1[1] == c == pos2[1]):
                                corridors.append((r, c))
        return corridors

    def find_nearest_trap(self, position, trap_positions):
        if not trap_positions:
            return None
        return min(trap_positions, key=lambda trap: self._mannhatan_distance(position, trap))

    def find_herding_position(self, my_pos, enemy_pos, target_trap, map_state):
        trap_to_ghost_row = enemy_pos[0] - target_trap[0]
        trap_to_ghost_col = enemy_pos[1] - target_trap[1]
        herding_row = target_trap[0] - trap_to_ghost_row
        herding_col = target_trap[1] - trap_to_ghost_col
        herding_pos = (herding_row, herding_col)
        if not self._is_valid_position(herding_pos, map_state):
            herding_pos = self._find_nearest_valid_position(herding_pos, map_state)
        return herding_pos

    # =====================================================
    # MAIN STRATEGY
    # =====================================================
    def step(self, map_state, my_position, enemy_position, step_number):
        """Trap-based hunting strategy."""
        dead_ends = self.find_dead_ends(map_state)
        corridors = self.find_narrow_corridors(map_state)
        trap_positions = dead_ends + corridors

        nearest_trap = self.find_nearest_trap(enemy_position, trap_positions)

        if nearest_trap and self._mannhatan_distance(enemy_position, nearest_trap) <= 3:
            blocking_pos = self.find_trap_blocking_position(enemy_position, nearest_trap, map_state)
            path = self.a_star(my_position, blocking_pos, map_state)
        elif self._mannhatan_distance(my_position, enemy_position) <= 2:
            predicted = self.predict_enemy_position(enemy_position, my_position, map_state)
            path = self.a_star(my_position, predicted, map_state)
        else:
            if trap_positions:
                target_trap = min(trap_positions, key=lambda t: self._mannhatan_distance(t, enemy_position))
                herding_pos = self.find_herding_position(my_position, enemy_position, target_trap, map_state)
                path = self.a_star(my_position, herding_pos, map_state)
            else:
                predicted = self.predict_enemy_position(enemy_position, my_position, map_state)
                path = self.a_star(my_position, predicted, map_state)

        if path:
            return path[0]
        return Move.STAY


# ======================================================================
#                   GHOST (HIDE) – Expectimax Depth=2
# ======================================================================
class GhostAgent(BaseGhostAgent, AgentHelpers):
    """HIDE AGENT (GHOST) – Expectimax Depth=2"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = 2
        self.time_limit = 0.9

    def evaluate(self, my_position, enemy_position, map_state):
        dist = self._manhattan(my_position, enemy_position)
        penalty = 0
        for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            nxt = self._apply_move(my_position, move)
            if not self._is_valid_position(nxt, map_state):
                penalty += 0.5
        return dist - penalty

    def expectimax(self, my_position, enemy_position, map_state, depth, start_t):
        if time.time() - start_t > self.time_limit * 0.95:
            return None, self.evaluate(my_position, enemy_position, map_state)
        if depth >= self.max_depth:
            return None, self.evaluate(my_position, enemy_position, map_state)
        if depth % 2 == 0:
            best_val = -float("inf")
            best_move = None
            for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY):
                nxt = self._apply_move(my_position, move)
                if not self._is_valid_position(nxt, map_state):
                    continue
                _, val = self.expectimax(nxt, enemy_position, map_state, depth + 1, start_t)
                if val is None:
                    continue
                if val > best_val:
                    best_val, best_move = val, move
            return (best_move or Move.STAY), best_val
        else:
            vals = []
            valid_moves = []
            for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY):
                nxt = self._apply_move(enemy_position, move)
                if not self._is_valid_position(nxt, map_state):
                    continue
                valid_moves.append(nxt)
            if not valid_moves:
                return None, self.evaluate(my_position, enemy_position, map_state)
            for nxt_enemy_pos in valid_moves:
                _, val = self.expectimax(my_position, nxt_enemy_pos, map_state, depth + 1, start_t)
                vals.append(val if val is not None else self.evaluate(my_position, nxt_enemy_pos, map_state))
            return None, sum(vals) / len(vals)

    def step(self, map_state, my_position, enemy_position, step_number):
        try:
            start_t = time.time()
            move, _ = self.expectimax(my_position, enemy_position, map_state, 0, start_t)
            if not isinstance(move, Move):
                best_move = Move.STAY
                best_dist = self._manhattan(my_position, enemy_position)
                for m in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
                    nxt = self._apply_move(my_position, m)
                    if self._is_valid_position(nxt, map_state):
                        d = self._manhattan(nxt, enemy_position)
                        if d > best_dist:
                            best_dist = d
                            best_move = m
                return best_move
            return move
        except Exception:
            return Move.STAY
