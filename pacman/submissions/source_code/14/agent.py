

from agent_interface import PacmanAgent as BasePacmanAgent 
from agent_interface import GhostAgent as BaseGhostAgent
from typing import Tuple, List
from environment import Move
import numpy as np
from collections import deque, defaultdict
from heapq import heappush, heappop
import math
from collections import deque, defaultdict

class PacmanAgent(BasePacmanAgent):
   

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # histories
        self.past_moves = deque(maxlen=200)
        self.ghost_positions = deque(maxlen=200)
        self.ghost_dirs = deque(maxlen=16)   
        self.visited_pairs = {}
        self.cycle_hits = defaultdict(int)
        # loop/swap
        self.swap_count = 0

        self.cutoff_base = 70   # base cutoff (tuned for your map size)
        self.intercept_prefer_ratio = 0.85  


    def step(self, map_state: np.ndarray, my_pos: tuple, enemy_pos: tuple, step_number: int):
 
        state = (my_pos, enemy_pos)
        self.ghost_positions.append(enemy_pos)


        if len(self.ghost_positions) >= 2:
            prev_g = self.ghost_positions[-2]
            dr, dc = enemy_pos[0] - prev_g[0], enemy_pos[1] - prev_g[1]
            self.ghost_dirs.append((dr, dc))

       
        if len(self.past_moves) > 0 and len(self.ghost_positions) > 1:
            last_enemy = self.ghost_positions[-2]
            prev_my_move = self.past_moves[-1]
            delta = prev_my_move.value
            prev_my_pos = (my_pos[0] - delta[0], my_pos[1] - delta[1])
            if enemy_pos == prev_my_pos and my_pos == last_enemy:
                self.swap_count += 1
            else:
                self.swap_count = max(0, self.swap_count - 1)
        else:
            self.swap_count = max(0, self.swap_count - 1)

        if self.swap_count >= 2:
            # break swap-loop 
            self.past_moves.append(Move.STAY)
            return Move.STAY

      
        cycle_detected = False
        if state in self.visited_pairs:
            prev_step = self.visited_pairs[state]
            cycle_len = step_number - prev_step
            if cycle_len >= 8:
                self.cycle_hits[state] += 1
                if self.cycle_hits[state] >= 2:
                    cycle_detected = True
            self.visited_pairs[state] = step_number
        else:
            self.visited_pairs[state] = step_number

        # also detect repeating 
        if self._detect_pattern(list(self.ghost_positions), min_len=3):
            cycle_detected = True

        goal_pred = self._predictive_goal(map_state, enemy_pos, horizon=6)    
        goal_intc = self._choose_intercept(map_state, my_pos, enemy_pos, step_number)

        # choose/blend goal
        if cycle_detected:
            goal = goal_intc
        else:
            d_pred = self._manhattan_distance(my_pos, goal_pred)
            d_intc = self._manhattan_distance(my_pos, goal_intc)

            if d_intc <= self.intercept_prefer_ratio * d_pred:
                goal = goal_intc
            else:
                goal = goal_pred


        
        weight = 1.0 + 0.003 * step_number
 
        cutoff = min(350, self.cutoff_base + step_number // 3)

        # run weighted A*
        path = self._astar(map_state, my_pos, goal, cutoff=cutoff, weight=weight)

        # fallback if no path
        if not path:
            move = self._fallback_move(map_state, my_pos, enemy_pos)
        else:
            move = path[0]

      
        if not self._is_valid_move(my_pos, move, map_state):
            move = self._fallback_move(map_state, my_pos, enemy_pos)

        self.past_moves.append(move)
        return move

    # ---------------- Pattern detection ----------------
    def _detect_pattern(self, seq, min_len=3):
        if len(seq) < 2 * min_len:
            return False
       
        max_L = min(8, len(seq)//2)
        for L in range(min_len, max_L+1):
            if seq[-L:] == seq[-2*L:-L]:
                return True
        return False

   
    def _predictive_goal(self, map_state, enemy_pos, horizon=6):
        if not self.ghost_dirs:
            return enemy_pos
        
        sum_dr = sum(d[0] for d in self.ghost_dirs)
        sum_dc = sum(d[1] for d in self.ghost_dirs)
        avg_dr = int(round(sum_dr / len(self.ghost_dirs)))
        avg_dc = int(round(sum_dc / len(self.ghost_dirs)))
      
        if avg_dr == 0 and avg_dc == 0:
            freq = defaultdict(int)
            for d in self.ghost_dirs:
                freq[d] += 1
            avg_dr, avg_dc = max(freq.items(), key=lambda x: x[1])[0]

        r, c = enemy_pos
        best_pred = enemy_pos
        for t in range(1, horizon+1):
            nr, nc = r + avg_dr * t, c + avg_dc * t
            cand = (nr, nc)
            if self._is_valid_position(cand, map_state):
                best_pred = cand
            else:
                break
        return best_pred

    def _choose_intercept(self, map_state, my_pos, enemy_pos, step_number):

        candidates = []
        hist = list(self.ghost_positions)

        for p in reversed(hist[-10:]):
            candidates.append(p)
        pred = self._predictive_goal(map_state, enemy_pos, horizon=8)
        candidates.append(pred)

        px, py = pred
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                q = (px + dr, py + dc)
                if self._is_valid_position(q, map_state):
                    candidates.append(q)

        # unique and valid
        seen = set()
        uniq = []
        for p in candidates:
            if p not in seen and self._is_valid_position(p, map_state):
                seen.add(p)
                uniq.append(p)

        # evaluate: prefer candidate where my_dist <= ghost_dist 
        best = None
        best_score = (float('inf'), float('inf'))  # (my_d - ghost_d, my_d)
        for cand in uniq:
            my_d = self._manhattan_distance(my_pos, cand)
            ghost_d = self._manhattan_distance(enemy_pos, cand)
            score = (my_d - ghost_d, my_d)
            if score < best_score:
                best_score = score
                best = cand
        return best or enemy_pos


    def _astar(self, map_state, start, goal, cutoff=200, weight=1.0):
        if start == goal:
            return []

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_heap = []
        counter = 0
        heappush(open_heap, (heuristic(start, goal) * weight, counter, start))
        came_from = {}
        gscore = {start: 0}
        expanded = 0
        closed = set()

        while open_heap and expanded <= cutoff:
            _, _, current = heappop(open_heap)
            if current == goal:
                break
            if current in closed:
                continue
            closed.add(current)
            expanded += 1

            for neighbor, mv in self._get_neighbors_with_move(current, map_state):
                tentative_g = gscore[current] + 1
                if tentative_g < gscore.get(neighbor, math.inf):
                    gscore[neighbor] = tentative_g
                    came_from[neighbor] = (current, mv)
                    f = tentative_g + heuristic(neighbor, goal) * weight
                    counter += 1
                    heappush(open_heap, (f, counter, neighbor))

 
        if goal not in gscore:
            best_node = min(gscore.keys(), key=lambda n: gscore[n] + heuristic(n, goal))
            cur = best_node
        else:
            cur = goal

        # reconstruct
        path_moves = []
        while cur in came_from:
            prev, mv = came_from[cur]
            path_moves.append(mv)
            cur = prev
        path_moves.reverse()
        return path_moves

    # ---------------- Fallback ----------------
    def _fallback_move(self, map_state, my_pos, enemy_pos):
        neighbors = self._get_neighbors_with_move(my_pos, map_state)
        if not neighbors:
            return Move.STAY
       
        neighbors.sort(key=lambda nm: (self._manhattan_distance(nm[0], enemy_pos),
                                       0 if (self.past_moves and nm[1] == self.past_moves[-1]) else 1))
        return neighbors[0][1]



    def _get_neighbors_with_move(self, pos, map_state):
        r, c = pos
        res = []
        for mv in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            nr, nc = r + mv.value[0], c + mv.value[1]
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] == 0:
                res.append(((nr, nc), mv))
        return res

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        if r < 0 or r >= map_state.shape[0] or c < 0 or c >= map_state.shape[1]:
            return False
        return map_state[r, c] == 0

    def _is_valid_move(self, pos, move, map_state):
        nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
        return 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] == 0

    def _manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# -----------------------
# Shared small utilities
# -----------------------
def apply_move(pos: Tuple[int, int], mv: Move) -> Tuple[int, int]:
    dr, dc = mv.value
    return (pos[0] + dr, pos[1] + dc)

def in_bounds(pos: Tuple[int, int], h: int, w: int) -> bool:
    r, c = pos
    return 0 <= r < h and 0 <= c < w

def is_free(pos: Tuple[int, int], grid: np.ndarray) -> bool:
    r, c = pos
    return grid[r, c] == 0

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GhostAgent(BaseGhostAgent):
    """
    Evasive agent: maximize Manhattan distance to Pacman every step
    while avoiding forbidden cells (9,4) and (9,16). If no move increases
    the distance, pick the valid move with the largest distance. Else STAY.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Forbidden absolute cells (row, col)
        self.forbidden = {(9, 4), (9, 16)}

    def step(self,
             map_state: np.ndarray,
             my_position: Tuple[int, int],
             enemy_position: Tuple[int, int],
             step_number: int) -> Move:

        candidates: List[Move] = [Move.LEFT, Move.RIGHT,Move.UP, Move.DOWN,  Move.STAY]
        h, w = map_state.shape
        cur_dist = manhattan(my_position, enemy_position)

        # 1) Valid moves that avoid forbidden and strictly increase distance
        improving: List[Tuple[int, Move]] = []
        for mv in candidates:
            new_pos = apply_move(my_position, mv)
            if not in_bounds(new_pos, h, w):
                continue
            if not is_free(new_pos, map_state):
                continue
            if new_pos in self.forbidden:
                continue

            d = manhattan(new_pos, enemy_position)
            if d > cur_dist:
                improving.append((d, mv))

        if improving:
            # Pick the move with the largest distance (compare by distance only)
            return max(improving, key=lambda x: x[0])[1]

        # 2) Otherwise choose any valid move (avoid forbidden) with the largest distance
        fallback: List[Tuple[int, Move]] = []
        for mv in candidates:
            new_pos = apply_move(my_position, mv)
            if not in_bounds(new_pos, h, w):
                continue
            if not is_free(new_pos, map_state):
                continue
            if new_pos in self.forbidden:
                continue

            d = manhattan(new_pos, enemy_position)
            fallback.append((d, mv))

        if fallback:
            return max(fallback, key=lambda x: x[0])[1]

        # 3) No valid moves at all
        return Move.STAY
