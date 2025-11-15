import sys
from pathlib import Path
from collections import deque

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import heapq


def is_valid_position(pos: tuple, map_state: np.ndarray) -> bool:
    """Check if a position is valid (not a wall and within bounds)."""
    row, col = pos
    height, width = map_state.shape

    if row < 0 or row >= height or col < 0 or col >= width:
        return False

    return map_state[row, col] == 0


def is_valid_move(pos: tuple, move: Move, map_state: np.ndarray) -> bool:
    """Check if a move from pos is valid."""
    delta_row, delta_col = move.value
    new_pos = (pos[0] + delta_row, pos[1] + delta_col)
    return is_valid_position(new_pos, map_state)


def manhattan_distance(pos1: tuple, pos2: tuple) -> int:
    """Khoảng cách Manhattan giữa hai ô."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def apply_move(pos: tuple, move: Move) -> tuple:
    """Di chuyển từ pos theo hướng move, trả về vị trí mới."""
    delta_row, delta_col = move.value
    return pos[0] + delta_row, pos[1] + delta_col

def get_direction(current: tuple, next_pos: tuple) -> Move:
    """Xác định hướng di chuyển giữa hai ô."""
    dr = next_pos[0] - current[0]
    dc = next_pos[1] - current[1]
    if dr == -1 and dc == 0:
        return Move.UP
    elif dr == 1 and dc == 0:
        return Move.DOWN
    elif dr == 0 and dc == -1:
        return Move.LEFT
    elif dr == 0 and dc == 1:
        return Move.RIGHT
    return Move.STAY


class PacmanAgent(BasePacmanAgent):
    # Tunable parameters
    PREDICTION_STEPS = 2  # Steps ahead to predict ghost position
    MINI_COMMIT_STEPS = 3  # Continue in same direction for N steps
    REPLAN_THRESHOLD = 3  # Distance deviation to trigger replanning
    ANTI_LOOP_MEMORY = 5  # Remember last N positions to avoid loops

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_enemy_state = None
        self.name = "Pacman"

        # Path planning
        self.path = []  # Current planned path (cache)
        self.target_position = None  # Current target

        # Velocity tracking
        self.last_enemy_position = None  # Previous ghost position
        self.enemy_velocity = (0, 0)  # Estimated velocity vector

        # Movement stabilization
        self.last_move = None  # Previous move
        self.commit_counter = 0  # Steps remaining in current direction
        self.recent_positions = []  # Last N positions (anti-loop)
        self.idle_counter = 0  # Track consecutive STAY actions

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Simplified chasing loop for simultaneous movement:
        1. Update history
        2. Update velocity (for direction tracking only)
        3. Replan periodically
        4. Stabilize movement (mini-commit + anti-reverse + anti-loop)
        5. Follow path
        """

        # STEP 1: Update history
        self._update_history(my_position)

        # STEP 2: Update velocity (direction tracking only)
        self._update_velocity(enemy_position)

        # Disable stabilization when Ghost is close
        current_distance = manhattan_distance(my_position, enemy_position)
        is_close_range = current_distance <= 4  # Within 4 Manhattan distance

        if is_close_range:
            # Clear commit counter to allow instant direction changes
            self.commit_counter = 0
            # print(f"[DEBUG] Ghost close (dist={current_distance}) - stabilization bypassed")

        # STEP 3: Replan periodically or when needed
        if step_number % 2 == 0 or not self.path or self.idle_counter > 2:
            self._replan(my_position, enemy_position, map_state)
            if self.idle_counter > 2:
                # Force clear state to break idle loop
                self.path = []
                self.target_position = None
                self.idle_counter = 0

        # Replan when Ghost moves too far from previous target
        if self.target_position is not None:
            if manhattan_distance(enemy_position, self.target_position) > self.REPLAN_THRESHOLD:
                self._replan(my_position, enemy_position, map_state)

        # STEP 4: Stabilize movement (mini-commit + anti-reverse + anti-loop)
        # Only apply stabilization when Ghost is far
        if not is_close_range:
            stabilized_move = self._stabilize_movement(my_position, enemy_position, map_state)
            if stabilized_move != Move.STAY:
                return stabilized_move

        # STEP 5: Follow path (with relaxed constraints when close)
        move = self._follow_path(my_position, enemy_position, map_state, relax_constraints=is_close_range)

        # --- Smart wait breaker (fix infinite STAY) ---
        if move == Move.STAY:
            # Detect if ghost changed position or direction -> break waiting state
            if self._last_enemy_state is None:
                self._last_enemy_state = (enemy_position, self.enemy_velocity)
            else:
                last_pos, last_vel = self._last_enemy_state
                # If ghost moves or changes direction => force replan
                if enemy_position != last_pos or self.enemy_velocity != last_vel:
                    self.path = []
                    self.target_position = None
                    self.commit_counter = 0
                    self.idle_counter = 0
                    move = self._find_fallback_move(my_position, enemy_position, map_state)
                self._last_enemy_state = (enemy_position, self.enemy_velocity)

        return move

    # ===== STEP 1: Update History =====
    def _update_history(self, position: tuple):
        """Track recent positions for anti-loop"""
        self.recent_positions.append(position)
        if len(self.recent_positions) > self.ANTI_LOOP_MEMORY:
            self.recent_positions.pop(0)

    # ===== STEP 2: Update Velocity =====
    def _update_velocity(self, enemy_position: tuple):
        """Calculate ghost velocity vector"""
        if self.last_enemy_position is not None:
            self.enemy_velocity = (
                enemy_position[0] - self.last_enemy_position[0],
                enemy_position[1] - self.last_enemy_position[1]
            )
        self.last_enemy_position = enemy_position

    def _replan(self, my_pos: tuple, target: tuple, map_state: np.ndarray):
        """Recalculate path using A* and validate result"""
        path = self._astar(map_state, my_pos, target)
        # defensive: ensure path is a list of tuples (positions)
        if not path or not all(isinstance(p, tuple) and len(p) == 2 for p in path):
            self.path = []
            self.target_position = None
            return

        if len(path) >= 2:
            self.path = path[1:]  # Skip current position
            self.target_position = target
        else:
            self.path = []
            self.target_position = None

    # ===== STEP 4: Stabilize Movement =====
    def _stabilize_movement(self, my_pos: tuple, enemy_pos: tuple, map_state: np.ndarray) -> Move:
        """
        Apply mini-commit, anti-reverse, and anti-loop stabilization.
        Returns Move.STAY if no stabilization applied.
        """
        # Mini-commit: continue in same direction if committed
        if self.commit_counter > 0 and self.last_move is not None:
            if is_valid_move(my_pos, self.last_move, map_state):
                next_pos = apply_move(my_pos, self.last_move)

                # Check if moving away too much
                curr_dist = manhattan_distance(my_pos, enemy_pos)
                next_dist = manhattan_distance(next_pos, enemy_pos)

                # Check anti-loop: avoid recently visited positions
                if next_pos in self.recent_positions[-3:]:
                    self.commit_counter = 0  # Break commit if looping
                    return Move.STAY

                # Continue commit if not moving away significantly
                if next_dist <= curr_dist + 2:
                    self.commit_counter -= 1
                    self.idle_counter = 0  # Reset idle counter
                    return self.last_move

            # Commit no longer valid
            self.commit_counter = 0

        return Move.STAY

    # ===== STEP 5: Follow Path =====
    def _follow_path(self, my_pos: tuple, enemy_pos: tuple, map_state: np.ndarray, relax_constraints: bool = False) -> Move:
        """Follow the cached path with anti-reverse and anti-loop checks

        Args:
            relax_constraints: If True, bypass anti-loop and anti-reverse when Ghost is close
        """
        # Try to follow cached path
        if self.path:
            # Skip current position if in path
            while self.path and self.path[0] == my_pos:
                self.path.pop(0)

            if self.path:
                next_pos = self.path[0]
                move = get_direction(my_pos, next_pos)

                # Skip safety checks when Ghost is close
                if relax_constraints:
                    # print(f"[DEBUG] Relaxed mode - following A* path directly")
                    if is_valid_move(my_pos, move, map_state):
                        self.last_move = move
                        self.commit_counter = 0  # No commit in close range
                        self.idle_counter = 0
                        return move
                else:
                    # NORMAL MODE: Apply full safety checks
                    # Anti-loop: avoid recently visited positions
                    if next_pos in self.recent_positions[-3:]:
                        # Try to find alternative
                        alt_move = self._find_alternative_move(my_pos, enemy_pos, map_state, avoid_positions=self.recent_positions[-3:])
                        if alt_move != Move.STAY:
                            move = alt_move

                    # Anti-reverse: avoid reversing direction
                    if self.last_move and self._is_reverse(move, self.last_move):
                        alt_move = self._find_alternative_move(my_pos, enemy_pos, map_state)
                        if alt_move != Move.STAY:
                            move = alt_move

                    # Execute move if valid
                    if is_valid_move(my_pos, move, map_state):
                        self.last_move = move
                        self.commit_counter = self.MINI_COMMIT_STEPS
                        self.idle_counter = 0  # Reset idle counter
                        return move

        # Fallback: direct A* to enemy
        path = self._astar(map_state, my_pos, enemy_pos)
        if path and len(path) >= 2:
            next_pos = path[1]
            move = get_direction(my_pos, next_pos)

            # Skip anti-reverse check when Ghost is close
            if not relax_constraints:
                # Anti-reverse check (only when far)
                if self.last_move and self._is_reverse(move, self.last_move):
                    alt_move = self._find_alternative_move(my_pos, enemy_pos, map_state)
                    if alt_move != Move.STAY:
                        move = alt_move

            self.last_move = move
            self.commit_counter = 0 if relax_constraints else self.MINI_COMMIT_STEPS
            self.idle_counter = 0  # Reset idle counter
            return move
        elif path and len(path) == 1:
            # A* returned only current position - try escape move
            escape_move = self._find_escape_move(my_pos, enemy_pos, map_state)
            if escape_move != Move.STAY:
                self.last_move = escape_move
                self.commit_counter = self.MINI_COMMIT_STEPS
                self.idle_counter = 0  # Reset idle counter
                return escape_move

        # Last resort: try any valid move not in recent positions
        fallback_move = self._find_fallback_move(my_pos, enemy_pos, map_state)
        if fallback_move != Move.STAY:
            self.last_move = fallback_move
            self.idle_counter = 0  # Reset idle counter
            return fallback_move

        # Truly stuck - increment idle counter
        self.idle_counter += 1
        return Move.STAY

    # ===== Helper Methods =====
    def _is_reverse(self, move1: Move, move2: Move) -> bool:
        """Check if move1 is opposite of move2"""
        reverse_pairs = [
            (Move.UP, Move.DOWN), (Move.DOWN, Move.UP),
            (Move.LEFT, Move.RIGHT), (Move.RIGHT, Move.LEFT)
        ]
        return (move1, move2) in reverse_pairs

    def _find_alternative_move(self, pos: tuple, enemy_pos: tuple,
                              map_state: np.ndarray, avoid_positions: list = None) -> Move:
        """Find best alternative move that gets closer to enemy"""
        if avoid_positions is None:
            avoid_positions = []

        best_move = Move.STAY
        best_dist = float('inf')

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if is_valid_move(pos, move, map_state):
                next_pos = apply_move(pos, move)

                # Skip positions to avoid
                if next_pos in avoid_positions:
                    continue

                # Skip if reverse of last move
                if self.last_move and self._is_reverse(move, self.last_move):
                    continue

                dist = manhattan_distance(next_pos, enemy_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_move = move

        return best_move

    def _find_escape_move(self, pos: tuple, enemy_pos: tuple, map_state: np.ndarray) -> Move:
        """Find a move that moves away from enemy (escape move)"""
        best_move = Move.STAY
        best_dist = -1

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if is_valid_move(pos, move, map_state):
                next_pos = apply_move(pos, move)

                # Skip if reverse of last move
                if self.last_move and self._is_reverse(move, self.last_move):
                    continue

                # Skip recent positions
                if next_pos in self.recent_positions[-3:]:
                    continue

                dist = manhattan_distance(next_pos, enemy_pos)
                if dist > best_dist:
                    best_dist = dist
                    best_move = move

        return best_move

    def _find_fallback_move(self, pos: tuple, enemy_pos: tuple, map_state: np.ndarray) -> Move:
        """Find any valid move not in recent positions and not reverse"""
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if is_valid_move(pos, move, map_state):
                next_pos = apply_move(pos, move)

                # Skip if reverse of last move
                if self.last_move and self._is_reverse(move, self.last_move):
                    continue

                # Skip recent positions
                if next_pos in self.recent_positions[-3:]:
                    continue

                return move

        # If all moves are blocked by constraints, just pick any valid move
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if is_valid_move(pos, move, map_state):
                return move

        return Move.STAY


    def _astar(self, map_state: np.ndarray,
               start: tuple,
               goal: tuple) -> list[tuple]:
        """
        Thuật toán A* cơ bản để tìm đường ngắn nhất từ start đến goal.
        """
        n, m = map_state.shape
        dirs = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

        def h(a, b):
            # heuristic: khoảng cách Manhattan
            return manhattan_distance(a, b)

        open_set = []
        heapq.heappush(open_set, (h(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}
        visited = set()

        while open_set:
            f, g, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                # reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for move in dirs:
                delta_row, delta_col = move.value
                nx, ny = current[0] + delta_row, current[1] + delta_col

                if not (0 <= nx < n and 0 <= ny < m):
                    continue
                if map_state[nx, ny] == 1:
                    continue

                tentative_g = g + 1
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + h((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, tentative_g, (nx, ny)))
                    came_from[(nx, ny)] = current

        return []  # không tìm được đường đi


class GhostAgent(BaseGhostAgent):
    # Tunable parameters
    PATTERN_DETECTION_WINDOW = 15
    MINIMAX_DEPTH = 3
    STRATEGY_SWITCH_DURATION = 3
    POSITION_HISTORY_SIZE = 5
    REPLAN_INTERVAL = 2
    MOBILITY_WEIGHT = 0.7

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost"
        self.position_history = []
        self.strategy_counter = 0
        self.target_position = None
        self.last_replan_step = 0

    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = apply_move(pos, move)
            if is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors

    def _count_valid_neighbors(self, pos: tuple, map_state: np.ndarray) -> int:
        # Count escape routes
        count = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = apply_move(pos, move)
            if is_valid_position(next_pos, map_state):
                count += 1
        return count

    def _is_corner_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        # Corners are bad - easy to get trapped
        row, col = pos
        height, width = map_state.shape

        margin = 2
        is_top = row < margin
        is_bottom = row >= height - margin
        is_left = col < margin
        is_right = col >= width - margin

        return (is_top and is_left) or (is_top and is_right) or \
               (is_bottom and is_left) or (is_bottom and is_right)

    def _bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            curr, path = queue.popleft()
            if curr == goal:
                return path

            for next_pos, move in self._get_neighbors(curr, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
        return []

    def _detect_repetitive_pattern(self) -> bool:
        if len(self.position_history) < self.PATTERN_DETECTION_WINDOW * 2:
            return False

        recent = self.position_history[-self.PATTERN_DETECTION_WINDOW:]
        unique = len(set(recent))
        return unique <= 3  # stuck in small loop

    def _find_furthest_position(self, my_pos: tuple, pacman_pos: tuple,
                               map_state: np.ndarray) -> tuple:
        # BFS from pacman to find distance to all reachable spots
        queue = deque([(pacman_pos, 0)])
        distances = {pacman_pos: 0}

        while queue:
            curr, dist = queue.popleft()
            for next_pos, _ in self._get_neighbors(curr, map_state):
                if next_pos not in distances:
                    distances[next_pos] = dist + 1
                    queue.append((next_pos, dist + 1))

        # Pick furthest spot that's still reachable
        max_dist = -1
        best_targets = []

        for pos, dist_from_pacman in distances.items():
            dist_to_me = manhattan_distance(pos, my_pos)

            if dist_to_me <= 10:
                if dist_from_pacman > max_dist:
                    max_dist = dist_from_pacman
                    best_targets = [pos]
                elif dist_from_pacman == max_dist:
                    best_targets.append(pos)

        if best_targets:
            return min(best_targets, key=lambda p: manhattan_distance(p, my_pos))
        return my_pos

    def _minimax_evasion(self, my_pos: tuple, pacman_pos: tuple,
                        depth: int, is_pacman_turn: bool,
                        map_state: np.ndarray, memo: dict = None) -> float:
        if memo is None:
            memo = {}

        state_key = (my_pos, pacman_pos, depth, is_pacman_turn)
        if state_key in memo:
            return memo[state_key]

        # Base case
        if depth == 0 or my_pos == pacman_pos:
            dist = manhattan_distance(my_pos, pacman_pos)
            mobility = self._count_valid_neighbors(my_pos, map_state)
            score = dist + (mobility * self.MOBILITY_WEIGHT)

            # Avoid corners at all costs
            if self._is_corner_position(my_pos, map_state):
                score -= 50

            memo[state_key] = score
            return score

        if is_pacman_turn:
            # Pacman minimizes distance
            best = float('inf')
            for next_pos, _ in self._get_neighbors(pacman_pos, map_state):
                score = self._minimax_evasion(my_pos, next_pos, depth - 1,
                                             False, map_state, memo)
                best = min(best, score)
            memo[state_key] = best
            return best
        else:
            # Ghost maximizes distance
            best = float('-inf')
            for next_pos, _ in self._get_neighbors(my_pos, map_state):
                score = self._minimax_evasion(next_pos, pacman_pos, depth - 1,
                                             True, map_state, memo)
                best = max(best, score)
            memo[state_key] = best
            return best

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:

        # Track position history
        self.position_history.append(my_position)
        if len(self.position_history) > self.POSITION_HISTORY_SIZE:
            self.position_history.pop(0)

        # Switch strategy if stuck in pattern
        if self._detect_repetitive_pattern() and self.strategy_counter == 0:
            self.strategy_counter = self.STRATEGY_SWITCH_DURATION
            self.target_position = None

        # Alternative strategy: go to furthest position
        if self.strategy_counter > 0:
            self.strategy_counter -= 1

            if (self.target_position is None or
                step_number - self.last_replan_step >= self.REPLAN_INTERVAL):
                self.target_position = self._find_furthest_position(
                    my_position, enemy_position, map_state
                )
                self.last_replan_step = step_number

            path = self._bfs(my_position, self.target_position, map_state)

            if path:
                next_pos = apply_move(my_position, path[0])
                curr_dist = manhattan_distance(my_position, enemy_position)
                next_dist = manhattan_distance(next_pos, enemy_position)

                # Don't go to corners or closer to pacman
                if self._is_corner_position(next_pos, map_state):
                    self.strategy_counter = 0
                elif next_dist >= curr_dist:
                    return path[0]
                else:
                    self.strategy_counter = 0

        # Main strategy: minimax with safety checks
        memo = {}
        curr_dist = manhattan_distance(my_position, enemy_position)

        # Evaluate all possible moves
        move_scores = []
        for next_pos, move in self._get_neighbors(my_position, map_state):
            score = self._minimax_evasion(
                next_pos, enemy_position, self.MINIMAX_DEPTH,
                True, map_state, memo
            )

            # Extra penalty for corners
            if self._is_corner_position(next_pos, map_state):
                score -= 100

            next_dist = manhattan_distance(next_pos, enemy_position)
            move_scores.append((move, score, next_dist, next_pos))

        if not move_scores:
            return Move.STAY

        # Prefer moves that don't get closer to pacman
        safe_moves = [(m, s, d, p) for m, s, d, p in move_scores if d >= curr_dist]

        if safe_moves:
            best_move, _, _, _ = max(safe_moves, key=lambda x: x[1])
            return best_move
        else:
            # All moves get closer - pick best minimax score
            best_move, best_score, _, _ = max(move_scores, key=lambda x: x[1])
            if best_score < -50:
                best_move = Move.STAY

        if best_move != Move.STAY:
            return best_move

        # Fallback: greedy distance maximization
        best_dist = curr_dist
        best_move = Move.STAY
        best_mobility = self._count_valid_neighbors(my_position, map_state)

        for next_pos, move in self._get_neighbors(my_position, map_state):
            dist = manhattan_distance(next_pos, enemy_position)
            mobility = self._count_valid_neighbors(next_pos, map_state)

            if self._is_corner_position(next_pos, map_state):
                continue  # never go to corners

            if dist > best_dist or (dist == best_dist and mobility > best_mobility):
                best_dist = dist
                best_mobility = mobility
                best_move = move

        return best_move