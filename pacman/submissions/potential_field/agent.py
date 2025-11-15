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
import random
import math


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        # - self.name = "Your Agent Name"
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
            enemy_position: Ghost's current (row, col)
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        
        # Example: Simple greedy approach (replace with your algorithm)
        row_diff = enemy_position[0] - my_position[0]
        col_diff = enemy_position[1] - my_position[1]
        
        # Try to move towards ghost
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
        
        # Check if move is valid
        if self._is_valid_move(my_position, move, map_state):
            return move
        
        # If not valid, try other moves
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                return move
        
        return Move.STAY
    
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

class GhostAgent(BaseGhostAgent):
    """
    Smart Ghost v4.1 – Rule-Aware + Map Learning (Junction Graph v4.1)

    CHIẾN LƯỢC:
    - Bỏ hoàn toàn Potential Field.
    - "Ở xa" (distance >= 7): Dùng Junction Graph Escape (Nâng cấp).
      Chủ động tìm (A*) đến ngã rẽ "TỐT NHẤT" (vừa xa, vừa an toàn -
      nhiều lối thoát), đặc biệt né các ngõ cụt (degree 1).
    - "Ở gần" (distance < 7): Dùng Rule-Aware Minimax.
      Minimax hiểu luật: pacman_speed, max_steps, capture_threshold.

    CÀI ĐẶT QUAN TRỌNG:
    Bạn PHẢI sửa các giá trị trong hàm __init__ (dòng 50-52)
    để khớp với các tham số (--pacman-speed, --max-steps, v.v.)
    mà bạn dùng để chạy arena.py.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Smart Ghost v4.1 (Map Learner)"

        # === ‼️ CÀI ĐẶT LUẬT CHƠI (PHẢI KHỚP VỚI ARENA.PY) ===
        # Sửa các giá trị này cho khớp với lệnh bạn chạy test.
        # Ví dụ: Nếu chạy --pacman-speed 2 --capture-distance 2
        
        self.pacman_speed = 2      # Sửa thành 2 nếu Pacman chạy 2 bước
        self.max_steps = 200       # Sửa nếu bạn chạy --max-steps
        self.capture_threshold = 2 # Sửa thành 2 nếu Pacman bắt ở dist < 2
        
        # ==========================================================

        # === Hằng số chiến lược ===
        self.MINIMAX_DEPTH = 4  # Tăng lên 6 nếu timeout (1.0s) cho phép
        self.MINIMAX_TRIGGER_DISTANCE = 7 # Tăng tầm nhìn

        # === Biến trạng thái & Cache ===
        self._map_height = 0
        self._map_width = 0
        self._map_state_cached = None 
        
        # Cache cho Junction Graph (để "học" map)
        self._junction_nodes = None
        self._junction_degrees = {} # Lưu độ "xịn" (số lối thoát) của ngã rẽ
        self._current_escape_path = deque() 
        self._escape_target_node = None 

    # ===================================================================
    # ========== BỘ NÃO CHÍNH (HYBRID STEP) ==========
    # ===================================================================

    def step(self, map_state, my_pos, pac_pos, step_number):
        # 1. Khởi tạo/Học map (chỉ làm 1 lần)
        if self._map_state_cached is None or not np.array_equal(self._map_state_cached, map_state):
            self._map_state_cached = map_state
            self._map_height, self._map_width = map_state.shape
            self._build_junction_graph(map_state) # "Học" map ngay lần đầu

        if my_pos == pac_pos:
            return Move.STAY

        # 2. Tính khoảng cách
        distance_to_pacman = self._manhattan_distance(my_pos, pac_pos)

        # 3. Quyết định chiến lược
        if distance_to_pacman < self.MINIMAX_TRIGGER_DISTANCE:
            # === CHIẾN LƯỢC GẦN: RULE-AWARE MINIMAX ===
            self._current_escape_path.clear() 
            self._escape_target_node = None

            _, best_move = self._minimax(
                map_state, pac_pos, my_pos,
                depth=self.MINIMAX_DEPTH,
                is_ghost_turn=True,
                step_number=step_number
            )
            return best_move if best_move else Move.STAY
        else:
            # === CHIẾN LƯỢC XA: JUNCTION GRAPH ESCAPE (v4.1) ===
            return self._junction_graph_escape_move(map_state, my_pos, pac_pos)

    # ===================================================================
    # ========== CHIẾN LƯỢC 1: RULE-AWARE MINIMAX ==========
    # ===================================================================

    def _minimax(self, map_state, pacman_pos, ghost_pos, depth, is_ghost_turn,
                 step_number, alpha=-float('inf'), beta=float('inf')):
        
        # === ĐIỀU KIỆN DỪNG (ĐÃ BIẾT LUẬT) ===
        
        # 1. Bị bắt? (Dùng luật capture_threshold)
        current_dist_to_pacman = self._manhattan_distance(pacman_pos, ghost_pos)
        if current_dist_to_pacman < self.capture_threshold:
            return -float('inf'), None  # Thua

        # 2. Hết giờ? (Dùng luật max_steps)
        estimated_current_game_turn = step_number + (self.MINIMAX_DEPTH - depth) // 2
        if estimated_current_game_turn >= self.max_steps:
            return float('inf'), None  # Thắng (câu giờ thành công)

        # 3. Hết độ sâu tìm kiếm?
        if depth == 0:
            return self._heuristic_eval(ghost_pos, pacman_pos, map_state), None

        # === LƯỢT CỦA GHOST (MAX) ===
        if is_ghost_turn:
            max_eval = -float('inf')
            best_move = Move.STAY
            for next_ghost_pos, move in self._get_valid_moves(ghost_pos, map_state):
                eval_score, _ = self._minimax(
                    map_state, pac_pos, next_ghost_pos,
                    depth - 1, False, step_number + 1, alpha, beta
                )
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return max_eval, best_move

        # === LƯỢT CỦA PACMAN (MIN) (Đã biết pacman_speed) ===
        else:
            min_eval = float('inf')
            for next_pacman_pos, _ in self._get_pacman_simulated_moves(pacman_pos, map_state):
                eval_score, _ = self._minimax(
                    map_state, next_pacman_pos, ghost_pos,
                    depth - 1, True, step_number + 1, alpha, beta
                )
                if eval_score < min_eval:
                    min_eval = eval_score
                beta = min(beta, eval_score)
                if beta <= alpha: break

            if min_eval == float('inf'): # Fallback nếu Pacman bị kẹt
                 eval_score, _ = self._minimax(
                    map_state, pacman_pos, ghost_pos,
                    depth - 1, True, step_number + 1, alpha, beta
                )
                 min_eval = eval_score
            return min_eval, None

    def _heuristic_eval(self, ghost_pos, pacman_pos, map_state):
        """Hàm đánh giá: Ghost muốn điểm cao."""
        distance_to_pacman = self._manhattan_distance(ghost_pos, pacman_pos)

        if distance_to_pacman < self.capture_threshold:
            return -float('inf')

        score = distance_to_pacman * 10.0
        openness = self._flood_fill_reachable_tiles(map_state, ghost_pos, limit=15)
        score += openness * 3.0
        adjacent_wall_count = self._count_adjacent_walls(ghost_pos, map_state)
        score -= adjacent_wall_count * 2.0
        return score

    def _get_pacman_simulated_moves(self, pacman_pos, map_state):
        """Mô phỏng các nước đi của Pacman (với pacman_speed)."""
        possible_moves = []
        for initial_move_dir in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            final_pos = self._simulate_single_pacman_dash(
                pacman_pos, initial_move_dir, map_state, self.pacman_speed
            )
            if final_pos != pacman_pos:
                possible_moves.append((final_pos, initial_move_dir))
        
        if not possible_moves:
            possible_moves.append((pacman_pos, Move.STAY))
        return possible_moves

    def _simulate_single_pacman_dash(self, start_pos, move_direction, map_state, speed):
        """Mô phỏng cú "lướt" N bước của Pacman."""
        current_pos = start_pos
        dr, dc = move_direction.value

        for _ in range(speed):
            next_pos = (current_pos[0] + dr, current_pos[1] + dc)
            if not self._is_valid_position(next_pos, map_state):
                break # Dừng khi đụng tường
            current_pos = next_pos
            if len(self._get_valid_moves(current_pos, map_state)) > 2:
                break # Dừng khi tới ngã rẽ
        return current_pos

    # ===================================================================
    # ========== STRATEGY 2: JUNCTION GRAPH ESCAPE (v4.1) ==========
    # ===================================================================

    def _junction_graph_escape_move(self, map_state, my_pos, pac_pos):
        """
        Chiến lược "ở xa" v4.1: Chạy đến ngã rẽ "TỐT NHẤT".
        "TỐT NHẤT" = (Độ an toàn * 100) + Khoảng cách
        """
        
        replan_needed = False
        if not self._current_escape_path:
            replan_needed = True
        elif self._escape_target_node and self._manhattan_distance(pac_pos, self._escape_target_node) < self.MINIMAX_TRIGGER_DISTANCE:
            replan_needed = True # Pacman quá gần mục tiêu -> Đổi mục tiêu

        if replan_needed:
            best_node = None
            best_score = -float('inf') 

            if not self._junction_nodes: # Fallback nếu map không có ngã rẽ
                return self._get_random_safe_move(my_pos, pac_pos, map_state)

            for node in self._junction_nodes:
                dist_to_pacman = self._manhattan_distance(node, pac_pos)
                degree = self._junction_degrees.get(node, 0)

                # KHÔNG BAO GIỜ chủ động chạy vào ngõ cụt (degree <= 1)
                # Đây là cách "học" map của bạn
                if degree <= 1: 
                    continue
                
                # Công thức điểm: Ưu tiên (Degree * 100) rồi mới tới (Khoảng cách)
                score = (degree * 100) + dist_to_pacman
                
                if score > best_score:
                    best_score = score
                    best_node = node
            
            if best_node:
                self._escape_target_node = best_node
                path_moves = self._astar_path_to_target(my_pos, self._escape_target_node, map_state)
                self._current_escape_path = deque(path_moves)
            else:
                self._current_escape_path.clear()

        # Thực thi nước đi
        if self._current_escape_path:
            return self._current_escape_path.popleft()
        else:
            # Fallback nếu hết đường hoặc không tìm thấy
            return self._get_random_safe_move(my_pos, pac_pos, map_state)

    def _build_junction_graph(self, map_state):
        """
        "Học map": Tìm tất cả ngã rẽ VÀ lưu trữ "độ an toàn" (degree)
        của chúng.
        """
        h, w = self._map_height, self._map_width
        junctions = set()
        degrees = {}

        for r in range(h):
            for c in range(w):
                pos = (r, c)
                if self._is_wall(pos, map_state):
                    continue
                
                degree = len(self._get_valid_moves(pos, map_state))
                
                # Ngã rẽ = không phải đường thẳng (degree != 2)
                if degree != 2:
                    junctions.add(pos)
                    degrees[pos] = degree
                    
        self._junction_nodes = junctions
        self._junction_degrees = degrees # Đã "học" độ an toàn

    def _astar_path_to_target(self, start, goal, map_state):
        """A* tìm đường (Tile-level), trả về list[Move]."""
        open_heap = []
        heapq.heappush(open_heap, (0, start, [])) # (f_score, pos, path)
        g_score = {start: 0}
        visited = set()

        while open_heap:
            f_score, current_pos, path_so_far = heapq.heappop(open_heap)

            if self._manhattan_distance(current_pos, goal) <= 0:
                return path_so_far

            if current_pos in visited:
                continue
            visited.add(current_pos)

            for next_pos, move in self._get_valid_moves(current_pos, map_state):
                tentative_g_score = g_score[current_pos] + 1
                if tentative_g_score < g_score.get(next_pos, float('inf')):
                    g_score[next_pos] = tentative_g_score
                    f_score = tentative_g_score + self._manhattan_distance(next_pos, goal)
                    heapq.heappush(open_heap, (f_score, next_pos, path_so_far + [move]))
        return [] # Không tìm thấy đường

    # ===================================================================
    # ========== CÁC HÀM HỖ TRỢ (UTILITY) ==========
    # ===================================================================

    def _is_wall(self, pos: tuple, map_state: np.ndarray) -> bool:
        r, c = pos
        if not (0 <= r < self._map_height and 0 <= c < self._map_width):
            return True  # Ngoài bản đồ là tường
        return map_state[r, c] != 0 

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        return not self._is_wall(pos, map_state)

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _get_valid_moves(self, pos: tuple, map_state: np.ndarray) -> list[tuple[tuple, Move]]:
        """Trả về list[(next_pos, move)]."""
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                valid_moves.append((next_pos, move))
        return valid_moves

    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _flood_fill_reachable_tiles(self, map_state: np.ndarray, start_pos: tuple, limit: int = 20) -> int:
        """Đếm số ô trống xung quanh (độ thoáng)."""
        q = deque([start_pos])
        visited = {start_pos}
        count = 0
        while q and count < limit:
            curr_pos = q.popleft()
            count += 1
            for next_pos, _ in self._get_valid_moves(curr_pos, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    q.append(next_pos)
        return count

    def _count_adjacent_walls(self, pos: tuple, map_state: np.ndarray) -> int:
        """Đếm số tường xung quanh."""
        wall_count = 0
        for move_dir in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_wall(self._apply_move(pos, move_dir), map_state):
                wall_count += 1
        return wall_count

    def _get_random_safe_move(self, my_pos, pac_pos, map_state):
        """Fallback: Chọn nước đi ngẫu nhiên an toàn (ưu tiên chạy xa)."""
        valid_moves = self._get_valid_moves(my_pos, map_state)
        best_moves = []
        max_dist_increase = -float('inf')
        
        for next_pos, move in valid_moves:
            new_dist = self._manhattan_distance(next_pos, pac_pos)
            old_dist = self._manhattan_distance(my_pos, pac_pos)
            dist_increase = new_dist - old_dist

            if dist_increase > max_dist_increase:
                max_dist_increase = dist_increase
                best_moves = [(next_pos, move)]
            elif dist_increase == max_dist_increase:
                best_moves.append((next_pos, move))
        
        if best_moves:
            return random.choice(best_moves)[1] # Trả về Move
        return Move.STAY
    """
    Smart Ghost v4.0 – Rule-Aware Hybrid Agent

    This agent combines two core strategies:
    1.  Junction Graph Escape for long distances: When Pacman is far away,
        the Ghost actively plans a path to the furthest available junction
        to maximize evasion time, abandoning the reactive Potential Field.
    2.  Rule-Aware Minimax for short distances: When Pacman is close,
        Minimax is used to predict optimal evasion. This Minimax is
        "rule-aware," meaning it incorporates the actual game rules:
        -   Pacman's variable speed (`pacman_speed`).
        -   The total `max_steps` for a draw (allowing strategic stalling).
        -   The exact `capture_threshold` for determining a loss.

    Configuration Note:
    The `self.pacman_speed`, `self.max_steps`, and `self.capture_threshold`
    parameters in the `__init__` method MUST be manually set to match
    the parameters used when running `arena.py` for optimal performance.
    If left at default, the Ghost will make decisions based on potentially
    incorrect game rules.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Smart Ghost v4.0 (Rule-Aware Hybrid)"

        # === CONFIGURE GAME RULES (MUST MATCH ARENA.PY SETTINGS) ===
        # These values determine how the Ghost perceives the game rules.
        # Adjust them to match the --pacman-speed, --max-steps, and
        # --capture-distance arguments you use when running arena.py.
        self.pacman_speed = 1  # e.g., if you run with `--pacman-speed 2`, change this to 2
        self.max_steps = 200    # e.g., if you run with `--max-steps 300`, change this to 300
        self.capture_threshold = 1 # e.g., if you run with `--capture-distance 2`, change this to 2

        # === STRATEGY HYPERPARAMETERS ===
        self.MINIMAX_DEPTH = 4  # Adjust based on step_timeout (e.g., to 6 or 8 for 1.0s timeout)
        self.MINIMAX_TRIGGER_DISTANCE = 7 # Distance threshold to switch to Minimax

        # === INTERNAL STATE AND CACHES ===
        self._map_height = 0
        self._map_width = 0
        self._map_state_cached = None # To avoid re-processing map data

        # Junction Graph for Escape Strategy (Similar to Pacman's approach)
        self._junction_nodes = None  # Set of (r, c) tuples representing junction coordinates
        self._current_escape_path = deque() # Path from A* for escape
        self._escape_target_node = None # The specific junction node being targeted

    # ===================================================================
    # ========== MAIN AGENT STEP LOGIC ==========
    # ===================================================================

    def step(self, map_state, my_pos, pac_pos, step_number):
        # Cache map dimensions and build junction graph only once
        if self._map_state_cached is None or not np.array_equal(self._map_state_cached, map_state):
            self._map_state_cached = map_state
            self._map_height, self._map_width = map_state.shape
            self._build_junction_graph(map_state) # Build once, or if map changes

        # Safety check: if Ghost is already on Pacman's tile, stay put (likely game over)
        if my_pos == pac_pos:
            return Move.STAY

        # Determine strategy based on Manhattan distance
        distance_to_pacman = self._manhattan_distance(my_pos, pac_pos)

        if distance_to_pacman < self.MINIMAX_TRIGGER_DISTANCE:
            # === STRATEGY: SHORT-RANGE EVASION (Rule-Aware Minimax) ===
            # Clear any active escape path, as Minimax takes over.
            self._current_escape_path.clear()
            self._escape_target_node = None

            _, best_move = self._minimax(
                map_state, pac_pos, my_pos,
                depth=self.MINIMAX_DEPTH,
                is_ghost_turn=True,
                step_number=step_number
            )
            # Fallback to STAY if minimax can't find a move (should be rare)
            return best_move if best_move else Move.STAY
        else:
            # === STRATEGY: LONG-RANGE EVASION (Junction Graph Escape) ===
            return self._junction_graph_escape_move(map_state, my_pos, pac_pos)

    # ===================================================================
    # ========== STRATEGY 1: RULE-AWARE MINIMAX ==========
    # ===================================================================

    def _minimax(self, map_state, pacman_pos, ghost_pos, depth, is_ghost_turn,
                 step_number, alpha=-float('inf'), beta=float('inf')):
        """
        Minimax algorithm for optimal evasion, aware of game rules.
        """

        # === TERMINAL STATES (GAME RULES) ===
        # 1. Capture Condition: Ghost is caught (using actual capture_threshold)
        current_dist_to_pacman = self._manhattan_distance(pacman_pos, ghost_pos)
        if current_dist_to_pacman < self.capture_threshold:
            return -float('inf'), None  # Ghost loses, minimize score

        # 2. Draw Condition: Max steps reached (Ghost wins by stalling)
        # We approximate remaining steps based on current depth.
        # (self.MINIMAX_DEPTH - depth) / 2 is approximate game turns elapsed in minimax
        estimated_current_game_turn = step_number + (self.MINIMAX_DEPTH - depth) // 2
        if estimated_current_game_turn >= self.max_steps:
            return float('inf'), None  # Ghost wins by draw, maximize score

        # 3. Depth Limit: Evaluate with heuristic
        if depth == 0:
            return self._heuristic_eval(ghost_pos, pacman_pos, map_state), None

        # === GHOST'S TURN (MAXIMIZING PLAYER) ===
        if is_ghost_turn:
            max_eval = -float('inf')
            best_move = Move.STAY

            # Iterate over valid Ghost moves
            for next_ghost_pos, move in self._get_valid_moves(ghost_pos, map_state):
                eval_score, _ = self._minimax(
                    map_state, pacman_pos, next_ghost_pos,
                    depth - 1, False, step_number + 1, alpha, beta
                )
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:  # Alpha-beta pruning
                    break
            return max_eval, best_move

        # === PACMAN'S TURN (MINIMIZING PLAYER) ===
        else:
            min_eval = float('inf')

            # Iterate over valid Pacman moves (aware of pacman_speed)
            for next_pacman_pos, _ in self._get_pacman_simulated_moves(pacman_pos, map_state):
                eval_score, _ = self._minimax(
                    map_state, next_pacman_pos, ghost_pos,
                    depth - 1, True, step_number + 1, alpha, beta
                )
                if eval_score < min_eval:
                    min_eval = eval_score
                beta = min(beta, eval_score)
                if beta <= alpha:  # Alpha-beta pruning
                    break

            # Fallback if Pacman has no valid moves (e.g., completely surrounded)
            if min_eval == float('inf'):
                 eval_score, _ = self._minimax(
                    map_state, pacman_pos, ghost_pos,
                    depth - 1, True, step_number + 1, alpha, beta
                )
                 min_eval = eval_score

            return min_eval, None

    def _heuristic_eval(self, ghost_pos, pacman_pos, map_state):
        """
        Heuristic evaluation function for the Ghost (maximizing player).
        Higher score means better state for the Ghost.
        """
        distance_to_pacman = self._manhattan_distance(ghost_pos, pacman_pos)

        # Critical: If already captured, it's the worst possible state
        if distance_to_pacman < self.capture_threshold:
            return -float('inf')

        # Reward for being further from Pacman
        score = distance_to_pacman * 10.0

        # Reward for open space around the Ghost (freedom of movement)
        openness = self._flood_fill_reachable_tiles(map_state, ghost_pos, limit=15)
        score += openness * 1.5

        # Penalize for being near walls (less escape routes)
        adjacent_wall_count = self._count_adjacent_walls(ghost_pos, map_state)
        score -= adjacent_wall_count * 2.0

        return score

    def _get_pacman_simulated_moves(self, pacman_pos, map_state):
        """
        Simulates all possible next positions Pacman can reach in one turn,
        considering its `pacman_speed` and map constraints.
        Returns a list of `(final_position, original_move_direction)`.
        """
        possible_moves = []
        for initial_move_dir in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            final_pos = self._simulate_single_pacman_dash(
                pacman_pos, initial_move_dir, map_state, self.pacman_speed
            )
            if final_pos != pacman_pos: # Only add if Pacman actually moved
                possible_moves.append((final_pos, initial_move_dir))
        
        # If Pacman can't move anywhere, its position remains the same
        if not possible_moves:
            possible_moves.append((pacman_pos, Move.STAY))
        return possible_moves

    def _simulate_single_pacman_dash(self, start_pos, move_direction, map_state, speed):
        """
        Simulates Pacman's movement for one turn along a straight path.
        Pacman stops if it hits a wall, reaches its max speed, or encounters a junction.
        """
        current_pos = start_pos
        dr, dc = move_direction.value

        for _ in range(speed):
            next_pos = (current_pos[0] + dr, current_pos[1] + dc)

            # Check for wall collision
            if not self._is_valid_position(next_pos, map_state):
                break # Stop at wall

            current_pos = next_pos

            # Check if current_pos is a junction (more than 2 valid neighbors)
            # Pacman typically has to "decide" at a junction, so it stops there.
            if len(self._get_valid_moves(current_pos, map_state)) > 2:
                break # Stop at junction
        return current_pos

    # ===================================================================
    # ========== STRATEGY 2: JUNCTION GRAPH ESCAPE ==========
    # ===================================================================

    def _junction_graph_escape_move(self, map_state, my_pos, pac_pos):
        """
        Strategy to escape using a pre-computed junction graph.
        The Ghost tries to reach the junction furthest from Pacman.
        """
        # 1. Replanning logic:
        #    - If current path is empty
        #    - If the target junction is too close to Pacman (Pacman is closing in)
        replan_needed = False
        if not self._current_escape_path:
            replan_needed = True
        elif self._escape_target_node and self._manhattan_distance(pac_pos, self._escape_target_node) < self.MINIMAX_TRIGGER_DISTANCE + 2:
            replan_needed = True # Pacman is too close to our escape target

        if replan_needed:
            # Find the best new escape target (junction furthest from Pacman)
            best_node = None
            max_dist_to_pacman_for_node = -float('inf')

            if not self._junction_nodes: # Fallback if no junctions found (e.g., small map)
                return self._get_random_safe_move(my_pos, map_state)

            for node in self._junction_nodes:
                dist = self._manhattan_distance(node, pac_pos)
                if dist > max_dist_to_pacman_for_node:
                    max_dist_to_pacman_for_node = dist
                    best_node = node

            if best_node:
                self._escape_target_node = best_node
                # Use A* to find path to this target junction
                path_moves = self._astar_path_to_target(my_pos, self._escape_target_node, map_state)
                self._current_escape_path = deque(path_moves)
            else:
                self._current_escape_path.clear() # No suitable target found

        # 2. Execute the next move on the planned path
        if self._current_escape_path:
            return self._current_escape_path.popleft()
        else:
            # Fallback if no path found or path completed (e.g., target reached)
            # Find a random safe move away from Pacman
            return self._get_random_safe_move(my_pos, pac_pos, map_state)

    def _build_junction_graph(self, map_state):
        """
        Identifies all junction nodes (tiles with degree != 2) on the map.
        This is a preprocessing step for the escape strategy.
        """
        h, w = self._map_height, self._map_width
        junctions = set()

        for r in range(h):
            for c in range(w):
                pos = (r, c)
                if self._is_wall(pos, map_state):
                    continue

                # A junction is a tile that is not a wall and has
                # a degree (number of valid neighbors) other than 2.
                # Degree 2 implies a straight path.
                degree = len(self._get_valid_moves(pos, map_state))
                if degree != 2:
                    junctions.add(pos)
        self._junction_nodes = junctions

    def _astar_path_to_target(self, start, goal, map_state):
        """
        Finds the shortest path in terms of moves from start to goal (tile-level A*).
        Returns a list of `Move` objects.
        """
        open_heap = []
        # (f_score, current_pos, path_to_current_pos)
        heapq.heappush(open_heap, (0, start, []))
        
        # g_score: cost from start to current_pos
        g_score = {start: 0}
        
        # Keep track of visited positions to avoid cycles and redundant processing
        # Using a set for faster lookup
        visited = set()

        while open_heap:
            f_score, current_pos, path_so_far = heapq.heappop(open_heap)

            if self._manhattan_distance(current_pos, goal) <= 0: # Reached or passed goal
                return path_so_far

            if current_pos in visited:
                continue
            visited.add(current_pos)

            for next_pos, move in self._get_valid_moves(current_pos, map_state):
                tentative_g_score = g_score[current_pos] + 1  # Each move costs 1

                # If this path to next_pos is better than any previous one
                if tentative_g_score < g_score.get(next_pos, float('inf')):
                    g_score[next_pos] = tentative_g_score
                    # f_score = g_score + heuristic (Manhattan distance to goal)
                    f_score = tentative_g_score + self._manhattan_distance(next_pos, goal)
                    heapq.heappush(open_heap, (f_score, next_pos, path_so_far + [move]))
        return [] # No path found

    # ===================================================================
    # ========== UTILITY FUNCTIONS ==========
    # ===================================================================

    def _is_wall(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Checks if a given position is a wall or out of bounds."""
        r, c = pos
        if not (0 <= r < self._map_height and 0 <= c < self._map_width):
            return True  # Out of bounds is considered a wall
        return map_state[r, c] != 0 # Assuming 0 is traversable, non-zero is wall

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Checks if a position is within bounds and not a wall."""
        return not self._is_wall(pos, map_state)

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        """Applies a Move enum to a position tuple."""
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _get_valid_moves(self, pos: tuple, map_state: np.ndarray) -> list[tuple[tuple, Move]]:
        """
        Returns a list of `(next_position, move)` tuples for all valid moves
        from the current position.
        """
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                valid_moves.append((next_pos, move))
        return valid_moves

    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Calculates Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _flood_fill_reachable_tiles(self, map_state: np.ndarray, start_pos: tuple, limit: int = 20) -> int:
        """
        Performs a limited flood fill to count reachable tiles from a start_pos.
        Used for evaluating 'openness' heuristic.
        """
        q = deque([start_pos])
        visited = {start_pos}
        count = 0

        while q and count < limit:
            curr_pos = q.popleft()
            count += 1
            for next_pos, _ in self._get_valid_moves(curr_pos, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    q.append(next_pos)
        return count

    def _count_adjacent_walls(self, pos: tuple, map_state: np.ndarray) -> int:
        """Counts how many adjacent tiles are walls (including out of bounds)."""
        wall_count = 0
        for move_dir in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            neighbor_pos = self._apply_move(pos, move_dir)
            if self._is_wall(neighbor_pos, map_state):
                wall_count += 1
        return wall_count

    def _get_random_safe_move(self, my_pos, pac_pos, map_state):
        """
        Fallback: Returns a random valid move that tries to increase distance to Pacman.
        """
        valid_moves = self._get_valid_moves(my_pos, map_state)
        
        # Prioritize moves that increase distance
        best_moves = []
        max_dist_increase = -float('inf')

        for next_pos, move in valid_moves:
            new_dist = self._manhattan_distance(next_pos, pac_pos)
            old_dist = self._manhattan_distance(my_pos, pac_pos)
            dist_increase = new_dist - old_dist

            if dist_increase > max_dist_increase:
                max_dist_increase = dist_increase
                best_moves = [(next_pos, move)]
            elif dist_increase == max_dist_increase:
                best_moves.append((next_pos, move))
        
        if best_moves:
            return random.choice(best_moves)[1] # Return the Move object
        return Move.STAY # If no safe move found
    """
    Smart Ghost v3.0 – Hybrid (Minimax + Potential Field)
    
    CHIẾN LƯỢC:
    - Khi Pacman ở xa (distance >= 6): Dùng Potential Field (v2.1) để
      phản ứng nhanh, giữ đà, và né tường.
    - Khi Pacman ở gần (distance < 6): Kích hoạt Minimax (depth=4) 
      sử dụng hàm heuristic_eval (khoảng cách + độ thoáng + né tường)
      để dự đoán và tránh bẫy.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Smart Ghost v3.0 (Hybrid)"
        
        # === Hằng số cho Potential Field (từ v2.1) ===
        self.K = 200.0
        self.wall_penalty = 3.0
        self.escape_bias = 1.6
        self.momentum_weight = 2.5
        self.random_eps = 0.08
        self.safe_distance = 4  # critical radius
        
        # === Hằng số cho Minimax (mới) ===
        self.MINIMAX_DEPTH = 4 # Độ sâu tìm kiếm
        self.MINIMAX_TRIGGER_DISTANCE = 6 # Khoảng cách kích hoạt
        
        # === Biến trạng thái ===
        self.last_move = Move.STAY
        self.last_pacman_pos = None

    # ===================================================================
    # ========== BỘ NÃO CHÍNH (HYBRID STEP) ==========
    # ===================================================================

    def step(self, map_state, my_pos, pac_pos, step_number):
        if my_pos == pac_pos:
            return Move.STAY

        # 1. Tính khoảng cách (dùng Manhattan cho nhanh và chuẩn)
        distance = self._manhattan_distance(my_pos, pac_pos)

        # 2. Quyết định chiến lược
        if distance < self.MINIMAX_TRIGGER_DISTANCE:
            # === CHIẾN LƯỢC GẦN: MINIMAX (Dự đoán) ===
            # Tìm nước đi tốt nhất bằng cách giả định Pacman chơi tối ưu
            _, best_move = self.minimax(
                map_state, pac_pos, my_pos,
                depth=self.MINIMAX_DEPTH, is_ghost_turn=True
            )
            final_move = best_move if best_move else Move.STAY
            
            # Fallback: Nếu Minimax bị kẹt (STAY), thử đi tiếp hướng cũ
            if final_move == Move.STAY:
                if self._valid(self._apply_move(my_pos, self.last_move), map_state):
                    final_move = self.last_move
                
        else:
            # === CHIẾN LƯỢC XA: POTENTIAL FIELD (Phản ứng) ===
            # Dùng lại code v2.1 của bạn
            final_move = self._potential_field_move(map_state, my_pos, pac_pos)

        # 3. Cập nhật và trả về
        self.last_move = final_move
        self.last_pacman_pos = pac_pos
        return final_move

    # ===================================================================
    # ========== CHIẾN LƯỢC 1: MINIMAX (Nâng cấp) ==========
    # ===================================================================

    def _heuristic_eval(self, ghost_pos, pacman_pos, map_state):
        """
        Hàm đánh giá "Ultimate" cho Minimax (tại lá).
        Ghost muốn TỐI ĐA HÓA điểm này.
        """
        # 1. Khoảng cách (Manhattan)
        distance = self._manhattan_distance(ghost_pos, pacman_pos)
        if distance == 0:
            return -float('inf') # Bị bắt = Rất tệ

        # 2. Độ thoáng (dùng hàm _flood_fill_space)
        # Cho tầm nhìn xa hơn (limit=15)
        openness = self._flood_fill_space(map_state, ghost_pos, limit=15) 

        # 3. Né tường (dùng hàm _adjacent_wall_count)
        wall_count = self._adjacent_wall_count(ghost_pos, map_state)

        # CÔNG THỨC ĐIỂM (Bạn có thể tinh chỉnh các trọng số này)
        # Ưu tiên: Xa Pacman, Chỗ thoáng, Ít tường
        score = (distance * 10.0) + (openness * 3.0) - (wall_count * 2.0)
        
        return score

    def minimax(self, map_state, pacman_pos, ghost_pos, depth, is_ghost_turn, alpha=-float('inf'), beta=float('inf')):
        """
        Minimax từ góc nhìn Ghost, sử dụng _heuristic_eval.
        """
        if depth == 0 or pacman_pos == ghost_pos:
            # Đến lá, gọi hàm đánh giá
            score = self._heuristic_eval(ghost_pos, pacman_pos, map_state)
            return score, None
        
        if is_ghost_turn:
            # Lượt của Ghost (Max) - Muốn điểm heuristic cao nhất
            max_eval = -float('inf')
            best_move = Move.STAY
            
            for next_pos, move in self._get_neighbors(ghost_pos, map_state):
                eval_score, _ = self.minimax(
                    map_state, pacman_pos, next_pos,
                    depth - 1, False, alpha, beta # Lượt sau là của Pacman
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        
        else: # Lượt của Pacman (Min)
            # Pacman muốn điểm heuristic (của Ghost) thấp nhất
            min_eval = float('inf')
            
            for next_pos, move in self._get_neighbors(pacman_pos, map_state):
                eval_score, _ = self.minimax(
                    map_state, next_pos, ghost_pos,
                    depth - 1, True, alpha, beta # Lượt sau là của Ghost
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, None

    # ===================================================================
    # ========== CHIẾN LƯỢC 2: POTENTIAL FIELD (Code v2.1) ==========
    # ===================================================================

    def _potential_field_move(self, map_state, my_pos, pac_pos):
        """
        Đây là toàn bộ logic từ hàm step() v2.1 của bạn.
        """
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        best_move = Move.STAY
        best_score = float("inf")
        fallback_move = Move.STAY

        predicted_pac = self._predict_pacman(pac_pos)
        dist_now = self._euclid_dist(my_pos, pac_pos)

        for move in moves:
            dr, dc = move.value
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            if not self._valid(new_pos, map_state):
                continue

            new_dist = self._euclid_dist(new_pos, pac_pos)
            
            # danger filter
            if dist_now <= self.safe_distance and new_dist < dist_now - 0.1:
                continue

            pot = self._repulsive_potential(new_pos, predicted_pac)
            wall_p = self._adjacent_wall_count(new_pos, map_state) * self.wall_penalty
            openness = self._flood_fill_space(map_state, new_pos) # Tự động dùng limit=8 (mặc định)
            escape_bonus = -openness * self.escape_bias
            momentum_bonus = -self.momentum_weight if move == self.last_move else 0
            
            jitter_strength = self.random_eps * self.K * (1 if dist_now > self.safe_distance else 0.3)
            jitter = random.uniform(-1, 1) * jitter_strength

            score = pot + wall_p + escape_bonus + momentum_bonus + jitter

            if score < best_score:
                best_score = score
                best_move = move

            # Fallback
            if new_dist > self._euclid_dist(my_pos, self._apply_move(my_pos, fallback_move)):
                fallback_move = move

        final_move = best_move if best_move != Move.STAY else fallback_move

        if final_move == Move.STAY:
            final_move = self.last_move
        
        return final_move

    # ===================================================================
    # ========== CÁC HÀM HỖ TRỢ (TỔNG HỢP) ==========
    # ===================================================================

    # --- HÀM MỚI/TỪ AGENT 1 ---
    
    def _manhattan_distance(self, pos1, pos2):
        """Tính khoảng cách Manhattan."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_neighbors(self, pos, map_state):
        """Lấy các nước đi hợp lệ (dùng hàm _valid và _apply_move)"""
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._valid(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors

    # --- CÁC HÀM CŨ TỪ V2.1 (Giữ nguyên) ---

    def _predict_pacman(self, pac_pos):
        if not self.last_pacman_pos:
            return pac_pos
        dr = pac_pos[0] - self.last_pacman_pos[0]
        dc = pac_pos[1] - self.last_pacman_pos[1]
        return (pac_pos[0] + dr, pac_pos[1] + dc)

    def _repulsive_potential(self, pos, pac_pos):
        dist = self._euclid_dist(pos, pac_pos)
        if dist <= 0:
            return float("inf")
        return self.K / (dist ** 1.4)

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _valid(self, pos, map_state):
        h, w = map_state.shape
        r, c = pos
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    def _adjacent_wall_count(self, pos, map_state):
        count = 0
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        h, w = map_state.shape
        for dr, dc in dirs:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < h and 0 <= nc < w) or map_state[nr, nc] != 0:
                count += 1
        return count

    def _flood_fill_space(self, map_state, start, limit=8): # **Đã sửa**
        """
        Đếm số ô trống xung quanh (với limit).
        Mặc định limit=8 (cho Potential Field).
        Minimax có thể gọi với limit cao hơn.
        """
        h, w = map_state.shape
        q = [start]
        visited = {start}
        count = 0
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        while q and count < limit: # Dùng limit truyền vào
            r, c = q.pop(0)
            count += 1
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] == 0 and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return count

    def _euclid_dist(self, a, b):
        """Vẫn cần cho Potential Field."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    """
    Smart Ghost v2.1 – Fixed Potential Field
    • Không đứng yên khi bị đuổi gần
    • Không chạy ngược về phía Pacman
    • Né vuông góc + wall hugging + openness bias
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Smart Ghost v2.1 (Safe Evasion)"
        self.K = 200.0
        self.wall_penalty = 3.0
        self.escape_bias = 1.6
        self.momentum_weight = 2.5
        self.random_eps = 0.08
        self.safe_distance = 4  # critical radius
        self.last_move = Move.STAY
        self.last_pacman_pos = None

    def step(self, map_state, my_pos, pac_pos, step_number):
        if my_pos == pac_pos:
            return Move.STAY

        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        best_move = Move.STAY
        best_score = float("inf")
        fallback_move = Move.STAY

        predicted_pac = self._predict_pacman(pac_pos)
        dist_now = self._euclid_dist(my_pos, pac_pos)

        for move in moves:
            dr, dc = move.value
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            if not self._valid(new_pos, map_state):
                continue

            new_dist = self._euclid_dist(new_pos, pac_pos)
            # ------------------ danger filter ------------------
            # Nếu Pacman gần và bước này làm khoảng cách giảm → bỏ
            if dist_now <= self.safe_distance and new_dist < dist_now - 0.1:
                continue

            # Base potential field
            pot = self._repulsive_potential(new_pos, predicted_pac)
            wall_p = self._adjacent_wall_count(new_pos, map_state) * self.wall_penalty
            openness = self._flood_fill_space(map_state, new_pos)
            escape_bonus = -openness * self.escape_bias

            # Momentum (ưu tiên hướng đang đi)
            momentum_bonus = -self.momentum_weight if move == self.last_move else 0

            # Random noise giảm dần khi gần Pacman
            jitter_strength = self.random_eps * self.K * (1 if dist_now > self.safe_distance else 0.3)
            jitter = random.uniform(-1, 1) * jitter_strength

            score = pot + wall_p + escape_bonus + momentum_bonus + jitter

            # Track best
            if score < best_score:
                best_score = score
                best_move = move

            # Fallback: chọn hướng xa Pacman nhất
            if new_dist > self._euclid_dist(my_pos, self._apply_move(my_pos, fallback_move)):
                fallback_move = move

        # Nếu không có hướng hợp lệ → fallback
        final_move = best_move if best_move != Move.STAY else fallback_move

        # Nếu vẫn STAY → đi tiếp hướng cũ để tránh đứng yên
        if final_move == Move.STAY:
            final_move = self.last_move

        self.last_move = final_move
        self.last_pacman_pos = pac_pos
        return final_move

    # ------------------------------------------------------------
    def _predict_pacman(self, pac_pos):
        if not self.last_pacman_pos:
            return pac_pos
        dr = pac_pos[0] - self.last_pacman_pos[0]
        dc = pac_pos[1] - self.last_pacman_pos[1]
        return (pac_pos[0] + dr, pac_pos[1] + dc)

    def _repulsive_potential(self, pos, pac_pos):
        dist = self._euclid_dist(pos, pac_pos)
        if dist <= 0:
            return float("inf")
        return self.K / (dist ** 1.4)

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _valid(self, pos, map_state):
        h, w = map_state.shape
        r, c = pos
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    def _adjacent_wall_count(self, pos, map_state):
        count = 0
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        h, w = map_state.shape
        for dr, dc in dirs:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < h and 0 <= nc < w) or map_state[nr, nc] != 0:
                count += 1
        return count

    def _flood_fill_space(self, map_state, start):
        """Đếm số ô trống xung quanh để ưu tiên vùng mở."""
        h, w = map_state.shape
        q = [start]
        visited = {start}
        count = 0
        limit = 8
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q and count < limit:
            r, c = q.pop(0)
            count += 1
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] == 0 and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return count

    def _euclid_dist(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)