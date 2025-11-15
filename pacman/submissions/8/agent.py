"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
"""

import sys
import time
import numpy as np
import random
import math
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

from collections import deque
from heapq import heappush, heappop


class PacmanAgent(BasePacmanAgent):
    """Pacman Agent với khả năng chặn đường và dự đoán chính xác."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Optimized Time-Limited Pacman"

        # --- THUỘC TÍNH KIỂM SOÁT THỜI GIAN ---
        self.ASTAR_TIME_LIMIT = 0.90  # Giới hạn 0.90s cho A* (để lại 0.1s cho các logic khác)
        self.INFINITY = float('inf')

        # --- Thuộc tính caching
        self.current_path = []  # Path A* đã tính
        self.last_ghost_pos = None  # Vị trí Ghost ở bước tính toán path gần nhất
        self.stuck_counter = 0
        self.last_distance = None
        self.ghost_history = deque(maxlen=5)
        self.my_history = deque(maxlen=5)

    #  Hàm trợ giúp
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        return neighbors

    def _detect_stuck(self, my_pos, ghost_pos):
        """Phát hiện tình trạng stuck (khoảng cách không đổi)."""
        current_distance = self._manhattan_distance(my_pos, ghost_pos)

        if self.last_distance is not None:
            if current_distance == self.last_distance:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

        self.last_distance = current_distance

        # Sửa ngưỡng stuck từ 3 xuống 2 để phản ứng nhanh hơn
        return self.stuck_counter >= 2

    def _get_chase_move(self, my_pos, ghost_pos, map_state):
        """Fallback: Di chuyển về phía Ghost (chase trực tiếp) bằng A* đơn giản."""
        path = self.astar(my_pos, ghost_pos, map_state)
        if path and path != [Move.STAY]:
            return path[0]
        return Move.STAY

    def _get_ghost_move_direction(self):
        """Lấy hướng di chuyển gần nhất của Ghost."""
        if len(self.ghost_history) < 2:
            return None

        prev = self.ghost_history[-2]
        curr = self.ghost_history[-1]

        dr = curr[0] - prev[0]
        dc = curr[1] - prev[1]

        return (dr, dc)

    def _predict_next_ghost_positions(self, ghost_pos, map_state, my_pos):
        """
        Dự đoán TẤT CẢ các vị trí Ghost CÓ THỂ đi trong bước tiếp theo.
        Ưu tiên các vị trí xa Pacman (Ghost đang chạy trốn).
        """
        possible_positions = []

        for next_pos, move in self._get_neighbors(ghost_pos, map_state):
            # Tính khoảng cách từ vị trí đó đến Pacman
            distance_to_pacman = self._manhattan_distance(next_pos, my_pos)

            # Tính số lối thoát từ vị trí đó
            escape_routes = len(self._get_neighbors(next_pos, map_state))

            # Điểm ưu tiên: Ghost muốn xa Pacman và có nhiều lối thoát
            priority = distance_to_pacman * 2 + escape_routes

            possible_positions.append((priority, next_pos, move))

        # Sắp xếp theo độ ưu tiên (cao nhất trước)
        possible_positions.sort(reverse=True, key=lambda x: x[0])

        return possible_positions

    def _find_cut_off_position(self, my_pos, ghost_pos, map_state):
        """
        Tìm vị trí để chặn đường Ghost.
        Chiến thuật: Đi tới vị trí mà Ghost SẼ ĐẾN trong bước tiếp theo.
        """
        # Dự đoán các vị trí Ghost có thể đến
        predicted_positions = self._predict_next_ghost_positions(ghost_pos, map_state, my_pos)

        if not predicted_positions:
            return ghost_pos

        # Lấy top 3 vị trí Ghost có thể đến
        for priority, predicted_ghost_pos, ghost_move in predicted_positions[:3]:
            # Kiểm tra xem Pacman có thể đến vị trí đó trong 1 bước không
            for my_next_pos, my_move in self._get_neighbors(my_pos, map_state):
                if my_next_pos == predicted_ghost_pos:
                    # !!!! Pacman có thể chặn đúng vị trí Ghost sẽ đến
                    return predicted_ghost_pos

            # Nếu không thể chặn trong 1 bước -> thử tìm vị trí chặn gần đó
            # Tìm các vị trí cạnh predicted_ghost_pos
            for adjacent_pos, _ in self._get_neighbors(predicted_ghost_pos, map_state):
                pacman_dist = self._manhattan_distance(my_pos, adjacent_pos)
                if pacman_dist == 1:  # Pacman có thể đến trong 1 bước
                    return adjacent_pos

        # Fallback: Dự đoán vị trí Ghost sẽ đến (vị trí xa nhất)
        return predicted_positions[0][1]

    def _get_blocking_move(self, my_pos, ghost_pos, map_state):
        """
        Tìm nước đi tốt nhất để chặn Ghost.
        Ưu tiên: Chặn vị trí Ghost SẼ ĐẾN, không phải vị trí hiện tại.
        """
        # Dự đoán vị trí Ghost tiếp theo
        predicted_positions = self._predict_next_ghost_positions(ghost_pos, map_state, my_pos)

        if not predicted_positions:
            return self._get_chase_move(my_pos, ghost_pos, map_state)

        # Thử từng vị trí mà Ghost có thể đến
        for _, predicted_ghost_pos, _ in predicted_positions:
            # Thử từng nước đi của Pacman
            for my_next_pos, my_move in self._get_neighbors(my_pos, map_state):
                # Kiểm tra xem nước đi này có chặn được Ghost không
                if my_next_pos == predicted_ghost_pos:
                    return my_move

                # Hoặc ít nhất giảm khoảng cách đến vị trí dự đoán
                current_dist = self._manhattan_distance(my_pos, predicted_ghost_pos)
                next_dist = self._manhattan_distance(my_next_pos, predicted_ghost_pos)

                if next_dist < current_dist:
                    return my_move

        # Fallback: đuổi theo Ghost hiện tại
        return self._get_chase_move(my_pos, ghost_pos, map_state)

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:

        # Cập nhật lịch sử và kiểm tra bắt ngay (Giữ nguyên)
        self.ghost_history.append(enemy_position)
        self.my_history.append(my_position)
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(my_position, move)
            if next_pos == enemy_position:
                return move

        distance = self._manhattan_distance(my_position, enemy_position)
        is_stuck = self._detect_stuck(my_position, enemy_position)

        # --- BẮT ĐẦU KIỂM TRA THỜI GIAN TRONG HÀM STEP ---
        start_time = time.time()

        recalculate_path = False
        if not self.current_path or self.last_ghost_pos is None or self._manhattan_distance(enemy_position,
                                                                                            self.last_ghost_pos) > 1 or is_stuck:
            recalculate_path = True

        path = None
        target_pos = enemy_position

        if recalculate_path:
            try:
                # 1. Lựa chọn Mục tiêu (Giữ nguyên)
                if distance <= 3 and not is_stuck:
                    return self._get_blocking_move(my_position, enemy_position, map_state)

                if distance <= 6 or is_stuck:
                    target_pos = self._find_cut_off_position(my_position, enemy_position, map_state)
                else:
                    target_pos = self._find_cut_off_position(my_position, enemy_position, map_state)

                # 2. CHẠY A* TỚI MỤC TIÊU (Truyền thời gian bắt đầu)
                path = self.astar(my_position, target_pos, map_state, start_time)

                # 3. FALLBACK TỐI ƯU (Nếu không tìm thấy đường)
                if (not path or path == [Move.STAY]) and distance > 1:
                    path = self.astar(my_position, enemy_position, map_state, start_time)

                # CẬP NHẬT CACHE
                if path and path != [Move.STAY]:
                    self.current_path = path
                else:
                    self.current_path = []

                self.last_ghost_pos = enemy_position

            except TimeoutError:
                # LỖI XẢY RA: Nếu A* quá 0.9s, sử dụng Blocking Move hoặc Chase Move
                self.current_path = []  # Xóa cache path cũ
                return self._get_blocking_move(my_position, enemy_position, map_state)  # Sử dụng hàm nhanh nhất

        # --- FOLLOW PATH LOGIC ---
        if self.current_path and self.current_path != [Move.STAY]:
            next_move = self.current_path.pop(0)
            return next_move

        # FALLBACK CUỐI CÙNG (nếu mọi thứ thất bại)
        return self._get_chase_move(my_position, enemy_position, map_state)

    def astar(self, start: tuple, goal: tuple, map_state: np.ndarray, start_time: float) -> list:
        """A* Search có kiểm soát thời gian. Sẽ dừng và báo lỗi nếu vượt quá giới hạn."""

        if start == goal:
            return [Move.STAY]

        h_start = self._manhattan_distance(start, goal)
        # Priority queue: (f_cost, g_cost, position, path)
        frontier = [(h_start, 0, start, [])]
        g_costs = {start: 0}

        max_iterations = 4000
        iterations = 0

        while frontier:
            # 1. KIỂM TRA THỜI GIAN TRỰC TIẾP
            if (time.time() - start_time) > self.ASTAR_TIME_LIMIT:
                raise TimeoutError("A* search exceeded time limit")

            # 2. KIỂM TRA GIỚI HẠN VÒNG LẶP (Phòng vệ thứ cấp)
            iterations += 1
            if iterations > max_iterations:
                break

            f_cost, g_cost, current_pos, path = heappop(frontier)

            if current_pos == goal:
                return path

            # Tối ưu: Nếu tìm thấy đường đi tốn kém hơn đường đi đã biết, BỎ QUA
            if g_cost > g_costs.get(current_pos, float('inf')):
                continue

            for next_pos, move in self._get_neighbors(current_pos, map_state):
                new_g_cost = g_cost + 1

                if new_g_cost < g_costs.get(next_pos, float('inf')):
                    g_costs[next_pos] = new_g_cost
                    h_cost = self._manhattan_distance(next_pos, goal)
                    new_f_cost = new_g_cost + h_cost
                    new_path = path + [move]
                    heappush(frontier, (new_f_cost, new_g_cost, next_pos, new_path))

        return [Move.STAY]

class GhostAgent(BaseGhostAgent):
    """
    Ghost agent sử dụng Minimax với Tăng dần độ sâu (Iterative Deepening)
    để đảm bảo chạy trong 1 giây.
    """

    def __init__(self, **kwargs):
        """
        Khởi tạo Ghost agent.
        """
        super().__init__(**kwargs)
        self.name = "Minimax Ghost (Time-Limited)"

        # Giới hạn thời gian (ít hơn 1 giây một chút để an toàn)
        self.TIME_LIMIT = 0.95

        # Sử dụng math.inf cho giá trị vô cực
        self.INFINITY = math.inf

    # =================================================================
    # --- HÀM STEP CHÍNH (LOGIC TĂNG DẦN ĐỘ SÂU) ---
    # =================================================================

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Hàm này được gọi mỗi lượt.
        Nó sẽ chạy Minimax lặp đi lặp lại với độ sâu tăng dần
        cho đến khi gần hết 1 giây.
        """

        # 1. Ghi lại thời điểm bắt đầu
        start_time = time.time()

        current_depth = 1
        best_move_so_far = None  # Nước đi tốt nhất tìm được

        # --- 2. VÒNG LẶP TĂNG DẦN ĐỘ SÂU ---
        while True:
            # 3. KIỂM TRA THỜI GIAN
            time_elapsed = time.time() - start_time
            if time_elapsed > self.TIME_LIMIT:
                # print(f"[DEBUG] Hết giờ! Dừng ở độ sâu {current_depth - 1}")
                break  # Hết giờ, thoát khỏi vòng lặp

            # print(f"[DEBUG] Đang tìm kiếm ở độ sâu {current_depth}...")

            try:
                # 4. CHẠY MINIMAX VỚI ĐỘ SÂU HIỆN TẠI
                score, move = self._minimax_recursive(
                    hider_pos=my_position,
                    seeker_pos=enemy_position,
                    depth=current_depth,  # Sử dụng độ sâu hiện tại
                    is_maximizing_player=True,
                    alpha=-self.INFINITY,
                    beta=self.INFINITY,
                    map_state=map_state,
                    start_time=start_time  # <<< Quan trọng: Truyền thời gian bắt đầu
                )

                # 5. LƯU KẾT QUẢ
                if move is not None:
                    best_move_so_far = move
                else:
                    # Nếu hàm minimax trả về None (bị kẹt), dừng lại
                    break

            except TimeoutError:
                # 7. BẮT LỖI "QUÁ GIỜ" TỪ HÀM ĐỆ QUY
                # print(f"[DEBUG] Ngắt khẩn cấp ở độ sâu {current_depth}")
                break  # Thoát vòng lặp

            # 6. TĂNG ĐỘ SÂU CHO VÒNG LẶP TỚI
            current_depth += 1

        # --- Hết giờ hoặc tìm kiếm xong ---

        if best_move_so_far is None:
            # Nếu không tìm được gì cả (ngay cả ở depth=1), dùng fallback
            return self._get_fallback_move(my_position, map_state)

        # Trả về nước đi tốt nhất đã tìm được trong 0.95 giây
        return best_move_so_far

    # =================================================================
    # --- CÁC PHƯƠNG THỨC MINIMAX (ĐÃ SỬA ĐỔI) ---
    # =================================================================

    def _minimax_recursive(self, hider_pos, seeker_pos, depth, is_maximizing_player,
                           alpha, beta, map_state, start_time):  # <<< THÊM start_time
        """
        Hàm Minimax đệ quy. Giờ đây có thêm kiểm tra thời gian.
        """

        # --- 8. NÚT DỪNG KHẨN CẤP (QUAN TRỌNG NHẤT) ---
        # Kiểm tra ngay khi bắt đầu mỗi lần gọi đệ quy
        if (time.time() - start_time) > self.TIME_LIMIT:
            raise TimeoutError("Minimax search timed out")

        # --- Điều kiện dừng (Base Cases) ---
        if self._is_game_over(hider_pos, seeker_pos):
            return -self.INFINITY, None

        if depth == 0:
            # Hết độ sâu, trả về điểm heuristic
            # (Đã sửa lỗi thiếu 'map_state' từ lần trước)
            return self._calculate_heuristic(hider_pos, seeker_pos, map_state), None

        # --- Lượt của Hider (MAX) ---
        if is_maximizing_player:
            max_eval = -self.INFINITY
            best_move = None

            for move in self._get_valid_moves(hider_pos, map_state):
                new_hider_pos = (hider_pos[0] + move.value[0], hider_pos[1] + move.value[1])

                # Gọi đệ quy, truyền start_time xuống
                eval, _ = self._minimax_recursive(
                    new_hider_pos, seeker_pos, depth - 1, False,
                    alpha, beta, map_state, start_time
                )

                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return max_eval, best_move

        # --- Lượt của Seeker (MIN) ---
        else:  # (is_minimizing_player)
            min_eval = +self.INFINITY
            best_move_for_min = None

            for move in self._get_valid_moves(seeker_pos, map_state):
                new_seeker_pos = (seeker_pos[0] + move.value[0], seeker_pos[1] + move.value[1])

                # Gọi đệ quy, truyền start_time xuống
                eval, _ = self._minimax_recursive(
                    hider_pos, new_seeker_pos, depth - 1, True,
                    alpha, beta, map_state, start_time
                )

                if eval < min_eval:
                    min_eval = eval
                    best_move_for_min = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return min_eval, best_move_for_min

    # =================================================================
    # --- CÁC PHƯƠNG THỨC TRỢ GIÚP (GIỮ NGUYÊN) ---
    # =================================================================

    def _calculate_heuristic(self, hider_pos, seeker_pos, map_state) -> float:
        """
        Tính điểm heuristic thông minh, với hình phạt NẶNG cho ngõ cụt.
        (Đây là hàm bạn tự điều chỉnh, giữ nguyên)
        """

        # --- Trọng số (Bạn có thể điều chỉnh) ---
        W_DISTANCE = 10  # Vẫn là trọng số cho khoảng cách

        # --- BẢNG TRA CỨU HÌNH PHẠT/THƯỞNG CHO ĐƯỜNG THOÁT ---
        ESCAPE_SCORES = {
            4: 50,  # Ngã tư: Thưởng lớn (Bảng điểm của bạn)
            3: 30,  # Cạnh tường: Thưởng nhỏ
            2: -15,  # Trong góc/hành lang: Bắt đầu phạt
            1: -70,  # Ngõ cụt: Phạt RẤT NẶNG
            0: -200  # Bị kẹt hoàn toàn: Phạt CỰC LỚN
        }

        # --- 1. Thành phần Khoảng cách ---
        distance_to_seeker = abs(hider_pos[0] - seeker_pos[0]) + abs(hider_pos[1] - seeker_pos[1])

        if distance_to_seeker == 0:
            return -self.INFINITY

        # --- 2. Thành phần Đường thoát ---
        escape_routes = self._count_escape_routes(hider_pos, map_state)

        # --- 3. Tính điểm cuối cùng ---
        score = (W_DISTANCE * distance_to_seeker)
        score += ESCAPE_SCORES.get(escape_routes, ESCAPE_SCORES[0])  # Dùng bảng của bạn

        return score

    def _count_escape_routes(self, position: tuple, map_state: np.ndarray) -> int:
        """
        Đếm số ô hợp lệ (không phải tường) xung quanh một vị trí.
        Không bao gồm 'STAY'.
        """
        row, col = position
        possible_moves = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        count = 0
        for r, c in possible_moves:
            height, width = map_state.shape
            if 0 <= r < height and 0 <= c < width and map_state[r, c] == 0:
                count += 1
        return count

    def _is_game_over(self, hider_pos, seeker_pos) -> bool:
        """Kiểm tra xem Seeker có bắt được Hider không."""
        return hider_pos == seeker_pos

    def _get_valid_moves(self, position: tuple, map_state: np.ndarray) -> list:
        """
        Trả về một danh sách các đối tượng 'Move' hợp lệ từ một vị trí.
        Bao gồm cả 'STAY'.
        """
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        valid_moves = []

        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (position[0] + delta_row, position[1] + delta_col)

            if self._is_valid_position(new_pos, map_state):
                valid_moves.append(move)

        return valid_moves

    def _get_fallback_move(self, my_position: tuple, map_state: np.ndarray) -> Move:
        """
        Sử dụng nếu Minimax không trả về gì.
        """
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)

        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)

            if self._is_valid_position(new_pos, map_state):
                return move

        return Move.STAY

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        # Giả sử 0 là đường đi, 1 là tường
        return map_state[row, col] == 0

