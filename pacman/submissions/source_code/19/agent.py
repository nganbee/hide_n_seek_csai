import heapq
import math
import sys
import time
from collections import deque
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move

"""
Hợp lệ theo yêu cầu: chỉ dùng thư viện chuẩn + time.
"""


class PacmanAgent(BasePacmanAgent):
    """
    Tác nhân Pacman sử dụng thuật toán A* (A Star) để đuổi bắt Ghost.
    Có xử lý chống xuyên qua nhau (swap collision).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A* Pacman Hunter"

    # =====================================================
    # HÀM CHÍNH: ĐƯỢC GỌI MỖI LƯỢT DI CHUYỂN
    # =====================================================
    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Hàm quyết định hướng di chuyển mỗi bước.
        - Sử dụng thuật toán A* để tìm đường tối ưu đuổi theo Ghost.
        - Có giới hạn thời gian tối đa 1 giây cho mỗi bước.
        """

        start_time = (
            time.time()
        )  # Ghi nhận thời điểm bắt đầu để giới hạn thời gian

        # Tìm đường đi ngắn nhất bằng A*
        path = self.a_star_search(
            map_state, my_position, enemy_position, start_time
        )

        # Nếu tìm được đường đi hợp lệ (ít nhất 2 ô: hiện tại + bước kế tiếp)
        if path and len(path) > 1:
            next_pos = path[1]  # Lấy bước kế tiếp để di chuyển

            # =====================================================
            #  NGĂN XUYÊN QUA NHAU (SWAP COLLISION)
            # =====================================================
            # Dự đoán các ô mà Ghost có thể di chuyển đến trong lượt tiếp theo
            enemy_next_positions = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                new_enemy_pos = (enemy_position[0] + dr, enemy_position[1] + dc)
                if self._is_valid_position(new_enemy_pos, map_state):
                    enemy_next_positions.append(new_enemy_pos)

            # Nếu Pacman định đi vào đúng ô Ghost đang đứng
            # Hoặc Pacman và Ghost cùng đổi vị trí trong cùng một lượt
            if next_pos == enemy_position or (
                my_position in enemy_next_positions
                and enemy_position == next_pos
            ):
                # Khi đó Pacman sẽ đứng yên để tránh “xuyên qua nhau”
                return Move.STAY

            # Di chuyển theo hướng từ vị trí hiện tại đến vị trí kế tiếp
            return self._position_to_move(my_position, next_pos)

        # Nếu không tìm thấy đường hợp lệ trong 1s → dùng chiến lược đuổi "tham lam"
        return self._greedy_chase(map_state, my_position, enemy_position)

    # =====================================================
    # THUẬT TOÁN A* (A STAR SEARCH)
    # =====================================================
    def a_star_search(self, map_state, start, goal, start_time):
        """
        Thuật toán A*: tìm đường ngắn nhất từ start → goal.
        Có giới hạn thời gian thực thi là 1 giây.
        """

        # Hàm heuristic: khoảng cách Manhattan
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Hàng đợi ưu tiên (priority queue)
        frontier = []
        heapq.heappush(frontier, (0, 0, start, [start]))  # (f, tie, node, path)
        explored = {start: 0}
        tie = 0  # Dùng để phá hòa khi f bằng nhau

        while frontier:
            # Giới hạn thời gian 1 giây để tránh vượt quá yêu cầu bài
            if time.time() - start_time > 1.0:
                break

            f, _, current, path = heapq.heappop(frontier)
            if current == goal:
                return path  # Đã đến đích

            g = len(path) - 1  # Chi phí thực tế từ start → current
            if explored.get(current, float("inf")) < g:
                continue

            # Duyệt 4 hướng có thể đi: lên, xuống, trái, phải
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                neighbor = (current[0] + dr, current[1] + dc)

                # Bỏ qua nếu vị trí không hợp lệ (ngoài bản đồ hoặc là tường)
                if not self._is_valid_position(neighbor, map_state):
                    continue

                new_g = g + 1
                # Nếu đã thăm với chi phí tốt hơn → bỏ qua
                if neighbor in explored and explored[neighbor] <= new_g:
                    continue

                explored[neighbor] = new_g
                h = manhattan_distance(neighbor, goal)
                tie += 1
                # Thêm vào hàng đợi với tổng chi phí f = g + h
                heapq.heappush(
                    frontier, (new_g + h, tie, neighbor, path + [neighbor])
                )

        # Nếu không tìm thấy đường hợp lệ → trả về rỗng
        return []

    # =====================================================
    # CÁC HÀM HỖ TRỢ
    # =====================================================
    def _greedy_chase(self, map_state, my_pos, enemy_pos):
        """
        Chiến lược đuổi đơn giản khi A* thất bại.
        Pacman sẽ đi theo hướng gần Ghost nhất (theo Manhattan distance).
        """
        row_diff = enemy_pos[0] - my_pos[0]
        col_diff = enemy_pos[1] - my_pos[1]
        moves = []

        # Ưu tiên hướng có khoảng cách lớn hơn
        if abs(row_diff) >= abs(col_diff):
            if row_diff > 0:
                moves.append(Move.DOWN)
            elif row_diff < 0:
                moves.append(Move.UP)
            if col_diff > 0:
                moves.append(Move.RIGHT)
            elif col_diff < 0:
                moves.append(Move.LEFT)
        else:
            if col_diff > 0:
                moves.append(Move.RIGHT)
            elif col_diff < 0:
                moves.append(Move.LEFT)
            if row_diff > 0:
                moves.append(Move.DOWN)
            elif row_diff < 0:
                moves.append(Move.UP)

        # Kiểm tra và chọn hướng di chuyển hợp lệ đầu tiên
        for move in moves + [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            new_pos = (my_pos[0] + dr, my_pos[1] + dc)
            if self._is_valid_position(new_pos, map_state):
                return move

        return Move.STAY  # Không có hướng hợp lệ thì đứng yên

    def _position_to_move(self, current, target):
        """Chuyển từ tọa độ đích sang hướng di chuyển (Move)."""
        dr = target[0] - current[0]
        dc = target[1] - current[1]
        if dr > 0:
            return Move.DOWN
        elif dr < 0:
            return Move.UP
        elif dc > 0:
            return Move.RIGHT
        elif dc < 0:
            return Move.LEFT
        return Move.STAY

    def _is_valid_position(self, pos, map_state):
        """Kiểm tra xem vị trí có hợp lệ không (trong bản đồ và không phải tường)."""
        r, c = pos
        if r < 0 or r >= len(map_state) or c < 0 or c >= len(map_state[0]):
            return False
        return map_state[r][c] == 0


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

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ) -> Move:
        """
        Kết hợp Minimax (gần) và Predictive BFS (xa)

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Pacman's current (row, col)
            step_number: Current step number (starts at 1)

        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # Tính khoảng cách đến Pacman
        distance_to_pacman = self._calculate_manhattan_distance(
            my_position, enemy_position
        )

        # Vùng nguy hiểm : Nếu gần Pacman (<=6) → dùng Minimax để chọn nước đi tốt nhất
        if distance_to_pacman <= 6:
            depth = 3  # Gần - depth trung bình
            if distance_to_pacman <= 3:
                depth = 4  # Rất gần - depth cao

            # Gọi minimax với depth=3
            _, best_move = self._minimax(
                my_position,
                enemy_position,
                depth=depth,  # Nhìn trước 3 bước
                is_ghost_turn=True,
                map_state=map_state,
            )

            # Nếu minimax trả về move hợp lệ -> dùng
            if best_move != Move.STAY:
                return best_move

            # Fallback nếu minimax thất bại
            return self._greedy_escape(my_position, enemy_position, map_state)

        # Vùng an toàn : Pacman XA (> 6 cells) → Dùng PREDICTIVE BFS
        else:
            # Bước 1 - Dự đoán Pacman sẽ ở đâu
            predicted_enemy_pos = self._predict_enemy_position(
                enemy_position, my_position, map_state
            )

            # Bước 2 - Tìm vị trí xa predicted position
            safe_positions = self._find_safe_positions(
                map_state, my_position, predicted_enemy_pos, top_k=5
            )

            if not safe_positions:
                return self._greedy_escape(
                    my_position, enemy_position, map_state
                )

            # Bước 3 - Chọn vị trí có nhiều lối thoát nhất
            best_position = max(
                safe_positions,
                key=lambda pos: self._count_escape_routes(pos, map_state),
            )

            # Bước 4 - Di chuyển đến vị trí đó
            next_move = self._get_next_move(
                my_position, best_position, map_state
            )

            return next_move if next_move else Move.STAY

        # # Bước 1: Tìm top 5 vị trí xa Pacman nhất bằng BFS
        # safe_positions = self._find_safe_positions(
        #     map_state, my_position, enemy_position, top_k=5
        # )

        # # Nếu không tìm được vị trí nào → dùng fallback (chạy ngược hướng)
        # if not safe_positions:
        #     return self._greedy_escape(my_position, enemy_position, map_state)

        # # Bước 2: Trong 5 vị trí xa nhất, chọn vị trí có NHIỀU LỐI THOÁT NHẤT
        # # max() với key=lambda sẽ chọn vị trí có _count_escape_routes() cao nhất
        # best_position = max(
        #     safe_positions,
        #     key=lambda pos: self._count_escape_routes(pos, map_state)
        # )

        # # Bước 3: Tìm bước đi ĐẦU TIÊN để đến vị trí best_position
        # next_move = self._get_next_move(my_position, best_position, map_state)

        # # Trả về bước đi (hoặc STAY nếu không tìm được)
        # return next_move if next_move else Move.STAY

    # Helper methods (you can add more)

    def _is_valid_move(
        self, pos: tuple, move: Move, map_state: np.ndarray
    ) -> bool:
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

    def _calculate_manhattan_distance(self, pos1, pos2):
        # Tính khoảng cách Manhattan giữa hai vị trí
        # Manhattan distance = |row1 - row2| + |col1 - col2|
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _find_safe_positions(self, map_state, my_pos, enemy_pos, top_k=5):
        """
        Tìm top_k vị trí xa Pacman nhất mà Ghost có thể đến được

        # Sử dụng BFS để duyệt toàn bộ map và tìm các vị trí xa nhất

        # Returns về list các vị trí [(row, col), ...] xa Pacman nhất
        """

        # Bước 1: Chuẩn bị BFS

        visited = set()  # Lưu các ô đã thăm
        queue = deque([(my_pos, 0)])  # Queue: (vị trí, khoảng cách từ ghost)
        visited.add(my_pos)
        positions_with_distance = []  # Lưu: (vị trí, khoảng cách đến Pacman)

        # Bước 2: BFS duyệt toàn bộ map
        while queue:
            current_pos, dist_from_start = queue.popleft()

            # Tính khoảng cách Manhattan từ current_pos đến Pacman
            dist_to_enemy = self._calculate_manhattan_distance(
                current_pos, enemy_pos
            )

            # Lưu vị trí này cùng khoảng cách đến Pacman
            positions_with_distance.append((current_pos, dist_to_enemy))

            # Duyệt 4 hướng (UP, DOWN, LEFT, RIGHT)
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value  # Lấy delta_row, delta_col
                new_pos = (current_pos[0] + dr, current_pos[1] + dc)

                # Kiểm tra new_pos có hợp lệ không?
                #           Chưa được thăm (not in visited)
                #           Hợp lệ (_is_valid_position)
                if new_pos not in visited and self._is_valid_position(
                    new_pos, map_state
                ):
                    visited.add(new_pos)
                    queue.append((new_pos, dist_from_start + 1))

        # Bước 3: Sắp xếp theo khoảng cách đến Pacman (giảm dần)
        # Vị trí xa nhất sẽ ở đầu list
        positions_with_distance.sort(key=lambda x: x[1], reverse=True)

        # Bước 4: Trả về top_k vị trí (chỉ lấy vị trí, bỏ khoảng cách)
        return [pos for pos, dist in positions_with_distance[:top_k]]

    def _count_escape_routes(self, pos, map_state):
        """
        Đếm số lối thoát từ một vị trí

        Returns:
            int: Số lượng ô trống xung quanh (0-4)
        """
        count = 0

        # Duyệt 4 hướng và đếm số ô hợp lệ
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            neighbor_pos = (pos[0] + dr, pos[1] + dc)

            if self._is_valid_position(neighbor_pos, map_state):
                count += 1

        return count

    def _get_next_move(self, start, goal, map_state):
        """
        Tìm bước đi đầu tiên để đi từ start đến goal.

        Sử dụng BFS để tìm đường ngắn nhất, rồi trả về bước đầu tiên.

        Returns:
            Move: Bước đi đầu tiên (UP/DOWN/LEFT/RIGHT/STAY)
        """
        from collections import deque

        if start == goal:
            return Move.STAY

        # BFS với lưu path (đường đi)
        visited = {start}
        queue = deque([(start, [])])  # (vị trí, [list các move đã đi])

        while queue:
            pos, path = queue.popleft()

            # Thử 4 hướng
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                new_pos = (pos[0] + dr, pos[1] + dc)

                # Nếu đến đích → trả về bước ĐẦU TIÊN trong path
                if new_pos == goal:
                    if path:  # Nếu đã có path trước đó
                        return path[0]
                    else:  # Nếu goal ngay cạnh start
                        return move

                # Nếu chưa thăm và hợp lệ → thêm vào queue
                if new_pos not in visited and self._is_valid_position(
                    new_pos, map_state
                ):
                    visited.add(new_pos)
                    new_path = path + [move]  # Thêm move vào path
                    queue.append((new_pos, new_path))

        # Không tìm được đường → đứng yên
        return Move.STAY

    def _greedy_escape(self, my_pos, enemy_pos, map_state):
        """
        Fallback: Chạy ngược hướng Pacman đơn giản (dùng khi không tìm được vị trí tốt).

        Returns:
            Move: Bước đi tránh Pacman
        """
        row_diff = my_pos[0] - enemy_pos[0]
        col_diff = my_pos[1] - enemy_pos[1]

        # Chạy ngược hướng Pacman
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT

        # Kiểm tra move có hợp lệ không
        if self._is_valid_move(my_pos, move, map_state):
            return move

        # Nếu không hợp lệ, thử các hướng khác
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_pos, move, map_state):
                return move

        return Move.STAY

    def _predict_enemy_position(self, enemy_pos, my_pos, map_state):
        """
        Dự đoán Pacman sẽ di chuyển đến đâu

        Giả định: Pacman sẽ chọn bước đi gần Ghost nhất (greedy)

        Returns:
            tuple: Vị trí dự đoán của Pacman ở bước tiếp theo
        """
        best_move = Move.STAY
        best_distance = float("inf")  # Pacman muốn minimize khoảng cách

        # Duyệt 4 hướng Pacman có thể đi
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            new_pos = (enemy_pos[0] + dr, enemy_pos[1] + dc)

            # Kiểm tra valid
            if not self._is_valid_position(new_pos, map_state):
                continue

            # Tính khoảng cách từ new_pos đến Ghost (my_pos)
            distance = self._calculate_manhattan_distance(new_pos, my_pos)

            # Pacman chọn move làm distance nhỏ nhất
            if distance < best_distance:
                best_distance = distance
                best_move = move

        # Apply move để có predicted position
        if best_move != Move.STAY:
            dr, dc = best_move.value
            return (enemy_pos[0] + dr, enemy_pos[1] + dc)

        return enemy_pos

    def _minimax(self, my_pos, enemy_pos, depth, is_ghost_turn, map_state):
        """
        Minimax algorithm cho Ghost

        Ghost = Maximizing player (muốn xa Pacman)
        Pacman = Minimizing player (muốn gần Ghost)

        Args:
            depth: Số bước nhìn trước (3-4 là tốt)
            is_ghost_turn: True = lượt Ghost, False = lượt Pacman

        Returns:
            (score, best_move): Score là khoảng cách, move là nước đi tốt nhất
        """

        # Hết depth hoặc bị bắt
        if depth == 0 or my_pos == enemy_pos:
            # Return khoảng cách Manhattan
            distance = self._calculate_manhattan_distance(my_pos, enemy_pos)
            return distance, Move.STAY

        if is_ghost_turn:  # Lượt Ghost - MAXIMIZE distance
            best_score = -float("inf")
            best_move = Move.STAY

            # Thử 4 hướng Ghost có thể đi
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                new_pos = (my_pos[0] + dr, my_pos[1] + dc)

                # Kiểm tra valid
                if not self._is_valid_position(new_pos, map_state):
                    continue

                # Gọi đệ quy - lượt Pacman (is_ghost_turn=False)
                score, _ = self._minimax(
                    new_pos, enemy_pos, depth - 1, False, map_state
                )

                # Cập nhật best nếu score CAO hơn (maximize)
                if score > best_score:
                    best_score = score
                    best_move = move

            return best_score, best_move

        else:  # Lượt Pacman - MINIMIZE distance
            best_score = float("inf")
            best_move = Move.STAY

            # Thử 4 hướng Pacman có thể đi
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                new_pos = (enemy_pos[0] + dr, enemy_pos[1] + dc)

                # Kiểm tra valid
                if not self._is_valid_position(new_pos, map_state):
                    continue

                # Gọi đệ quy - lượt Ghost (is_ghost_turn=True)
                score, _ = self._minimax(
                    my_pos, new_pos, depth - 1, True, map_state
                )

                # Cập nhật best nếu score THẤP hơn (minimize)
                if score < best_score:
                    best_score = score
                    best_move = move

            return best_score, best_move
