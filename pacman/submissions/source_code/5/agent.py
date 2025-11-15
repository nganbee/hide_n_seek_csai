import sys
from pathlib import Path
from itertools import count
import random
# Add src to path so we import the same modules the framework uses
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
from collections import deque
from typing import Tuple
import numpy as np
import random
import heapq
import itertools
counter = itertools.count()


# ================================
# SEEKER (Pacman) - A* khi xa, BFS khi gần (chuẩn (row, col))
# ================================
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_ghost = None
        self._tie = count()

    # ---------- helpers (chuẩn row,col) ----------
    def _get_neighbors(self, game_map: np.ndarray, pos):
        r, c = pos
        H, W = game_map.shape
        moves = (
            (Move.UP,    (r-1, c)),
            (Move.DOWN,  (r+1, c)),
            (Move.LEFT,  (r, c-1)),
            (Move.RIGHT, (r, c+1)),
        )
        res = []
        for mv, (nr, nc) in moves:
            if 0 <= nr < H and 0 <= nc < W and game_map[nr][nc] == 0:
                res.append((mv, (nr, nc)))
        return res

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _bfs_path(self, map_state: np.ndarray, start, goal):
        from collections import deque
        if start == goal:
            return []
        q = deque([(start, [])])
        seen = {start}
        while q:
            (r, c), path = q.popleft()
            for mv, (nr, nc) in self._get_neighbors(map_state, (r, c)):
                if (nr, nc) in seen:
                    continue
                if (nr, nc) == goal:
                    return path + [mv]
                seen.add((nr, nc))
                q.append(((nr, nc), path + [mv]))
        return []

    def _astar_path(self, map_state: np.ndarray, start, goal):
        import heapq
        def h(p):
            return self._manhattan(p, goal)

        openh = []
        # tuple: (f, g, tie, cur, path)
        heapq.heappush(openh, (h(start), 0, next(self._tie), start, []))
        seen = set()

        while openh:
            f, g, _, cur, path = heapq.heappop(openh)
            if cur in seen:
                continue
            if cur == goal:
                return path
            seen.add(cur)

            for mv, nxt in self._get_neighbors(map_state, cur):
                if nxt in seen:
                    continue
                ng = g + 1
                nf = ng + h(nxt)
                heapq.heappush(
                    openh,
                    (nf, ng, next(self._tie), nxt, path + [mv])
                )
        return []

    def _apply_moves(self, pos, moves):
        r, c = pos
        res = []
        for mv in moves:
            dr, dc = mv.value
            r, c = r + dr, c + dc
            res.append((r, c))
        return res
    
    def _is_deadend(self, map_state: np.ndarray, pos) -> bool:
        # ngõ cụt: số láng giềng hợp lệ <= 1
        return len(self._get_neighbors(map_state, pos)) <= 1

    def _nearest_exit_from_deadend(self, map_state: np.ndarray, start):
        """Trả về path (list Move) từ ngõ cụt ra chỗ không còn là ngõ cụt."""
        from collections import deque
        q = deque([(start, [])])
        seen = {start}
        while q:
            cur, path = q.popleft()
            if not self._is_deadend(map_state, cur):
                return path
            for mv, nxt in self._get_neighbors(map_state, cur):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, path + [mv]))
        return []

    def _predict_ghost_next(self, map_state: np.ndarray, ghost, pac):
        """Dự đoán Ghost đi 1 bước: giả định Ghost chọn láng giềng
        làm TĂNG khoảng cách tới Pacman (tránh đâm đầu)."""
        best = ghost
        bestd = self._manhattan(ghost, pac)
        for mv, nxt in self._get_neighbors(map_state, ghost):
            d = self._manhattan(nxt, pac)
            if d > bestd:
                bestd = d
                best = nxt
        return best

    # ---------- main step ----------
    def step(self, map_state, my_pos, enemy_pos, step_number):
        if my_pos is None or enemy_pos is None:
            return Move.STAY

        # chuẩn hóa (row, col)
        start = (int(my_pos[0]), int(my_pos[1]))
        ghost = (int(enemy_pos[0]), int(enemy_pos[1]))

        # Nếu Ghost trong ngõ cụt -> chọn chốt ở lối ra của ngõ đó
        if self._is_deadend(map_state, ghost):
            exit_path = self._nearest_exit_from_deadend(map_state, ghost)
            # mục tiêu là ô cuối của path thoát (đi chặn)
            target = ghost if not exit_path else \
                     self._apply_moves(ghost, exit_path[-1:])[0] if exit_path else ghost
        else:
            # Dự đoán 1 bước Ghost và đuổi theo vị trí dự đoán
            predicted_ghost = self._predict_ghost_next(map_state, ghost, start)
            target = predicted_ghost

        dist = self._manhattan(start, ghost)

        # Xa thì A*, gần thì BFS
        path = self._astar_path(map_state, start, target) if dist > 6 \
               else self._bfs_path(map_state, start, target)

        # Lấy bước đầu
        if path:
            first = path[0]
            if isinstance(first, Move):
                return first

        # Fallback: chọn láng giềng tiến gần target nhất
        best_mv = Move.STAY
        best_h = 10**9
        for mv, nxt in self._get_neighbors(map_state, start):
            h = self._manhattan(nxt, target)
            if h < best_h:
                best_h = h
                best_mv = mv

        return best_mv






# ================================
# HIDER (Ghost)
# ================================
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_pos = None  # chống lắc qua lại

    def step(
        self,
        map_state: np.ndarray,
        my_position: Tuple[int, int],
        enemy_position: Tuple[int, int],
        step_number: int
    ) -> Move:
        start = (int(my_position[0]), int(my_position[1]))
        enemy = (int(enemy_position[0]), int(enemy_position[1]))

        # Ngưỡng nguy hiểm: khoảng cách nhỏ dễ bị bắt do di chuyển đồng thời
        danger = self._manhattan(start, enemy) <= 4

        # 1) Nếu nguy hiểm -> ưu tiên thoát gần, tránh ô Pacman sẽ tới ở bước kế
        if danger:
            mv = self._safe_escape_move(start, enemy, map_state)
            if isinstance(mv, Move):
                self._prev_pos = self._apply_move(start, mv)
                return mv

        # 2) Bình thường -> đi tới ô xa Pacman nhất (BFS từ Pacman),
        #    nhưng kiểm tra an toàn cho bước đầu (không giảm khoảng cách mạnh,
        #    không bước vào ô Pacman dự đoán sẽ tới).
        target = self.find_furthest_point(start, enemy, map_state)
        if target and target != start:
            path = self._bfs_path(start, target, map_state)
            if path:
                first_move = path[0]
                nxt = self._apply_move(start, first_move)
                pac_next = self._predict_pacman_next(enemy, start, map_state)

                # guard 1: không đi vào ô Pacman có khả năng sẽ tới
                if nxt == pac_next:
                    mv = self._safe_escape_move(start, enemy, map_state)
                    if isinstance(mv, Move):
                        self._prev_pos = self._apply_move(start, mv)
                        return mv
                # guard 2: không giảm khoảng cách mạnh + tránh đảo chiều ngay
                if (self._manhattan(nxt, enemy) >= self._manhattan(start, enemy) - 1
                        and (self._prev_pos is None or nxt != self._prev_pos)):
                    self._prev_pos = nxt
                    return first_move

        # 3) Fallback: tối đa hóa khoảng cách cục bộ, tránh ô Pacman dự đoán tới
        mv = self._maximize_local_distance_move(start, enemy, map_state)
        self._prev_pos = self._apply_move(start, mv)
        return mv

    # -------------------- Helpers --------------------
    def _is_valid_position(self, pos: Tuple[int, int], map_state: np.ndarray) -> bool:
        r, c = pos
        h, w = map_state.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        return map_state[r][c] == 0

    def _apply_move(self, pos: Tuple[int, int], move: Move) -> Tuple[int, int]:
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _get_neighbors(self, pos: Tuple[int, int], map_state: np.ndarray):
        neighbors = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._apply_move(pos, m)
            if self._is_valid_position(nxt, map_state):
                neighbors.append((nxt, m))
        return neighbors

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _bfs_path(self, start: Tuple[int, int], goal: Tuple[int, int], map_state: np.ndarray):
        from collections import deque
        if start == goal:
            return []
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            cur, path = queue.popleft()
            for nxt, mv in self._get_neighbors(cur, map_state):
                if nxt in visited:
                    continue
                if nxt == goal:
                    return path + [mv]
                visited.add(nxt)
                queue.append((nxt, path + [mv]))
        return []

    def _predict_pacman_next(
        self,
        pacman_pos: Tuple[int, int],
        ghost_pos: Tuple[int, int],
        map_state: np.ndarray
    ) -> Tuple[int, int]:
        """Dự đoán vị trí Pacman sau 1 bước: lấy bước đầu của BFS từ Pacman -> Ghost."""
        path = self._bfs_path(pacman_pos, ghost_pos, map_state)
        if path:
            return self._apply_move(pacman_pos, path[0])
        return pacman_pos  # nếu không tìm được đường, coi như đứng yên

    def _safe_escape_move(
        self,
        my_pos: Tuple[int, int],
        enemy_pos: Tuple[int, int],
        map_state: np.ndarray
    ) -> Move:
        """Chọn nước đi cục bộ an toàn khi ở gần Pacman:
        - Tránh ô Pacman dự đoán sẽ tới
        - Tối đa hóa khoảng cách sau khi Pacman cũng di chuyển
        - Hạn chế đảo chiều liên tục
        """
        pac_next = self._predict_pacman_next(enemy_pos, my_pos, map_state)
        cur_dist = self._manhattan(my_pos, enemy_pos)
        best_move = Move.STAY
        best_score = -10**9

        for nxt, mv in self._get_neighbors(my_pos, map_state):
            # Tránh bước ngay vào ô Pacman có thể tới
            if nxt == pac_next:
                continue
            # Tránh quay lại ô vừa đứng (giảm oscillation)
            if self._prev_pos is not None and nxt == self._prev_pos:
                continue

            # Ước lượng sau khi Pacman cũng tiến 1 bước (tới pac_next)
            dist_after = self._manhattan(nxt, pac_next)

            # Scoring: ưu tiên tăng khoảng cách; phạt nếu giảm mạnh
            score = dist_after
            if dist_after < cur_dist:
                score -= 5  # phạt nhẹ khi lùi khoảng cách

            if score > best_score:
                best_score = score
                best_move = mv

        # Nếu tất cả bị chặn bởi guard, chọn nước đi tăng khoảng cách hiện thời (không nhìn 1 bước)
        if best_move == Move.STAY:
            best_move = self._maximize_local_distance_move(my_pos, enemy_pos, map_state)
        return best_move

    def find_furthest_point(
        self,
        my_pos: Tuple[int, int],
        enemy_pos: Tuple[int, int],
        map_state: np.ndarray
    ) -> Tuple[int, int] | None:
        """BFS từ Pacman để tính khoảng cách tới mọi ô; chọn ô có khoảng cách tối đa,
        ưu tiên ô gần Ghost để dễ tiếp cận."""
        from collections import deque
        queue = deque([(enemy_pos, 0)])
        distance = {enemy_pos: 0}
        while queue:
            cur, d = queue.popleft()
            for nxt, _ in self._get_neighbors(cur, map_state):
                if nxt not in distance:
                    distance[nxt] = d + 1
                    queue.append((nxt, d + 1))

        if not distance:
            return None
        max_d = max(distance.values())
        cand = [p for p, v in distance.items() if v == max_d]
        if not cand:
            return None

        # Ưu tiên mục tiêu dễ tiếp cận (gần Ghost), tránh quay lại ô trước đó
        target = min(
            cand,
            key=lambda p: (self._manhattan(my_pos, p), 0 if self._prev_pos is None or p != self._prev_pos else 1)
        )
        return target

    def _maximize_local_distance_move(
        self,
        my_pos: Tuple[int, int],
        enemy_pos: Tuple[int, int],
        map_state: np.ndarray
    ) -> Move:
        """Chọn nước đi hợp lệ làm tăng khoảng cách Manhattan tức thời."""
        best_move = Move.STAY
        best_dist = self._manhattan(my_pos, enemy_pos)
        for nxt, mv in self._get_neighbors(my_pos, map_state):
            if self._prev_pos is not None and nxt == self._prev_pos:
                continue  # tránh lắc
            d = self._manhattan(nxt, enemy_pos)
            if d > best_dist:
                best_dist = d
                best_move = mv
        return best_move
