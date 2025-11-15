import sys
from pathlib import Path
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """Pacman (Seek) Agent - Dùng BFS để tìm đường ngắn nhất đến Ghost."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS_Seeker"
        self.path = []  # lưu đường đi tạm thời

    def step(self, map_state: np.ndarray, my_pos: tuple, enemy_pos: tuple, step_number: int) -> Move:
        # Nếu chưa có đường đi hoặc đã đến cuối đường thì tính lại BFS
        if not self.path:
            self.path = self._bfs(map_state, my_pos, enemy_pos)

        # Nếu BFS có kết quả -> đi theo bước đầu tiên trong path
        if self.path:
            return self.path.pop(0)

        # Nếu không có đường hợp lệ -> đứng yên
        return Move.STAY

    def _bfs(self, map_state, start, goal):
        """Tìm đường ngắn nhất từ start -> goal bằng BFS."""
        queue = deque([(start, [])])
        visited = {start}
        directions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

        while queue:
            pos, path = queue.popleft()
            if pos == goal:
                return path

            for move in directions:
                delta = move.value
                new_pos = (pos[0] + delta[0], pos[1] + delta[1])

                if self._is_valid_position(new_pos, map_state) and new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, path + [move]))
        return []

    def _is_valid_position(self, pos, map_state):
        """Kiểm tra xem vị trí có hợp lệ không (trong bản đồ và không phải tường)."""
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0


class GhostAgent(BaseGhostAgent):  
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS_Hider"
    
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple,step_number: int) -> Move:
        # Duyệt toàn bản đồ bằng BFS từ vị trí Ghost
        farthest_path = self._bfs_farthest(map_state, my_position, enemy_position)

        # Nếu BFS tìm được đường đi -> đi theo bước đầu tiên
        if farthest_path:
            return farthest_path[0]

        # Không tìm thấy đường nào hợp lệ -> đứng yên
        return Move.STAY
    
    def _bfs_farthest(self, map_state, start, pacman_pos):
            """Chạy BFS từ Ghost để tìm vị trí xa Pacman nhất."""
            from collections import deque
            queue = deque([(start, [])])
            visited = {start}
            directions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

            best_path = []
            max_dist = -1

            while queue:
                pos, path = queue.popleft()

                # Tính khoảng cách Manhattan giữa ô hiện tại và Pacman
                dist_to_pacman = abs(pos[0] - pacman_pos[0]) + abs(pos[1] - pacman_pos[1])
                if dist_to_pacman > max_dist:
                    max_dist = dist_to_pacman
                    best_path = path

                for move in directions:
                    delta = move.value
                    new_pos = (pos[0] + delta[0], pos[1] + delta[1])

                    if self._is_valid_position(new_pos, map_state) and new_pos not in visited:
                        visited.add(new_pos)
                        queue.append((new_pos, path + [move]))

            return best_path  # trả về đường đến vị trí xa nhất
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
