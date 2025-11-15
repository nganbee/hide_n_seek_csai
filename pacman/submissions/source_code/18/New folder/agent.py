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

class Queue:
    def __init__(self):
        """Khởi tạo một hàng đợi rỗng (dùng list)."""
        self.items = []

    def put(self, item):
        """Thêm một phần tử vào CUỐI hàng đợi."""
        self.items.append(item)

    def get(self):
        """Lấy và xóa một phần tử từ ĐẦU hàng đợi."""
        if not self.empty():
            return self.items.pop(0)
        return None # Trả về None nếu hàng đợi rỗng

    def empty(self) -> bool:
        """Kiểm tra xem hàng đợi có rỗng không."""
        return len(self.items) == 0


class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.name = "Example Greedy Pacman"

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:

        # Trường hợp hai agent đứng cạnh nhau và đường cùng
        # Nếu 2 agent cùng di chuyển thì 2 agent không the đứng trên cùng 1 ô -> chưa kết thúc
        # điều này cũng xảy ra nếu cả khi 2 agent di chuyển ngược hướng nhau
        # Hiện tại chưa thể xác định đối thủ sẽ di chuyển hay không trong trường hợp này
        # nên sẽ cho agent P quyết định ngẫu nhiên di chuyển hay đứng yên
        if self.manhattan(enemy_position,my_position) == 1 and step_number < 180:
            random.seed()
            t = random.randrange(0, 2, 1)
            if t == 0:
                return Move.STAY

        """ 
        Thuật toán sơ khai (cho lần nộp 1): thuật toán tìm kiếm loang (BFS)
        - Tạo một ma trận, (i,j) lưu so bước tối thiểu từ P tới ô i,j
        - Chon ô của G làm đích, hoặc ô gần G nhất mà P có thể tới
        - Truy ngược lại từ đích về P để tìm ra đường nhanh nhất từ P đến ô đích
        _ Lựa chọn nước đi tiếp theo dựa trên đường tìm được
        Kích cỡ ma trận bản đồ là 21x21, đủ nhỏ để đảm bảo thuật toán chạy dưới 1s
        """

        # Khởi tạo ma trận số bước
        rows, cols = map_state.shape
        map_step = [[-1 for _ in range (cols)] for _ in range(rows)]

        # khởi tạo các biến cần thiết
        min_dist = rows + cols + 100
        min_move = rows + cols + 100

        # Hàng đợi để duyệt các bước di chuyển
        moves = Queue()
        moves.put(my_position)
        map_step[my_position[0]][my_position[1]] = 0
        dest = [-1, -1]

        direction = [[-1, 0], [1, 0], [0, 1], [0, -1]]

        # Tính ma trận bước đi
        while not moves.empty():
        # Khi vẫn còn ô để duyệt
            pos = moves.get()
            for mv in direction:
                new_pos = (pos[0] + mv[0], pos[1] + mv[1])
                if self._is_valid_position(new_pos, map_state):
                    if map_step[new_pos[0]][new_pos[1]] == -1:
                    # nếu ô chưa được khám phá
                        map_step[new_pos[0]][new_pos[1]] = map_step[pos[0]][pos[1]] + 1
                        moves.put(new_pos)
                    else:
                    # chỉ cập nhật nếu đường đi đang xét là tốt hơn
                        if map_step[pos[0]][pos[1]] + 1 < map_step[new_pos[0]][new_pos[1]]:
                            map_step[new_pos[0]][new_pos[1]] = map_step[pos[0]][pos[1]] + 1
                            moves.put(new_pos)

                    # kiểm tra ô hiện tại là ô địch đứng (hoặc ô gần địch nhất)
                    # trong trường hợp có 2 ô cùng khoảng cách manhattan, chọn ô gần P nhất
                    x = self.manhattan(new_pos, enemy_position)
                    if (min_dist > x) or (min_dist == x and min_move > map_step[new_pos[0]][new_pos[1]]):
                        min_dist = x
                        min_move = map_step[new_pos[0]][new_pos[1]]
                        dest = new_pos

        # từ ma trận bước đi, truy vết đường đi nhanh nhất
        pos = dest
        while map_step[pos[0]][pos[1]] > 0:
            for mv in direction:
                new_pos = (pos[0] + mv[0], pos[1] + mv[1])
                if map_step[new_pos[0]][new_pos[1]] == map_step[pos[0]][pos[1]] - 1:
                # 2 ô lân cận có khoảng cách bước đi = 1 sẽ là 2 ô trên cùng 1 đường di chuyển từ P
                    if map_step[new_pos[0]][new_pos[1]] == 0:
                    # tới được vị trí của P
                        dest = pos
                    pos = new_pos

        # trả đáp án
        if (my_position[0] + 0, my_position[1] + 1) == dest:
            return Move.RIGHT
        if (my_position[0] + 0, my_position[1] - 1) == dest:
            return Move.LEFT
        if (my_position[0] + 1, my_position[1] + 0) == dest:
            return Move.DOWN
        if (my_position[0] - 1, my_position[1] + 0) == dest:
            return Move.UP

        # đảm bảo hàm vẫn trả giá trị nếu đoạn mã trên lỗi
        return Move.STAY


    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """kiểm tra một ô là hợp lệ"""
        row, col = pos
        height, width = map_state.shape

        if row < 0 or row >= height or col < 0 or col >= width:
            return False

        return map_state[row, col] == 0

    def manhattan(self, pos1: tuple, pos2: tuple) -> int:
        """ khoảng cách manhattan"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class GhostAgent(BaseGhostAgent):
    """
    Chạy đến điểm xa nhất từ pacman, tránh đường cụt
    1. Chạy BFS từ Pacman

    2. heuristic = Khoảng cách từ Pacman) + (0.5 nếu là Giao lộ).



    3. Logic Chính:
       a. Tìm các điểm xa Pacman nhất (max_dist).
       b. Phân loại chúng thành:
        Giao lộ là có 3, 4,... đường đi
        Hành lang là có 0,1 đường đi
          - Ưu tiên 1: Điểm an toàn là GIAO LỘ.
          - Ưu tiên 2: Điểm an toàn là HÀNH LANG.
       c. Tìm đường đến Ưu tiên 1 trước.
       d. Nếu không có/không đến được, mới tìm đường đến Ưu tiên 2.
       e. Nếu Pacman quá gần ưu tiên đi xa Pacman trước
    """
    
    def __init__(self, **kwargs):
        """Khởi tạo Ghost agent."""
        super().__init__(**kwargs)
        self.name = "Ghost Agent"

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        """
        Thực hiện một bước di chuyển.
        """

        # 1. Chạy BFS từ Pacman để lấy bản đồ khoảng cách
        dist_map = self._bfs_distance_map(enemy_position, map_state)
        
        # 2. Kiểm tra khoảng cách hiện tại
        my_current_distance = dist_map.get(my_position, 0)
        
        # Nếu Pacman ở quá gần (khoảng cách 4 hoặc ít hơn)
        if my_current_distance <= 4:
            return self._flee_greedily(my_position, dist_map, map_state)

        # 3. Tìm điểm xa pacman nhất
        if not dist_map: # Nếu bản đồ khoảng cách trống (bị kẹt)
            return Move.STAY
            
        max_dist = -1
        for dist in dist_map.values():
            if dist > max_dist:
                max_dist = dist
        
        # Lấy tất cả các vị trí an toàn nhất
        all_safest_spots = set()
        for pos, dist in dist_map.items():
            if dist == max_dist:
                all_safest_spots.add(pos)
                
        # Phân loại các điểm an toàn:
        # Ưu tiên 1: Các điểm an toàn VÀ là giao lộ
        safe_junctions = set()
        # Ưu tiên 2: Các điểm an toàn nhưng là hành lang/đường cụt
        safe_corridors = set()
        
        for pos in all_safest_spots:
            if self._is_junction(pos, map_state):
                safe_junctions.add(pos)
            else:
                safe_corridors.add(pos)

        # 4. Tìm đường đến điểm an toàn đã tìm
        path = []
        
        # Luôn luôn ưu tiên tìm đường đến một giao lộ an toàn trước
        if safe_junctions:
            path = self._bfs_path_to_nearest_goal(my_position, 
                                                  safe_junctions, 
                                                  map_state)
        
        # Nếu không tìm thấy đường (hoặc không có giao lộ an toàn nào)
        # thì mới chấp nhận đi đến một hành lang an toàn
        if not path and safe_corridors:
             path = self._bfs_path_to_nearest_goal(my_position, 
                                                   safe_corridors, 
                                                   map_state)
        if path:
            return path[0] # Đi bước đầu tiên
            
        return Move.STAY
    
    
    def _flee_greedily(self, my_position: tuple, 
                             dist_map: dict, 
                             map_state: np.ndarray) -> Move:
        """
        Tìm một bước di chuyển (bao gồm cả STAY) để tối đa hóa "điểm số".
        Điểm = (Khoảng cách) + (0.5 nếu là Giao lộ)
        """
        best_move = Move.STAY
        best_score = -9999.0
        
        # Kiểm tra tất cả các bước di chuyển khả thi, TÍNH CẢ STAY
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            next_pos = self._apply_move(my_position, move)
            
            # Chỉ xem xét nếu bước đi là hợp lệ
            if self._is_valid_position(next_pos, map_state):
                
                # Tính điểm cơ bản (là khoảng cách)
                # Dùng -1 nếu không có trong map
                score = float(dist_map.get(next_pos, -1))
                
                # Áp dụng heuristic:
                if self._is_junction(next_pos, map_state):
                    score += 0.5
                    
                # Nếu điểm này tốt hơn, cập nhật
                if score > best_score:
                    best_score = score
                    best_move = move
                    
        return best_move

    def _bfs_distance_map(self, start_pos: tuple, map_state: np.ndarray) -> dict:
        """
        Chạy BFS từ 'start_pos' để tính toán khoảng cách
        """
        queue = [(start_pos, 0)] 
        visited_distances = {start_pos: 0}
        
        while queue:
            current_pos, dist = queue.pop(0) 
            
            for next_pos, _ in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited_distances:
                    visited_distances[next_pos] = dist + 1
                    queue.append((next_pos, dist + 1))
                    
        return visited_distances

    def _bfs_path_to_nearest_goal(self, start_pos: tuple, 
                                  goals_set: set, 
                                  map_state: np.ndarray) -> list:
        """
        Chạy BFS từ 'start_pos' để tìm đường đi ngắn nhất
        đến BẤT KỲ vị trí nào trong 'goals_set'.
        """
        if not goals_set: # Thêm kiểm tra
            return []
        if start_pos in goals_set:
            return [] 

        queue = [(start_pos, [])] 
        visited = {start_pos}
        
        while queue:
            current_pos, path = queue.pop(0) 
            
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    new_path = path + [move]
                    
                    if next_pos in goals_set:
                        return new_path 
                    
                    visited.add(next_pos)
                    queue.append((next_pos, new_path))
        
        return [] 
        


    def _is_junction(self, pos: tuple, map_state: np.ndarray) -> bool:
        """
        Kiểm tra xem một vị trí có phải là 'giao lộ' (>= 3 neighbors) hay không.
        - 1 neighbors= Đường cụt (Dead end)
        - 2 neighbors = Hành lang (Corridor)
        - 3 hoặc 4 neighbors = Giao lộ (Junction) -> An toàn
        """
        # _get_neighbors chỉ trả về các ô hợp lệ (không phải tường, trong bản đồ)
        return len(self._get_neighbors(pos, map_state)) > 2

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Kiểm tra xem vị trí có hợp lệ không (không phải tường, trong bản đồ)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        """Áp dụng một bước di chuyển vào vị trí, trả về vị trí mới."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        """Lấy tất cả các vị trí hàng xóm hợp lệ và bước di chuyển tương ứng."""
        neighbors = []
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        
        return neighbors