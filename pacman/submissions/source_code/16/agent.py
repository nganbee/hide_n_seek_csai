import sys
from pathlib import Path
from heapq import heappush, heappop

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

class PacmanAgent(BasePacmanAgent):

    #Hàm khởi tạo
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    #Kiểm tra vị trí hợp lệ (Không vượt ra ngoài map, không phải tường)
    def _is_valid_position(self, pos, map_state):
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    #Tính vị trí mới sau khi thực hiện nước đi
    def _apply_move(self, pos, move):
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    #Lấy các ô bên cạnh có thể đi được
    def _get_neighbors(self, pos, map_state):
        neighbors = []
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        
        return neighbors

    #Tính khoảng cách Manhattan
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    #Nước đi hợp lệ của Ghost
    def _ghost_is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    #Dự đoán nước đi của Ghost
    def _predict_ghost_move(self, my_position, enemy_position, map_state):
        #Tính chênh lệch hàng, cột 
        row_diff = enemy_position[0] - my_position[0] 
        col_diff = enemy_position[1] - my_position[1]
        
        #Giả sử Ghost chạy ra xa khỏi Pacman, 
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
        
        # Kiểm tra xem nước đi của Ghost có hợp lệ không
        if self._ghost_is_valid_move(enemy_position, move, map_state):
            return self._apply_move(enemy_position, move)
        
        # Nếu không, thử các nước đi dự phòng
        for backup_move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._ghost_is_valid_move(enemy_position, backup_move, map_state):
                return self._apply_move(enemy_position, backup_move)
        
        # Nếu bị kẹt hoàn toàn, Ghost sẽ đứng yên
        return self._apply_move(enemy_position, Move.STAY)

    #Thuật toán A* để tìm đường đi tối ưu từ Pacman đến Ghost
    def astar(self, start, goal, map_state):  
        
        #Hàm heuristic dùng khoảng cách Manhattan
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        #Biến counter để không bị tình trạng trùng độ ưu tiên khi f_cost bằng nhau
        counter = 0
        
        #Khởi tạo hàng đợi ưu tiên
        frontier = [(0, 0, start, [])] 
        visited = set()
        
        #Bắt đầu vòng lặp
        while frontier:
            #Lấy ra từ hàng đợi ưu tiên ô có f_cost thấp nhất
            f_cost, _, current_pos, path = heappop(frontier)
            
            if current_pos == goal:
                return path
            
            if current_pos in visited:
                continue
            
            #Đánh dấu là đã xử lí
            visited.add(current_pos)
            
            #Đánh giá các ô có thể đi được từ ô hiện tại
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    #Nếu ô chưa xử lí, tính điểm cho ô
                    new_path = path + [move]
                    g_cost = len(new_path)
                    h_cost = heuristic(next_pos)
                    f_cost = g_cost + h_cost
                    
                    #Tăng counter lên 1
                    counter += 1

                    #Push ô này vào hàng đợi ưu tiên
                    heappush(frontier, (f_cost, counter, next_pos, new_path))
        
        return [Move.STAY]

    def step(self, map_state, my_position, enemy_position, step_number):
        #Dự đoán nước đi của Ghost
        predicted_enemy_pos = self._predict_ghost_move(my_position, enemy_position, map_state)

        #Nếu Ghost bị kẹt, sắp di chuyển về ô của Pacman
        if predicted_enemy_pos == my_position:
            #Pacman chỉ cần đứng yên và bắt
            return Move.STAY 
        
        #Chạy A* đến vị trí dự đoán
        path = self.astar(my_position, predicted_enemy_pos, map_state)
        #Nếu A* tìm thấy đường đi thì đi về ô đầu tiên trong đường đi đó
        if path and path[0] != Move.STAY:
            return path[0]
        
        #Nếu không tìm thấy đường đi, chạy đến vị trí hiện tại của Ghost
        path_backup = self.astar(my_position, enemy_position, map_state)
        if path_backup and path_backup[0] != Move.STAY:
            return path_backup[0]

        return Move.STAY


class GhostAgent(BaseGhostAgent):

    #Lưu trữ các ngõ cụt, khởi tạo là None
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dead_ends = None
    
    #Sử dụng thuật toán Greedy + hàm đánh giá Heuristic
    def step(self, map_state: np.ndarray, my_position: tuple,
             enemy_position: tuple, step_number: int) -> Move:

        #Chạy hàm compute_dead_ends để lưu địa chỉ các ngõ cụt
        if self.dead_ends is None:
            self.dead_ends = self._compute_dead_ends(map_state)
        
        #Khởi tạo nước đi tốt nhất, điểm tốt nhất mặc định
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        best_move = Move.STAY
        best_score = -float('inf')

        #Chạy vòng for để xét tọa độ của ô muốn đi tới
        for move in moves:
            nr, nc = my_position[0] + move.value[0], my_position[1] + move.value[1]
            
            #Nếu ô đó nằm ngoài map hoặc chạm tường, bỏ qua
            if not self._is_valid((nr, nc), map_state):
                continue

            #Nếu ô đó thuộc ngõ cụt, bỏ qua
            if (nr, nc) in self.dead_ends:
                continue  

            #Khi đã chọn được ô, chấm điểm Heuristic cho ô này
            #Tính khoảng cách Manhattan, đếm số lối thoát của ô
            dist = abs(nr - enemy_position[0]) + abs(nc - enemy_position[1])
            exits = self._count_exits((nr, nc), map_state)

            #Hàm đánh giá Heuristic ưu tiên khoảng cách, sau đó ưu tiên số lối thoát
            score = dist * 10 + exits * 5

            #Nếu Pacman đến quá gần, trừ điểm ô gần Pacman khiến Ghost ưu tiên chạy ra xa
            if dist <= 5:
                score -= 100
            
            #So sánh điểm của nước đi hiện tại với nước đi tốt nhất, chọn nước đi tốt hơn
            if score > best_score:
                best_score = score
                best_move = move
        #Sau khi thử tất cả hướng đi, chọn nước đi tốt nhất
        return best_move

    #Các hàm trợ giúp
    #Kiểm tra ô có phải là ô hợp lệ để đi hay không 
    def _is_valid(self, pos, grid):
        r, c = pos
        return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == 0

    #Đếm số ô trống xung quanh
    def _count_exits(self, pos, grid):
        r, c = pos
        count = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
                count += 1
        return count

    #Phát hiện các ngõ cụt
    def _compute_dead_ends(self, grid):
        #Khởi tạo rỗng
        dead = set()
        changed = True

        #Chạy vòng lặp để tìm ngõ cụt
        while changed:
            changed = False
            #Duyệt qua từng ô trên toàn bản đồ
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    #Nếu ô không phải là tường và chưa bị đánh dấu nằm trong ngõ cụt
                    if grid[r, c] == 0 and (r, c) not in dead:
                        #Kiểm tra các ô xung quanh để tìm lối thoát
                        exits = [(r+dr, c+dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                                 #Lối thoát nằm trong bản đồ, không phải là tường và không nằm trong danh sách dead 
                                 if 0 <= r+dr < grid.shape[0] and 0 <= c+dc < grid.shape[1]
                                 and grid[r+dr, c+dc] == 0 and (r+dr, c+dc) not in dead]
                        #Sau khi đếm xong, nếu số lối thoát <= 1 thì nó là ngõ cụt, thêm vào danh sách dead
                        if len(exits) <= 1:
                            dead.add((r, c))
                        #Đặt lại changed = True để vòng lặp chạy lại, tìm hết các ô nằm trong ngõ cụt
                            changed = True
        #Trả về danh sách các ô nằm trong ngõ cụt
        return dead
