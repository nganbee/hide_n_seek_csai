
"""
Improved Pacman & Ghost Agents (BFS distance-based minimax + optional A* chase)
-------------------------------------------------------------------------------
- PacmanAgent: Minimax 1-ply. For each Pacman move, assume Ghost responds to
	maximize distance. Pacman chooses the move minimizing the *shortest-path*
	(grid-aware) distance after Ghost's best response. Uses BFS distance map
	(instead of Manhattan) to avoid being "blind to walls".
	+ Anti-swap: tránh trường hợp 2 bên đổi chỗ trong 1 lượt.
	+ Tie-break: tránh dao động, ưu tiên không quay lại ô trước đó.
	+ Optional: nếu Ghost đứng yên (kẹt) → Pacman có thể dùng A* để đuổi thẳng.

- GhostAgent: Ngược lại, cũng là minimax 1-ply. Với mỗi nước đi của Ghost,
	giả định Pacman phản ứng để rút ngắn khoảng cách nhất. Ghost chọn nước đi
	làm *tăng* khoảng cách ngắn nhất sau phản ứng của Pacman.
	+ Anti-swap, tie-break tương tự.
	+ Nếu đang ở "cổ chai" (degree thấp), cộng điểm cho ô mở để ưu tiên thoát.

Ghi chú:
- Các hàm khoảng cách dùng BFS để tính "khoảng cách đường đi thật sự" giữa các ô
	(tính đến tường/chướng ngại vật). Hàm A* được cung cấp để có thể đuổi mục tiêu
	trong trường hợp đặc biệt (ví dụ ghost đứng yên hoặc không có rủi ro phản ứng).
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterable
from collections import deque
import heapq
import numpy as np
import random

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


# ------------------ Hằng số & hướng di chuyển ------------------

# Mỗi phần tử: (dr, dc, MoveEnum)
DIRS: List[Tuple[int,int,Move]] = [
		(-1, 0, Move.UP),
		( 1, 0, Move.DOWN),
		( 0,-1, Move.LEFT),
		( 0, 1, Move.RIGHT), 
]
	

# ------------------ Các hàm tiện ích ------------------

""" Kiểm tra tọa độ có nằm trong ma trận"""
def in_bounds(r: int, c: int, h: int, w: int) -> bool:
		return 0 <= r < h and 0 <= c < w

""" Kiểm tra ô có trống (không phải tường)"""	
def is_free(pos: Tuple[int,int], grid: np.ndarray) -> bool:
		r, c = pos
		return grid[r, c] == 0

"""Trả về các nước đi hợp lệ từ vị trí hiện tại"""
def valid_moves(grid: np.ndarray, pos: Tuple[int,int]) -> List[Tuple[Move, Tuple[int,int]]]:
		"""
		Trả về danh sách các nước đi hợp lệ từ vị trí pos.
		Mỗi phần tử: (MoveEnum, (r_next, c_next)).
		"""
		h, w = grid.shape
		r, c = pos
		moves: List[Tuple[Move, Tuple[int,int]]] = []
		for dr, dc, mv in DIRS:
				nr, nc = r + dr, c + dc
				if in_bounds(nr, nc, h, w) and grid[nr, nc] == 0:
						moves.append((mv, (nr, nc)))
		return moves


def degree_open_cells(grid: np.ndarray, pos: Tuple[int,int]) -> int:
		"""
		'Độ mở' của ô: số đường đi ra khỏi ô này.
		Hữu ích để: tránh tự chui vào ngõ cụt (Pacman) hoặc ưu tiên thoát cổ chai (Ghost).
		"""
		return len(valid_moves(grid, pos))


def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
		"""Khoảng cách Manhattan (chỉ dùng phụ, BFS mới là chuẩn ở scoring)."""
		return abs(a[0]-b[0]) + abs(a[1]-b[1])


# ------------------ BFS distance map / Tìm đường ------------------

def distance_map_from(grid: np.ndarray, source: Tuple[int,int]) -> np.ndarray:
		"""
		Tạo bản đồ khoảng cách (BFS) từ 'source' đến mọi ô.
		- dist[r,c] = số bước ngắn nhất để đi từ source tới (r,c).
		- np.inf nếu không thể tới.
		Dùng để tra cứu nhanh ở scoring minimax.
		"""
		h, w = grid.shape
		dist = np.full((h, w), np.inf, dtype=float)
		if not in_bounds(source[0], source[1], h, w) or not is_free(source, grid):
				return dist

		q = deque([source])
		dist[source] = 0.0

		while q:
				r, c = q.popleft()
				for _, (nr, nc) in valid_moves(grid, (r, c)):
						if dist[nr, nc] == np.inf:
								dist[nr, nc] = dist[r, c] + 1.0
								q.append((nr, nc))
		return dist


def reconstruct_first_move(prev: Dict[Tuple[int,int], Tuple[Tuple[int,int], Move]],
													start: Tuple[int,int],
													goal: Tuple[int,int]) -> Optional[Move]:
		"""
		Dùng map 'prev' để truy vết đường đi từ goal về start và trả về nước đi đầu tiên.
		- Trả về None nếu goal không có trong prev (không tới được).
		- Trả về Move.STAY nếu start == goal.
		"""
		if start == goal:
				return Move.STAY
		if goal not in prev:
				return None
		cur = goal
		path_moves = []
		while cur != start:
				p, mv = prev[cur]
				path_moves.append(mv)
				cur = p
		path_moves.reverse()
		return path_moves[0] if path_moves else Move.STAY


def bfs_first_move(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[Move]:
		"""
		Tìm đường đi ngắn nhất bằng BFS và trả về *nước đi đầu tiên*.
		- None: không có đường.
		- Move.STAY: đã ở đích.
		"""
		if start == goal:
				return Move.STAY

		q = deque([start])
		visited = {start}
		prev: Dict[Tuple[int,int], Tuple[Tuple[int,int], Move]] = {}

		while q:
				cur = q.popleft()
				if cur == goal:
						break

				for mv, nxt in valid_moves(grid, cur):
						if nxt in visited:
								continue
						visited.add(nxt)
						prev[nxt] = (cur, mv)
						q.append(nxt)

		return reconstruct_first_move(prev, start, goal)


def astar_first_move(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[Move]:
		"""
		Tìm đường đi bằng A* (heuristic: Manhattan) và trả về *nước đi đầu tiên*.
		- None: không có đường.
		- Move.STAY: đã ở đích.
		"""
		if start == goal:
				return Move.STAY

		open_heap: List[Tuple[float, int, Tuple[int,int]]] = []  # (f, g, pos)
		g_cost: Dict[Tuple[int,int], int] = {start: 0}
		prev: Dict[Tuple[int,int], Tuple[Tuple[int,int], Move]] = {}

		f0 = manhattan(start, goal)
		heapq.heappush(open_heap, (f0, 0, start))
		visited = set()

		while open_heap:
				f, g, cur = heapq.heappop(open_heap)  
				if cur in visited:
						continue
				visited.add(cur)

				if cur == goal:
						return reconstruct_first_move(prev, start, goal)

				for mv, nxt in valid_moves(grid, cur):
						tentative_g = g + 1
						if tentative_g < g_cost.get(nxt, 1_000_000_000):
								g_cost[nxt] = tentative_g
								prev[nxt] = (cur, mv)
								f_nxt = tentative_g + manhattan(nxt, goal)
								heapq.heappush(open_heap, (f_nxt, tentative_g, nxt))

		return None


# ------------------ Pacman ------------------1

class PacmanAgent(BasePacmanAgent):
		"""
		Pacman (minimax 1-ply dựa trên BFS) – KHÔNG anti-swap phía Pacman:
		- Progress-first: nếu có nước đi làm giảm khoảng cách ngay (p_next → Ghost),
			chỉ xét các nước đó trước (bỏ loop-penalty để 'commit' hướng đúng).
		- Chống lặp ở góc: loop-penalty mềm (recency + phát hiện 3/4-cycle).
		- Nếu Ghost kẹt: A* đuổi trực tiếp (fallback BFS).
		"""
		def __init__(self, **kwargs):
				super().__init__(**kwargs)
				self.recent_positions: deque = deque(maxlen=8)  # lịch sử 8 vị trí gần nhất

		# ---------- loop-penalty mềm ----------
		def _loop_penalty(self, map_state, my_position, p_next) -> float:
				if not self.recent_positions:
						return 0.0
				cur_deg = degree_open_cells(map_state, my_position)
				choke_mult = 1.75 if cur_deg <= 2 else 1.0

				# recency: phạt ~ 1/i (gần phạt mạnh hơn)
				for i, pos in enumerate(reversed(self.recent_positions), start=1):
						if p_next == pos:
								return 0.35 * (1.0 / i) * choke_mult

				# cycle-close: khép vòng 3/4 bước
				pen = 0.0
				if len(self.recent_positions) >= 3 and p_next == self.recent_positions[-3]:
						pen += 0.55
				if len(self.recent_positions) >= 4 and p_next == self.recent_positions[-4]:
						pen += 0.75
				return pen

		# ---------- scoring mặc định 1-ply ----------
		def _score_one_ply(self, map_state, my_position, enemy_position, p_next, ghost_moves, use_loop_penalty=True):
				dist_from_p = distance_map_from(map_state, p_next)
				worst_for_pac = -1.0
				for _, g_next in ghost_moves:
						d = dist_from_p[g_next]
						if np.isinf(d): d = 1e9
						if d > worst_for_pac:
								worst_for_pac = d

				score = worst_for_pac - 0.2 * degree_open_cells(map_state, p_next)
				if use_loop_penalty:
						score += self._loop_penalty(map_state, my_position, p_next)
				return score

		# ---- Thêm vào trong lớp PacmanAgent ----

		def _would_swap(self, my_position, p_next, enemy_position, ghost_moves) -> bool:
				"""
				Trả về True nếu: Pacman định vào ô của Ghost, và Ghost có thể bước vào ô hiện tại của Pacman.
				-> nguy cơ 'đổi chỗ' trong cùng lượt.
				"""
				if p_next != enemy_position:
						return False
				return any(g_next == my_position for _, g_next in ghost_moves)

		def _swap_penalty(self, cur_dist: float, ghost_deg: int) -> float:
				"""
				Penalty đủ lớn để tránh swap trừ khi bắt buộc:
				- Ở gần (cur_dist <= 2) phạt MẠNH hơn (vì rủi ro cao).
				- Ghost ở góc/cổ chai (deg <= 2) thì nhân mạnh hơn để không tự hại.
				"""
				base = 6.0 if (not np.isinf(cur_dist) and cur_dist <= 2) else 3.0
				if ghost_deg <= 2:
						base *= 1.5
				return base

		
		def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
				"""
				Pacman minimax 1-ply (BFS-based) với:
				- Odd-distance sync: nếu dist hiện tại là SỐ LẺ và không có nước đi rút ngắn rõ ràng,
					Pacman đứng yên 1 lượt để đồng bộ pha, tránh bật ra.
				- Progress-first: nếu có nước rút ngắn ngay → chỉ xét chúng trước (bỏ loop-penalty để 'commit').
				- Tự tránh swap: loại nước đi swap nếu còn lựa chọn an toàn; nếu kẹt thì vẫn cho nhưng cộng penalty lớn.
				- Loop-penalty mềm chống lặp ở góc/ngã ba (nếu có _loop_penalty/_score_one_ply).
				"""

				# ---------- 0) Odd-distance sync trước khi làm gì khác ----------
				
				# Khoảng cách hiện tại (đường đi thực) từ Pacman tới Ghost
				dist_map_now = distance_map_from(map_state, my_position)
				cur_dist = dist_map_now[enemy_position]

				# Nếu không tới được → thử đuổi thẳng; nếu vẫn không, đứng yên
				if np.isinf(cur_dist):
						m = astar_first_move(map_state, my_position, enemy_position)
						if m is None:
								m = bfs_first_move(map_state, my_position, enemy_position)
						if m is not None:
								# cập nhật lịch sử nếu bạn có dùng
								if hasattr(self, "recent_positions"):
										self.recent_positions.append(my_position)
								return m
						return Move.STAY

				# Lấy nước đi của Ghost để biết nó có thể di chuyển hay không
				ghost_moves_now = valid_moves(map_state, enemy_position)
				ghost_has_non_stay = any(mv != Move.STAY for mv, _ in (ghost_moves_now or []))

				# CHỈ sync nếu: (1) khoảng cách lẻ & gần, (2) Ghost CÓ thể di chuyển (không kẹt/không STAY bắt buộc),
				# và (3) Pacman KHÔNG có bước nào giúp giảm khoảng cách dù chỉ 1 ô.
				if int(cur_dist) % 2 == 1 and cur_dist <= 5 and ghost_has_non_stay:
						has_strictly_shorter = False
						for _, p_next in valid_moves(map_state, my_position):
								d_next = distance_map_from(map_state, p_next)[enemy_position]
								if d_next + 1e-9 < cur_dist:  # bất kỳ giảm nào (>= 1 ô)
										has_strictly_shorter = True
										break
						if not has_strictly_shorter:
								if hasattr(self, "recent_positions"):
										self.recent_positions.append(my_position)
								return Move.STAY


				# ---------- 1) Lấy nước đi có thể của Ghost ----------
				ghost_moves = valid_moves(map_state, enemy_position)
				if not ghost_moves:
						# Ghost kẹt → đuổi thẳng
						m = astar_first_move(map_state, my_position, enemy_position)
						if m is None:
								m = bfs_first_move(map_state, my_position, enemy_position)
						if m is not None:
								if hasattr(self, "recent_positions"):
										self.recent_positions.append(my_position)
								return m
						ghost_moves = [(Move.STAY, enemy_position)]

				# ---------- 2) Ứng viên nước đi của Pacman + tự tránh swap nếu đối thủ không chặn ----------
				all_moves = valid_moves(map_state, my_position) or [(Move.STAY, my_position)]

				non_swap, swap_moves = [], []
				for m_p, p_next in all_moves:
						# Nếu có helper _would_swap thì dùng; nếu không có, coi như không phát hiện swap
						is_swap = hasattr(self, "_would_swap") and self._would_swap(my_position, p_next, enemy_position, ghost_moves)
						(swap_moves if is_swap else non_swap).append((m_p, p_next))

				# Nếu có lựa chọn an toàn → CHỈ xét non_swap; nếu không, buộc xét swap_moves
				candidates = non_swap if non_swap else swap_moves

				# ---------- 3) Progress-first: chỉ xét những nước rút ngắn ngay nếu có ----------
				progressive, nonprogress = [], []
				for m_p, p_next in candidates:
						d_now = distance_map_from(map_state, p_next)[enemy_position]
						if d_now + 1e-9 < cur_dist:
								progressive.append((m_p, p_next, d_now))
						else:
								nonprogress.append((m_p, p_next, d_now))

				pool = progressive if progressive else nonprogress

				# Thưởng tiến độ mạnh hơn nếu Ghost ở góc/cổ chai
				ghost_deg = degree_open_cells(map_state, enemy_position)
				progress_weight = 0.4 if ghost_deg <= 2 else 0.2

				# ---------- 4) Chấm điểm minimax 1-ply ----------
				best_score = float('inf')
				best_move = Move.STAY

				for m_p, p_next, d_immediate in pool:
						# Nếu là bước rút ngắn ngay → bỏ loop-penalty để không kéo lệch
						use_loop_penalty = False if pool is progressive else True

						# Nếu có helper _score_one_ply thì dùng; nếu không, chấm điểm tối thiểu
						if hasattr(self, "_score_one_ply"):
								score = self._score_one_ply(
										map_state, my_position, enemy_position, p_next, ghost_moves, use_loop_penalty=use_loop_penalty
								)
						else:
								# Fallback: worst-case khoảng cách sau phản ứng của Ghost - 0.2*độ mở
								dist_from_p = distance_map_from(map_state, p_next)
								worst_for_pac = max(
										(dist_from_p[g_next] if not np.isinf(dist_from_p[g_next]) else 1e9)
										for _, g_next in ghost_moves
								)
								score = worst_for_pac - 0.2 * degree_open_cells(map_state, p_next)
								# Không có loop-penalty fallback ở đây

						# Nếu nước đi này có nguy cơ swap → cộng penalty lớn (nếu có helper)
						if hasattr(self, "_would_swap") and self._would_swap(my_position, p_next, enemy_position, ghost_moves):
								if hasattr(self, "_swap_penalty"):
										score += self._swap_penalty(cur_dist, ghost_deg)
								else:
										score += 5.0  # penalty cố định nếu không có helper

						# Thưởng thêm cho rút ngắn ngay lập tức
						if d_immediate + 1e-9 < cur_dist:
								score -= progress_weight

						# Chọn tốt nhất (tie-break phụ: ưu tiên ô mở hơn)
						if score < best_score - 1e-9:
								best_score, best_move = score, m_p
						elif abs(score - best_score) <= 1e-9:
								if degree_open_cells(map_state, p_next) > 0:
										best_score, best_move = score, m_p

				# ---------- 5) Fallback nếu vẫn chưa có nước tốt ----------
				if best_move == Move.STAY:
						# Chọn nước có loop-penalty nhỏ nhất (nếu có helper); nếu không, chọn bất kỳ
						if hasattr(self, "_loop_penalty"):
								cand = []
								for m_p, p_next in all_moves:
										pen = self._loop_penalty(map_state, my_position, p_next)
										cand.append((pen, m_p))
								cand.sort(key=lambda x: x[0])
								mv = cand[0][1] if cand else Move.STAY
						else:
								mv = all_moves[0][0] if all_moves else Move.STAY
						if hasattr(self, "recent_positions"):
								self.recent_positions.append(my_position)
						return mv

				# ---------- 6) Cập nhật lịch sử & trả kết quả ----------
				if hasattr(self, "recent_positions"):
						self.recent_positions.append(my_position)
				return best_move



# ------------------ Ghost ------------------

class GhostAgent(BaseGhostAgent):
		"""
		Ghost (minimax 1-ply dựa trên khoảng cách BFS) + escape_depth heuristic:
		-----------------------------------------------------------------------
		- Giả định Pacman sẽ chọn nước đi giảm khoảng cách nhất.
			Ghost chọn nước đi làm khoảng cách sau đó LỚN nhất.
		- Dùng BFS distance thật (grid-aware).
		- Anti-swap: tránh đổi chỗ trong 1 lượt.
		- Nếu đang ở "cổ chai" (degree ≤ 2):
				+ Cộng điểm cho ô đích có độ mở cao (như cũ).
				+ Cộng thêm điểm theo "độ sâu hành lang" (escape_depth)
					— số bước thẳng trước khi gặp ngã rẽ hoặc tường.
		- Tie-break: tránh quay lại ô vừa qua.
		- Fallback: chọn ngẫu nhiên an toàn nếu mọi hướng bị chặn swap.
		"""

		def __init__(self, **kwargs):
				super().__init__(**kwargs)
				self.prev_pos: Optional[Tuple[int, int]] = None
				self.recent_positions = deque(maxlen=8)  

		# ---- NEW HELPER: tính độ sâu hành lang theo hướng hiện tại ----
		def _border_distance(self, grid: np.ndarray, pos: Tuple[int,int]) -> int:
				h, w = grid.shape
				r, c = pos
				return min(r, c, h-1-r, w-1-c)

		def _is_corner_like(self, grid: np.ndarray, pos: Tuple[int,int]) -> bool:
				# “góc/hẻm” = độ mở <= 1
				return degree_open_cells(grid, pos) <= 1

		def _escape_depth(self, grid: np.ndarray, start: Tuple[int, int],
											direction: Tuple[int, int], max_depth: int = 8) -> int:
				"""
				Đi thẳng theo hướng 'direction' cho tới khi gặp tường hoặc ngã rẽ.
				Trả về số ô có thể đi thẳng được (<= max_depth).
				"""
				h, w = grid.shape
				dr, dc = direction
				r, c = start
				depth = 0
				while depth < max_depth:
						r, c = r + dr, c + dc
						if not (0 <= r < h and 0 <= c < w):
								break
						if grid[r, c] != 0:  # tường
								break
						deg = degree_open_cells(grid, (r, c))
						depth += 1
						if deg > 2:  # tới ngã rẽ
								break
				return depth

		# ── Helper: một ô có phải ngã rẽ? (độ mở > 2) ──
		def _is_junction(self, grid: np.ndarray, pos: Tuple[int,int]) -> bool:
				return degree_open_cells(grid, pos) > 2

		# ── Helper: junction đầu tiên theo 1 hướng (bước tối đa max_depth) ──
		def _first_junction_ahead(self, grid: np.ndarray, start: Tuple[int,int],
															direction: Tuple[int,int], max_depth: int = 8) -> Optional[Tuple[Tuple[int,int], int]]:
				"""
				Từ 'start', đi THẲNG theo 'direction' tối đa max_depth bước.
				- Dừng khi gặp tường/ra biên → None.
				- Nếu gặp NGÃ RẼ (degree > 2) → trả về (vị trí_junction, số_bước_đã_đi).
				- Nếu hết bước mà chưa là ngã rẽ → None.
				"""
				h, w = grid.shape
				dr, dc = direction
				r, c = start
				for d in range(1, max_depth + 1):
						r, c = r + dr, c + dc
						if not (0 <= r < h and 0 <= c < w):  # ra biên
								return None
						if grid[r, c] != 0:                  # tường
								return None
						if self._is_junction(grid, (r, c)):  # gặp ngã rẽ
								return ((r, c), d)
				return None

		# ── Helper: tại 1 junction, đo độ sâu thẳng của các NHÁNH (trừ hướng quay lại) ──
		def _max_branch_depth_at(self, grid: np.ndarray, junction: Tuple[int,int],
															came_from: Tuple[int,int], max_depth: int = 8) -> Tuple[int, Optional[Tuple[int,int]]]:
				"""
				Ở 'junction', xét 3 hướng (loại bỏ hướng quay về 'came_from').
				Với mỗi hướng, đi THẲNG cho tới tường/ngã rẽ/đủ 8 bước, lấy độ sâu.
				Trả về (độ_sâu_lớn_nhất, hướng_tương_ứng) — nếu không có hướng hợp lệ → (0, None).
				"""
				rj, cj = junction
				rf, cf = came_from
				back_dir = (rf - rj, cf - cj)  # vector quay lại

				best_depth = 0
				best_dir: Optional[Tuple[int, int]] = None
				h, w = grid.shape

				for dr, dc, _ in DIRS:
						if (dr, dc) == back_dir:
								continue
						r, c = rj + dr, cj + dc
						if not (0 <= r < h and 0 <= c < w):
								continue
						if grid[r, c] != 0:
								continue

						depth = 1
						rr, cc = r, c
						while depth < max_depth:
								nr, nc = rr + dr, cc + dc
								if not (0 <= nr < h and 0 <= nc < w) or grid[nr, nc] != 0:
										break
								rr, cc = nr, nc
								depth += 1
								if self._is_junction(grid, (rr, cc)):
										break

						if depth > best_depth:
								best_depth = depth
								best_dir = (dr, dc)

				return best_depth, best_dir
		# ---- MAIN STEP ----
		def step(self, map_state: np.ndarray, my_position: tuple,
				enemy_position: tuple, step_number: int) -> Move:
				"""
				Ghost minimax 1-ply (BFS distance) + anti-swap mềm + junction-lookahead + chống lắc.
				ĐÃ BỎ: mọi penalty/bonus liên quan 'không đi qua góc' (dead-end/hành lang hẹp, xa mép, cấm STAY ở góc).
				"""

				# 1) Nước đi khả dĩ của Pacman (nếu kẹt, cho STAY)
				pac_moves = valid_moves(map_state, enemy_position)
				if not pac_moves:
						pac_moves = [(Move.STAY, enemy_position)]

				# 2) Pacman có thể bước vào ô hiện tại của Ghost ở lượt tới?
				pac_can_enter_my_tile = any(p_next == my_position for _, p_next in pac_moves)

				# 3) Độ mở hiện tại (chỉ dùng để kích hoạt escape_depth khi đang ở chỗ hẹp)
				my_deg = degree_open_cells(map_state, my_position)
				danger_now = pac_can_enter_my_tile  # KHÔNG coi ở góc là nguy cấp mặc định nữa

				# 4) Ứng viên di chuyển của Ghost
				my_moves = valid_moves(map_state, my_position) or [(Move.STAY, my_position)]

				best_score = -float('inf')
				best_move  = Move.STAY
				best_next  = None

				# Lưu phương án "swap" tốt nhất để fallback
				best_swap_score = -float('inf')
				best_swap_move  = None
				best_swap_next  = None

				# Lịch sử ngắn hạn chống lắc (nếu có)
				recent = list(getattr(self, "recent_positions", []))[::-1]  # gần nhất ở đầu

				for m_g, g_next in my_moves:
						# ---- Anti-swap mềm: đánh dấu chứ không loại ngay
						is_swap = any(
								(p_next == my_position) and (g_next == enemy_position)
								for _, p_next in pac_moves
						)

						# ---- Minimax 1-ply: Pacman chọn hướng làm khoảng cách NHỎ nhất
						dist_from_g = distance_map_from(map_state, g_next)

						best_for_pac = float('inf')
						for _, p_next in pac_moves:
								d = dist_from_g[p_next]
								if np.isinf(d):
										d = 1e9
								if d < best_for_pac:
										best_for_pac = d

						score = best_for_pac  # Ghost maximize

						# ---- (ĐÃ BỎ) phạt/ưu tiên liên quan góc/hành lang/xa biên ----
						# KHÔNG còn: deg_next penalty, border_distance bonus, cấm STAY ở góc

						# ---- Khi đang ở chỗ hẹp (degree ≤ 2): chỉ cộng nhẹ escape_depth (không phải anti-corner) ----
						if my_deg <= 2:
								dr = g_next[0] - my_position[0]
								dc = g_next[1] - my_position[1]
								if (dr, dc) != (0, 0):
										esc = self._escape_depth(map_state, my_position, (dr, dc), max_depth=8)
										score += 0.12 * esc  # giữ nhẹ để không áp đặt hướng

						# ---- Junction-lookahead từ g_next (≤ 8 ô): thưởng nhánh sâu nhất ----
						dr = g_next[0] - my_position[0]
						dc = g_next[1] - my_position[1]
						if (dr, dc) != (0, 0):
								first_junc = self._first_junction_ahead(map_state, g_next, (dr, dc), max_depth=8)
								if first_junc is not None:
										(jr, jc), d_to_j = first_junc
										# thưởng nhỏ nếu junction gần (rẽ sớm khi bị dí)
										score += 0.05 * max(0, (8 - d_to_j)) / 8.0
										# thưởng theo nhánh sâu nhất tại junction
										best_branch_depth, _ = self._max_branch_depth_at(
												map_state, (jr, jc), came_from=g_next, max_depth=8
										)
										score += 0.20 * best_branch_depth

						# ---- Phạt lặp đường rất ngắn hạn (chống lắc ping-pong), KHÔNG phải anti-corner ----
						if recent:
								try:
										k = recent.index(g_next) + 1  # 1 = vừa mới ở đó
										score -= 0.6 * (1.0 / k)      # càng gần thời gian → phạt càng mạnh
								except ValueError:
										pass

						# ---- Tie-break: tránh quay lại prev_pos khi điểm bằng nhau ----
						prefer_this = False
						if score > best_score + 1e-9:
								prefer_this = True
						elif abs(score - best_score) <= 1e-9:
								if self.prev_pos is None or g_next != self.prev_pos:
										prefer_this = True

						if not is_swap:
								if prefer_this:
										best_score, best_move, best_next = score, m_g, g_next
						else:
								# lưu ứng viên swap tốt nhất làm fallback
								better_swap = False
								if score > best_swap_score + 1e-9:
										better_swap = True
								elif abs(score - best_swap_score) <= 1e-9:
										if self.prev_pos is None or g_next != self.prev_pos:
												better_swap = True
								if better_swap:
										best_swap_score, best_swap_move, best_swap_next = score, m_g, g_next

				# --- Quyết định nước đi ---

				# 1) Có non-swap tốt → dùng
				if best_move != Move.STAY or best_next is not None:
						if hasattr(self, "recent_positions"):
								self.recent_positions.append(my_position)
						self.prev_pos = my_position
						return best_move

				# 2) Nếu tất cả đều swap:
				#    - Khi nguy cấp (Pacman có thể đè) → cho phép swap tốt nhất
				#    - Hoặc chẳng còn gì khác → vẫn chọn swap tốt nhất (tốt hơn STAY)
				if best_swap_move is not None and (danger_now or True):
						if hasattr(self, "recent_positions"):
								self.recent_positions.append(my_position)
						self.prev_pos = my_position
						return best_swap_move

				# 3) Tránh STAY nếu Pacman có thể đè ngay lượt tới
				if pac_can_enter_my_tile:
						safe = [
								(m, n) for (m, n) in my_moves
								if not any((p_next == my_position) and (n == enemy_position) for _, p_next in pac_moves)
						]
						if safe:
								m, _ = random.choice(safe)
								if hasattr(self, "recent_positions"):
										self.recent_positions.append(my_position)
								self.prev_pos = my_position
								return m
						if best_swap_move is not None:
								if hasattr(self, "recent_positions"):
										self.recent_positions.append(my_position)
								self.prev_pos = my_position
								return best_swap_move

				# 4) Bất khả kháng
				if hasattr(self, "recent_positions"):
						self.recent_positions.append(my_position)
				self.prev_pos = my_position
				return Move.STAY


