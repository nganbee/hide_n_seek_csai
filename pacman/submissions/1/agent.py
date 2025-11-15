"""
Group 1 members:
23120022 - Hoàng Gia Bảo
23120028 - Nguyễn Hoàng Danh
23120338 - Hoàng Hùng Quân
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

class PacmanAgent(BasePacmanAgent):
    
    def __init__(self, **kwargs):
        """
        Initialize the Pacman agent.
        """
        self.name = "A-star & Predict ghost with Minimax - Alpha-Beta Pruning"
        self.dead_end_set = set()               # Set of positions identified as dead ends on the map
        self.transposition_table = {}           # Cache for previously evaluated states in Minimax (transposition table)
        from collections import deque           
        self.history_moves = ({}, deque([]))    # History of moves as (move frequency, move sequence)
        self.last_state = None                  # Previous positions of Pacman and Ghost
        self.is_swapped = False                 # Flag indicating if Pacman and Ghost have swapped positions
        self.is_dead_end_calculated = False     # Indicates whether dead-end positions have been precomputed
        self.blockade = False                   # Flag indicating if Pacman and Ghost are stuck in a blocking loop
        self.last_move = Move.STAY              # Pacman's previous move
        self.virtual_block_pos = None           # Temporary wall position used during blockade detection
        self.block_timeout = 5                  # Duration before removing temporary wall (used in blockade handling)
        self.MAX_PREDICTED_DEPTH = 10           # Max depth for Pacman's Minimax prediction of Ghost's future moves

    def step(
            self,
            map_state: np.ndarray, 
            my_position: tuple[int, int], 
            enemy_position: tuple[int, int],
            step_number: int
            ) -> Move:
        """
        Determine Pacman's next move based on the current game state.

        This agent combines two main strategies:
            - A* search to find the shortest path to a target.
            - Minimax with Alpha-Beta Pruning to predict Ghost's future moves.

        On the first step, Pacman pre-computes all dead-end positions on the map for future use.

        If Pacman and Ghost have swapped positions in the previous turn and are now adjacent, Pacman will stay still.

        If Ghost is currently in a dead-end, Pacman targets Ghost's current position.
        Otherwise, Pacman predicts Ghost's future position using Minimax and targets that instead.

        To avoid infinite loops, Pacman tracks the last 50 game states. If a loop is detected (repeated states),
        Pacman sets a temporary wall at its current position for a few steps and moves in the opposite direction.

        Args:
            map_state (np.ndarray): The current map layout.
            my_position (tuple[int, int]): Pacman's current position (row, column).
            enemy_position (tuple[int, int]): Ghost's current position (row, column).
            step_number (int): The current step number (maximum 200).

        Returns:
            Move: The next move Pacman should take. Returns Move.STAY if no valid path is found or if staying is strategic.
        """

        # Compute and store dead-end positions on the map (only once)
        if not self.is_dead_end_calculated:
            self.is_dead_end_calculated = True
            _, self.dead_end_set = self._compute_dead_end_map(map_state)

        # Initialize last_state with the current positions on the first step
        if not self.last_state:
            self.last_state = (my_position, enemy_position)

        # If positions were swapped and they're close, Pacman stays
        if self._manhattan_distance(my_position, enemy_position) <= 1 and self.is_swapped:
            self.last_move = Move.STAY
            return Move.STAY
        
        # Detect if Pacman and Ghost have swapped positions
        if self.last_state == (enemy_position, my_position):
            self.is_swapped = True

        # Update last_state with current positions
        self.last_state = (my_position, enemy_position)

        # If history exceeds 50 entries, remove oldest and update frequency
        if len(self.history_moves[1]) > 50:
            first_state = self.history_moves[1].popleft()
            self.history_moves[0][first_state] -= 1
            if not self.history_moves[0][first_state]:
                self.history_moves[0].pop(first_state)

        # Detect loop by checking if current state has appeared before
        if (my_position, enemy_position) in self.history_moves[0]:
            self.history_moves[0][(my_position, enemy_position)] += 1
            self.history_moves[1].append((my_position, enemy_position))
            self.blockade = self.history_moves[0][(my_position, enemy_position)] > 1
        else:
            self.history_moves[0][(my_position, enemy_position)] = 1
            self.history_moves[1].append((my_position, enemy_position))

        # Predict Ghost's future move using Minimax and Alpha-Beta Pruning
        _, predicted_enemy_move = self._minimax_alpha_beta_pruning((my_position, enemy_position), self.MAX_PREDICTED_DEPTH, float("-inf"), float("inf"), False, map_state)
        
        # Apply temporary wall if blockade is active, if Ghost is not in a dead end, find a path to its future position instead of its current position
        if self.virtual_block_pos:
            map_state[self.virtual_block_pos] = 1
        next_move = self._astar(my_position, enemy_position if enemy_position in self.dead_end_set else self._apply_move(enemy_position, predicted_enemy_move), map_state)

        # Restore original map by removing temporary wall
        if self.virtual_block_pos:
            map_state[self.virtual_block_pos] = 0
        
        # Reverse last move for use in blockade escape
        reversed_move = self._get_opposite_move(self.last_move)
        
        # Decrease blockade timer each step
        self.block_timeout -= 1

        # Clear temporary wall when timeout expires
        if self.block_timeout < 1:
            self.virtual_block_pos = None

        # If loop detected, initiate blockade escape strategy
        # Set a temporary wall at the Pacman's current position, last for 5 steps.
        # Then clear all the history moves for the next loop detection.
        # Pacman will move in the reverse direction.
        if self.blockade:
            self.block_timeout = 5
            self.virtual_block_pos = my_position
            self.blockade = False
            self.history_moves[0].clear()
            self.history_moves[1].clear()
            return reversed_move
        
        # If no blockade, proceed with predicted path to catch Ghost
        self.last_move = next_move
        return next_move
    
    def _minimax_alpha_beta_pruning(
            self,
            state: tuple[tuple[int, int], tuple[int, int]],
            depth: int,
            alpha: float | int,
            beta: float | int,
            is_maximizing_player: bool,
            map_state: np.ndarray
            ) -> tuple[int, Move]:
        """
        Predicts the ghost's next move using the Minimax algorithm with Alpha-Beta Pruning.

        This function evaluates game states recursively to determine the optimal move for 
        either Pacman (maximizing player) or Ghost (minimizing player). It uses a transposition 
        table to cache previously evaluated states (only states that have shallow depth) and avoid 
        redundant computation.

        Evaluation stops when the search depth reaches 0 or when Pacman catches the Ghost (both occupy 
        the same cell). In such cases, the score is set to the negative Manhattan distance between their 
        positions.

        Alpha-Beta pruning is applied to eliminate branches that cannot influence the final decision:
            - Pacman updates alpha and prunes when alpha >= beta (fail-high).
            - Ghost updates beta and prunes when beta <= alpha (fail-low).
        
        Cached entries in the transposition table are reused based on depth and pruning flags:
            - "EXACT": score is precise.
            - "LOWERBOUND": score is at least the cached value.
            - "UPPERBOUND": score is at most the cached value.
        
        Args:
            state (tuple[tuple[int, int], tuple[int, int]]): A tuple of (pacman_pos, ghost_pos), each as (row, column).
            depth (int): The remaining search depth.
            alpha (float | int): The best already explored option along the path to the maximizer.
            beta (float | int): The best already explored option along the path to the minimizer.
            is_maximizing_player (bool): True if Pacman is to move; False if Ghost is to move.
            map_state (np.ndarray): The map state.
        
        Returns:
            (tuple[int, Move]): The best score and corresponding move for the current player.
        """

        pacman_pos, ghost_pos = state
        
        # Key in caching table
        key = (state, is_maximizing_player)

        # If the key exists, then returns the cached score and cached move.
        if key in self.transposition_table:
            entry = self.transposition_table[key]
            if entry["depth"] >= depth:
                cached_score = entry["score"]
                cached_move = entry["move"]
                cached_flag = entry["flag"]
                
                if cached_flag == "EXACT":
                    return cached_score, cached_move
                if cached_flag == "LOWERBOUND" and cached_score >= beta:
                    return cached_score, cached_move
                if cached_flag == "UPPERBOUND" and cached_score <= alpha:
                    return cached_score, cached_move        
        
        # Base case
        if depth == 0 or pacman_pos == ghost_pos:
            return -self._manhattan_distance(pacman_pos, ghost_pos), Move.STAY
        
        # Pacman's turn (wants to minimize distance)
        if is_maximizing_player:
            best_score = float("-inf")
            best_move = Move.STAY
            flag = "EXACT"

            # Orders Pacman's possible moves by increasing Manhattan distance to the Ghost's position.
            neighbors = self._order_moves(
                self._get_neighbors(pacman_pos, map_state),
                ghost_pos,
                is_maximizing_player
            )
            
            for next_pos, move in neighbors:
                new_state = (next_pos, ghost_pos)  # Pacman moves
                score, _ = self._minimax_alpha_beta_pruning(
                    new_state, depth - 1, alpha, beta, False, map_state
                )
                
                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, score)
                if alpha >= beta:    # Pruning condition
                    flag = "LOWERBOUND" # Fail-high
                    break
        
        # Ghost's turn (want to maximize distance)
        else:
            best_score = float("inf")
            best_move = Move.STAY
            flag = "EXACT"

            # Orders Ghost's possible moves by decreasing Manhattan distance to the Pacman's position.
            neighbors = self._order_moves(
                self._get_neighbors(ghost_pos, map_state),
                pacman_pos,
                is_maximizing_player
            )
            
            for next_pos, move in neighbors:
                new_state = (pacman_pos, next_pos)
                score, _ = self._minimax_alpha_beta_pruning(
                    new_state, depth - 1, alpha, beta, True, map_state
                )

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, score)
                if beta <= alpha:           # Pruning condition
                    flag = "UPPERBOUND"     # Fail-low
                    break

        # Stored the current node's best score and best move to the transposition table
        self.transposition_table[key] = {
            "score": best_score,
            "move": best_move,
            "flag": flag,
            "depth": depth,
        }

        return best_score, best_move
                
    def _order_moves(
            self,
            moves: list[tuple[tuple[int, int], Move]],
            target_pos: tuple[int, int],
            is_maximizing_player: bool
            ) -> list[tuple[tuple[int, int], Move]]:
        """
        Orders a list of moves based on Manhattan distance to a target position.
        - Pacman (maximizing): Prioritize closer moves first.
        - Ghost (minimizing): Prioritize farther moves first.
        
        Args:
            moves (list[tuple[tuple[int, int], Move]]): A list of (position, move) pairs, 
                where each position is the result of applying the corresponding move.
            target_pos (tuple[int, int]): The reference position used to compute Manhattan distance.
            is_maximizing_player (bool): True if the current player is maximizing (Pacman); 
                False if minimizing (Ghost).

        Returns:
            (list[tuple[tuple[int, int], Move]]): The sorted list of moves based on Manhattan distance 
                to the target.
        """
        
        return sorted(
            moves,
            key=lambda move: self._manhattan_distance(move[0], target_pos),
            reverse=not is_maximizing_player    # Sort ascending for Pacman (closer first), descending for Ghost (farther first)
        )
    
    def _astar(
            self,
            start: tuple[int, int],
            goal: tuple[int, int],
            map_state: np.ndarray
            ) -> Move:
        """
        Find the optimal path from start to goal using the A-star algorithm.

        This implementation uses the Manhattan distance as the heuristic function h(n), which estimates 
        the cost from the current position to the goal. The total cost function is defined as:
            f(n) = g(n) + h(n)

        where:
            - g(n) is the actual cost from the start to the current position (path length so far),
            - h(n) is the estimated cost from the current postion to the goal.
            - f(n) is the total estimated cost of the path through the current node.
        
        The search frontier is maintained as a min-heap priority queue, where each entry is a tuple 
        containing the f-cost, a unique counter (to break ties), the current position, and the path of 
        moves taken to reach that position.

        Args:
            start (tuple[int, int]): The starting position on the map (row, column).
            goal (tuple[int, int]): The target position to reach (row, column).
            map_state (np.ndarray): The map state.

        Returns:
            Move: The first move along the optimal path. If no path is found, returns Move.STAY.
        """

        from heapq import heappush, heappop
        from itertools import count

        counter = count()   # Unique counter to distinguish each tuple in frontier
        frontier = [(0, next(counter), start, [])]
        visited = set()

        while frontier:
            f_cost, _, current_pos, move_path = heappop(frontier)
            if current_pos == goal:
                return move_path[0] if move_path else Move.STAY
            
            visited.add(current_pos)

            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    new_move_path = move_path + [move]
                    g_cost = len(new_move_path)
                    h_cost = self._a_star_heuristic(next_pos, goal)
                    f_cost = g_cost + h_cost

                    heappush(frontier, (f_cost, next(counter), next_pos, new_move_path))

        return Move.STAY
    
    def _a_star_heuristic(
            self,
            current_pos: tuple[int, int],
            target_pos: tuple[int, int]
            ) -> int:
        """
        Estimate the cost from a given position to a target using Manhattan distance.

        This heuristic is used in the A-star algorithm to guide pathfinding by approximating
        the distance between two positions on a grid.

        See also: 
            `_manhattan_distance()`: Calculate the Manhattan distance between two positions.

        Args:
            pos1 (tuple[int, int]): The position to evaluate, represented as (row, column).
            pos2 (tuple[int, int]): The target position, represented as (row, column).

        Returns:
            int: The estimated cost (heuristic value) between the two positions.
        """

        return self._manhattan_distance(current_pos, target_pos)
    
    def _compute_dead_end_map(
            self,
            map_state: np.ndarray
            ) -> tuple[np.ndarray, set[tuple[int, int]]]:
        """
        Identify all dead-end cells in the given map by using Depth-First Search (DFS) algorithm 
        and return an updated map with markings.
        
        A dead end is defined as a traversable cell (value 0) with one or fewer valid neighbors.
        All cells belonging to dead ends are marked with value 2 in the returned map.
        
        The function also returns a set containing the coordinates of all dead-end cells.

        Args:
            map_state (np.ndarray): The current map state.
        
        Returns:
            (tuple[np.ndarray, set[tuple[int, int]]]):
                - An updated map with dead-end cells marked as 2.
                - A set of coordinates representing all dead-end cells.
        """
        
        dead_end_map = map_state.copy()
        dead_end_set = set()
        stack = []
        for (i, j), _ in np.ndenumerate(dead_end_map):
            if (dead_end_map[i, j] == 0 and self._degree((i, j), dead_end_map) <= 1):
                stack.append((i, j))
        
        while stack:
            current_pos = stack.pop()
            if (self._degree(current_pos, dead_end_map) <= 1):
                dead_end_map[current_pos] = 2
                dead_end_set.add(current_pos)
            else:
                continue
            for neighbor_pos, _ in self._get_neighbors(current_pos, dead_end_map):
                stack.append(neighbor_pos)

        return dead_end_map, dead_end_set
     
    def _degree(
            self,
            pos: tuple[int, int],
            map_state: np.ndarray
            ) -> int:
        """
        Compute the degree of a given position on the map.
        
        The degree is defined as the number of valid neighboring cells (up to 4) that 
        can be reached from the current position using cardinal moves.

        Args:
            pos (tuple[int, int]): The current position, represented as (row, column).
            map_state (np.ndarray): The map state.

        Returns:
            int: The degree of the given position.
        """

        degree = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                degree += 1
        
        return degree

    def _get_opposite_move(
            self,
            move: Move
            ) -> Move:
        """
        Return the opposite direction of a given move. 
        
        For example, Move.UP becomes Move.DOWN, and Move.LEFT becomes Move.RIGHT.
        If the input is Move.STAY, the result remains Move.STAY.

        Args:
            move (Move): The input move (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY).
        
        Returns:
            Move: The move in the opposite direction.
        """

        if move == Move.UP:
            return Move.DOWN
        if move == Move.DOWN:
            return Move.UP
        if move == Move.LEFT:
            return Move.RIGHT
        if move == Move.RIGHT:
            return Move.LEFT
        return Move.STAY

    def _is_valid_position(
            self,
            pos: tuple[int, int],
            map_state: np.ndarray
            ) -> bool:
        """
        Checks whether a given position is valid (not a wall and within bounds).

        Args:
            pos (tuple[int, int]): The position to validate, represented as (row, column).
            map_state (np.ndarray): The map state, where 0 indicates a free cell.

        Returns:
            bool: True if the position is within bounds and not a wall; otherwise, False.
        """
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
    
    def _apply_move(
            self, 
            pos: tuple[int, int],
            move: Move
            ) -> tuple[int, int]:
        """
        Applies a move to the specified position and returns the resulting position.

        Args:
            pos (tuple[int, int]): The current position on the board.
            move (Move): The move to apply.
        
        Returns:
            (tuple[int, int]): The updated position after the move is applied.
        """
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(
            self,
            pos: tuple[int, int],
            map_state: np.ndarray
            ) -> list[tuple[tuple[int, int], Move]]:
        """
        Find all valid neighboring cells from a given position, along with the moves required to reach them.

        Args:
            pos (tuple[int, int]): The current position, represented as (row, column).
            map_state (np.ndarray): The map state.

        Returns:
            (list[tuple[tuple[int, int], Move]]): A list of tuples, each containing a valid neighboring position
            and the corresponding move to reach it.
        """
        
        neighbors = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        
        return neighbors
    
    def _manhattan_distance(
            self,
            pos1: tuple[int, int],
            pos2: tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance between two grid positions.

        Args:
            pos1 (tuple[int, int]): The first position, represented as (row, column).
            pos2 (tuple[int, int]): The second position, represented as (row, column).

        Returns:
            int: The Manhattan distance between the two positions.
        """
        
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        """
        Initialize the Ghost agent.
        """
        self.name = "Minimax - Alpha Beta Pruning & Dead-end evasion"
        self.transposition_table = {}           # Cache for previously evaluated states in Minimax (transposition table)
        self.dead_end_set = set()               # Set of positions identified as dead ends on the map
        self.dead_end_map = None                # Map state with dead-end cells marked (e.g., value 2)
        self.closest_exit_pos = None            # Nearest exit position if Ghost starts in a dead end
        self.is_dead_end_calculated = False     # Indicates whether dead-end positions have been precomputed
        self.MAX_PREDICTED_DEPTH = 10           # Max depth for Ghost's Minimax prediction
    
    def step(
            self,
            map_state: np.ndarray,
            my_position: tuple[int, int],
            enemy_position: tuple[int, int],
            step_number: int
            ) -> Move:
        """
        Determine Ghost's next move based on the current game state.

        This agent uses Minimax with Alpha-Beta Pruning as its main strategy to avoid Pacman.
        On the first step, Ghost pre-computes all dead-end positions on the map for future use.

        If Ghost is initialized in a dead-end, it attempts to find the nearest exit using DFS,
        and then navigates to that exit using BFS. If no exit is found, Ghost falls back to Minimax.

        In all other cases, Ghost uses Minimax to avoid Pacman while leveraging the dead-end map
        to steer clear of trapped positions.

        Args:
            map_state (np.ndarray): The current map layout.
            my_position (tuple[int, int]): Ghost's current position (row, column).
            enemy_position (tuple[int, int]): Pacman's current position (row, column).
            step_number (int): The current step number (maximum 200).

        Returns:
            Move: The next move Ghost should take. Returns Move.STAY if no valid path is found.
        """

        # Compute and store dead-end positions on the map (only once)
        if not self.is_dead_end_calculated:
            self.is_dead_end_calculated = True
            self.dead_end_map, self.dead_end_set = self._compute_dead_end_map(map_state)
        
        # Use BFS to reach the nearest exit if available
        # Otherwise, use Minimax with the current map state instead
        if my_position in self.dead_end_set:
            if not self.closest_exit_pos:
                self.closest_exit_pos = self._find_exit_from_dead_end(my_position, map_state)
            if self.closest_exit_pos:
                return self.bfs(my_position, self.closest_exit_pos, map_state)
            else:
                return self.minimax_alpha_beta_pruning((my_position, enemy_position), 10, float("-inf"), float("inf"), False, map_state)[1]

        # Use Minimax with the dead end map state to avoid all dead end cells
        _, best_move = self.minimax_alpha_beta_pruning((my_position, enemy_position), 10, float("-inf"), float("inf"), False, self.dead_end_map)

        return best_move
        
    def minimax_alpha_beta_pruning(
            self,
            state: tuple[tuple[int, int], tuple[int, int]],
            depth: int,
            alpha: float | int,
            beta: float | int,
            is_maximizing_player: bool,
            map_state: np.ndarray
            ) -> tuple[int, Move]:
        """
        Predicts the ghost's next move using the Minimax algorithm with Alpha-Beta Pruning.

        This function evaluates game states recursively to determine the optimal move for 
        either Pacman (maximizing player) or Ghost (minimizing player). It uses a transposition 
        table to cache previously evaluated states (only states that have shallow depth) and avoid 
        redundant computation.

        Evaluation stops when the search depth reaches 0 or when Pacman catches the Ghost (both occupy 
        the same cell). In such cases, the score is set to the negative Manhattan distance between their 
        positions.

        Alpha-Beta pruning is applied to eliminate branches that cannot influence the final decision:
            - Pacman updates alpha and prunes when alpha >= beta (fail-high).
            - Ghost updates beta and prunes when beta <= alpha (fail-low).
        
        Cached entries in the transposition table are reused based on depth and pruning flags:
            - "EXACT": score is precise.
            - "LOWERBOUND": score is at least the cached value.
            - "UPPERBOUND": score is at most the cached value.
        
        Args:
            state (tuple[tuple[int, int], tuple[int, int]]): A tuple of (pacman_pos, ghost_pos), each as (row, column).
            depth (int): The remaining search depth.
            alpha (float | int): The best already explored option along the path to the maximizer.
            beta (float | int): The best already explored option along the path to the minimizer.
            is_maximizing_player (bool): True if Pacman is to move; False if Ghost is to move.
            map_state (np.ndarray): The map state.
        
        Returns:
            (tuple[int, Move]): The best score and corresponding move for the current player.
        """

        ghost_pos, pacman_pos = state

        # Key in caching table
        key = (state, is_maximizing_player)

        # If the key exists, then returns the cached score and cached move.
        if key in self.transposition_table:
            entry = self.transposition_table[key]
            if entry["depth"] >= depth:
                cached_score = entry["score"]
                cached_move = entry["move"]
                cached_flag = entry["flag"]
                
                if cached_flag == "EXACT":
                    return cached_score, cached_move
                if cached_flag == "LOWERBOUND" and cached_score >= beta:
                    return cached_score, cached_move
                if cached_flag == "UPPERBOUND" and cached_score <= alpha:
                    return cached_score, cached_move
        
        # Base case
        if depth == 0 or ghost_pos == pacman_pos:
            return -self._manhattan_distance(ghost_pos, pacman_pos), Move.STAY
        
        # Pacman's turn (wants to minimize distance)
        if is_maximizing_player:
            best_score = float("-inf")
            best_move = Move.STAY
            flag = "EXACT"

            # Orders Pacman's possible moves by increasing Manhattan distance to the Ghost's position.
            neighbors = self._order_moves(
                self._get_neighbors(pacman_pos, map_state),
                ghost_pos,
                is_maximizing_player
            )

            for next_pos, move in neighbors:
                new_state = (ghost_pos, next_pos)   # Pacman moves
                score, _ = self.minimax_alpha_beta_pruning(
                    new_state, depth - 1, alpha, beta, False, map_state
                )

                if score > best_score:
                    best_score = score
                    best_move = move

                alpha = max(alpha, score)
                if alpha >= beta:    # Pruning condition
                    flag = "LOWERBOUND" # Fail-high
                    break

        # Ghost's turn (want to maximize distance)
        else:
            best_score = float("inf")
            best_move = Move.STAY
            flag = "EXACT"

            # Orders Ghost's possible moves by decreasing Manhattan distance to the Pacman's position.
            neighbors = self._order_moves(
                self._get_neighbors(ghost_pos, map_state),
                pacman_pos,
                is_maximizing_player
            )
            
            for next_pos, move in neighbors:
                new_state = (next_pos, pacman_pos)
                score, _ = self.minimax_alpha_beta_pruning(
                    new_state, depth - 1, alpha, beta, True, map_state
                )

                if score < best_score:
                    best_score = score
                    best_move = move

                beta = min(beta, score)
                if beta <= alpha:           # Pruning condition
                    flag = "UPPERBOUND"     # Fail-low
                    break
        
        # Stored the current node's best score and best move to the transposition table
        self.transposition_table[key] = {
            "score": best_score,
            "move": best_move,
            "flag": flag,
            "depth": depth,
        }

        return best_score, best_move
                
    def bfs(
            self,
            start: tuple[int, int],
            goal: tuple[int, int],
            map_state: np.ndarray
            ) -> Move:
        """
        Find the shortest path from start to goal using Breadth-First Search (BFS).

        This function performs BFS on a grid-based map to find the shortest path between two positions.
        It returns the first move Ghost should take to follow that path.

        Args:
            start (tuple[int, int]): The starting position (row, column).
            goal (tuple[int, int]): The target position (row, column).
            map_state (np.ndarray): The current map layout.

        Returns:
            Move: The first move along the shortest path to the goal.
                Returns Move.STAY if no path is found.
        """
        
        from collections import deque
        queue = deque([start])
        parent = {start: None}
        move_actions = {start: Move.STAY}

        while queue:
            current_pos = queue.popleft()
            if current_pos == goal:
                path = []
                while current_pos is not None:
                    path.append(move_actions[current_pos])
                    current_pos = parent[current_pos]
                return path[-2] if (len(path) > 1) else path[-1]

            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in parent:
                    parent[next_pos] = current_pos
                    move_actions[next_pos] = move
                    queue.append(next_pos)
                    
        return Move.STAY

    def _compute_dead_end_map(
            self,
            map_state: np.ndarray
            ) -> tuple[np.ndarray, set[tuple[int, int]]]:
        """
        Identify all dead-end cells in the given map by using Depth-First Search (DFS) algorithm 
        and return an updated map with markings.
        
        A dead end is defined as a traversable cell (value 0) with one or fewer valid neighbors.
        All cells belonging to dead ends are marked with value 2 in the returned map.
        
        The function also returns a set containing the coordinates of all dead-end cells.

        Args:
            map_state (np.ndarray): The current map state.
        
        Returns:
            (tuple[np.ndarray, set[tuple[int, int]]]):
                - An updated map with dead-end cells marked as 2.
                - A set of coordinates representing all dead-end cells.
        """
        
        dead_end_map = map_state.copy()
        dead_end_set = set()
        stack = []
        for (i, j), _ in np.ndenumerate(dead_end_map):
            if (dead_end_map[i, j] == 0 and self._degree((i, j), dead_end_map) <= 1):
                stack.append((i, j))
        
        while stack:
            current_pos = stack.pop()
            if (self._degree(current_pos, dead_end_map) <= 1):
                dead_end_map[current_pos] = 2
                dead_end_set.add(current_pos)
            else:
                continue
            for neighbor_pos, _ in self._get_neighbors(current_pos, dead_end_map):
                stack.append(neighbor_pos)

        return dead_end_map, dead_end_set

    def _find_exit_from_dead_end(
            self,
            start: tuple[int, int],
            map_state: np.ndarray
            ) -> tuple[int, int]:
        """
        Find the nearest exit from a dead-end area using Depth-First Search (DFS).

        Starting from the given position (typically where the Ghost is located), this function explores 
        connected cells until it finds a position with at least two accessible neighbors outside the dead-end set.
        Such a position is considered an "exit" from the dead-end.

        Args:
            start (tuple[int, int]): The starting position of the Ghost (row, column).
            map_state (np.ndarray): The current map state.

        Returns:
            (tuple[int, int] | None): The nearest exit position outside the dead-end area, or None if no exit is found.
        """
        
        visited = set()
        stack = [start]

        while stack:
            current_pos = stack.pop()
            if current_pos in visited:
                continue
            visited.add(current_pos)
            free = 0
            for next_pos, _ in self._get_neighbors(current_pos, map_state):
                if next_pos not in self.dead_end_set:
                    free += 1
                if next_pos not in visited:
                    stack.append(next_pos)
            
            if free >= 2:
                return current_pos
        
        return None

    def _is_valid_position(
            self,
            pos: tuple[int, int],
            map_state: np.ndarray
            ) -> bool:
        """
        Checks whether a given position is valid (not a wall and within bounds).

        Args:
            pos (tuple[int, int]): The position to validate, represented as (row, column).
            map_state (np.ndarray): The map state, where 0 indicates a free cell.

        Returns:
            bool: True if the position is within bounds and not a wall; otherwise, False.
        """
        
        row, col = pos
        height, width = map_state.shape

        if (row < 0 or row >= height or col < 0 or col >= width):
            return False
    
        return map_state[row, col] == 0
    
    def _order_moves(
            self,
            moves: list[tuple[tuple[int, int], Move]],
            target_pos: tuple[int, int],
            is_maximizing_player: bool
            ) -> list[tuple[tuple[int, int], Move]]:
        """
        Orders a list of moves based on Manhattan distance to a target position.
        - Pacman (maximizing): Prioritize closer moves first.
        - Ghost (minimizing): Prioritize farther moves first.
        
        Args:
            moves (list[tuple[tuple[int, int], Move]]): A list of (position, move) pairs, 
                where each position is the result of applying the corresponding move.
            target_pos (tuple[int, int]): The reference position used to compute Manhattan distance.
            is_maximizing_player (bool): True if the current player is maximizing (Pacman); 
                False if minimizing (Ghost).

        Returns:
            (list[tuple[tuple[int, int], Move]]): The sorted list of moves based on Manhattan distance 
                to the target.
        """
        
        return sorted(
            moves,
            key=lambda move: self._manhattan_distance(move[0], target_pos),
            reverse=not is_maximizing_player    # Sort ascending for Pacman (closer first), descending for Ghost (farther first)
        )

    def _degree(
            self,
            pos: tuple[int, int],
            map_state: np.ndarray) -> int:
        """
        Compute the degree of a given position on the map.
        
        The degree is defined as the number of valid neighboring cells (up to 4) that 
        can be reached from the current position using cardinal moves.

        Args:
            pos (tuple[int, int]): The current position, represented as (row, column).
            map_state (np.ndarray): The map state.

        Returns:
            int: The degree of the given position.
        """

        degree = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                degree += 1
        
        return degree
    
    def _apply_move(
            self,
            pos: tuple[int, int],
            move: Move
            ) -> tuple[int, int]:
        """
        Applies a move to the specified position and returns the resulting position.

        Args:
            pos (tuple[int, int]): The current position on the board.
            move (Move): The move to apply.
        
        Returns:
            (tuple[int, int]): The updated position after the move is applied.
        """

        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(
            self,
            pos: tuple[int, int],
            map_state: np.ndarray
            ) -> list[tuple[tuple[int, int], Move]]:
        """
        Find all valid neighboring cells from a given position, along with the moves required to reach them.

        Args:
            pos (tuple[int, int]): The current position, represented as (row, column).
            map_state (np.ndarray): The map state.

        Returns:
            (list[tuple[tuple[int, int], Move]]): A list of tuples, each containing a valid neighboring position
            and the corresponding move to reach it.
        """
        
        neighbors = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        
        return neighbors
    
    def _manhattan_distance(
            self,
            pos1: tuple[int, int],
            pos2: tuple[int, int]
            ) -> int:
        """
        Calculate the Manhattan distance between two grid positions.

        Args:
            pos1 (tuple[int, int]): The first position, represented as (row, column).
            pos2 (tuple[int, int]): The second position, represented as (row, column).

        Returns:
            int: The Manhattan distance between the two positions.
        """

        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
