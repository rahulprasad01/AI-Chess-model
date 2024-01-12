from __future__ import annotations
from typing import List, Tuple

class State:
    def __init__(self, board: Board, turn: str, moves_count: int):
        self.board = board
        self.turn = turn
        self.moves_count = moves_count

class Board:
    def __init__(self, initial_grid=None):
        self.rows, self.cols = 8, 8
        if initial_grid is None:
            self.grid = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        else:
            self.grid = initial_grid

    def display(self):
        for row in self.grid:
            print([str(piece) if piece is not None else '.' for piece in row])


class Piece:
    def __init__(self, name: str, color: str, position: Tuple[int, int], board: Board):
        self.name = name
        self.color = color
        self.position = position
        self.board= board

    def __str__(self):
        return f"{self.color[0].upper()}{self.name[0]}"

    def is_valid_move(self, new_position: Tuple[int, int], capturing: bool = False) -> bool:
        
        raise NotImplementedError("Subclasses must implement this method")

    def generate_legal_moves(self, capturing: bool = False) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        legal_moves = []
        for i in range(8):
            for j in range(8):
                new_position = (i, j)
                if self.is_valid_move(new_position, capturing):
                    legal_moves.append((self.position, new_position))
        return legal_moves


class King(Piece):
    def __init__(self, color: str, position: Tuple[int, int], board: Board):
        super().__init__('King', color, position, board)

    def is_valid_move(self, new_position: Tuple[int, int], capturing: bool = False) -> bool:
        
        return (
            0 <= new_position[0] < 8
            and 0 <= new_position[1] < 8
            and abs(new_position[0] - self.position[0]) <= 1
            and abs(new_position[1] - self.position[1]) <= 1
        )

class Rook(Piece):
    def __init__(self, color: str, position: Tuple[int, int], board: Board):
        super().__init__('Rook', color, position, board)

    def is_valid_move(self, new_position: Tuple[int, int], capturing: bool = False) -> bool:
        
        return (
            new_position[0] == self.position[0] and self.is_clear_rank(new_position) and new_position[1] != self.position[1]
            or new_position[1] == self.position[1] and self.is_clear_file(new_position) and new_position[1] == self.position[1]
        )

    def is_clear_rank(self, new_position: Tuple[int, int]) -> bool:
        
        start, end = min(self.position[1], new_position[1]), max(self.position[1], new_position[1])
        return all(self.board[self.position[0]][col] is None for col in range(start + 1, end))

    def is_clear_file(self, new_position: Tuple[int, int]) -> bool:
        
        start, end = min(self.position[0], new_position[0]), max(self.position[0], new_position[0])
        return all(self.board[row][self.position[1]] is None for row in range(start + 1, end))

class Bishop(Piece):
    def __init__(self, color: str, position: Tuple[int, int], board: Board):
        super().__init__('Bishop', color, position, board)

    def is_valid_move(self, new_position: Tuple[int, int], capturing: bool = False) -> bool:
        
        return (
            abs(new_position[0] - self.position[0]) == abs(new_position[1] - self.position[1])
            and self.is_clear_diagonal(new_position)
        )

    def is_clear_diagonal(self, new_position: Tuple[int, int]) -> bool:
        
        row, col = self.position
        row_dir, col_dir = (1, 1) if new_position[0] > row else (-1, 1)
        while (row, col) != new_position:
            row += row_dir
            col += col_dir
            if 0 <= row < 8 and 0 <= col < 8:
                if self.board[row][col] is not None:
                 return False
            else:
                return False    
        return True


class Knight(Piece):
    def __init__(self, color: str, position: Tuple[int, int], board: Board):
        super().__init__('Knight', color, position, board)

    def is_valid_move(self, new_position: Tuple[int, int], capturing: bool = False) -> bool:
        
        return (
            abs(new_position[0] - self.position[0]) == 2 and abs(new_position[1] - self.position[1]) == 1
        ) or (
            abs(new_position[0] - self.position[0]) == 1 and abs(new_position[1] - self.position[1]) == 2
        )

class Squire(Piece):
    def __init__(self, color: str, position: Tuple[int, int], board: Board):
        super().__init__('Squire', color, position, board)

    def is_valid_move(self, new_position: Tuple[int, int], capturing: bool = False) -> bool:
        
        return (
            0 <= new_position[0] < 8
            and 0 <= new_position[1] < 8
            and abs(new_position[0] - self.position[0]) + abs(new_position[1] - self.position[1]) == 2
        )

class Combatant(Piece):
    def __init__(self, color: str, position: Tuple[int, int], board: Board):
        super().__init__('Combatant', color, position, board)

    def is_valid_move(self, new_position: Tuple[int, int], capturing: bool = False) -> bool:
        
        capturing = self.board[new_position[0]][new_position[1]] is not None and \
                self.board[new_position[0]][new_position[1]].color != self.color
        return (
            0 <= new_position[0] < 8
            and 0 <= new_position[1] < 8
            and (
                (capturing and abs(new_position[0] - self.position[0]) == 1 and abs(new_position[1] - self.position[1]) == 1)
                or (not capturing and (
                    new_position[0] == self.position[0] and abs(new_position[1] - self.position[1]) == 1
                    or new_position[1] == self.position[1] and abs(new_position[0] - self.position[0]) == 1
                ))
            )
        )


def evaluate_board(state: State, turn: str) -> int:
    piece_values = {
        "King": 100,
        "Rook": 5,
        "Bishop": 3,
        "Knight": 3,
        "Squire": 2,
        "Combatant": 1
    }

    center_row, center_col = state.board.rows // 2, state.board.cols // 2
    total_score = 0

    opponent_king_position = None

    
    for row in range(state.board.rows):
        for col in range(state.board.cols):
            piece = state.board.grid[row][col]
            if isinstance(piece, King) and piece.color != turn:
                opponent_king_position = (row, col)
                break

    
    if is_winning_state(state.board, turn):
        return 1000 if turn == 'white' else -1000

    if is_winning_state(state.board, 'black' if turn == 'white' else 'white'):
        return -1000 if turn == 'white' else 1000

    for row in range(state.board.rows):
        for col in range(state.board.cols):
            piece = state.board.grid[row][col]
            if piece is not None:
                if isinstance(piece, Piece):
                
                    color_multiplier = 1 if piece.color == 'white' else -1
                    piece_value = piece_values.get(piece.name, 0)

                
                    distance_to_opponent_king = (
                        abs(row - opponent_king_position[0]) + abs(col - opponent_king_position[1])
                    )

                    total_score += (
                        color_multiplier * piece_value - 0.1 * distance_to_opponent_king
                    )

    return total_score



def is_winning_state(board: Board, turn: str) -> bool:
    opponent_turn = 'black' if turn == 'white' else 'white'
    for row in board.grid:
        for piece in row:
            if isinstance(piece, King) and piece.color == opponent_turn:
                return False  
    return True  

def is_draw_state(board: Board, moves_count: int) -> bool:
    
    only_kings_remaining = all(isinstance(piece, King) for row in board.grid for piece in row)
    max_turns_reached = moves_count >= 50
    return only_kings_remaining or max_turns_reached


def is_terminal_node(state: State) -> bool:
    
    if is_winning_state(state.board, state.turn):
        return True

    if is_draw_state(state.board, state.moves_count):
        return True
    
    return False

def get_legal_moves(state: State) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    legal_moves = []
    for row in range(len(state.board.grid)):
        for col in range(len(state.board.grid[row])):
            piece = state.board.grid[row][col]
            if piece != '':
                if piece is not None and piece.color == state.turn:
                    piece_legal_moves = piece.generate_legal_moves()
                    legal_moves.extend(piece_legal_moves)
                
                        
    return legal_moves


def make_move(state: State, move: Tuple[Tuple[int, int], Tuple[int, int]]) -> State:
    
    new_board = Board()
    for row in range(len(state.board.grid)):
        for col in range(len(state.board.grid[row])):
            new_board.grid[row][col] = state.board.grid[row][col]

    start_position, end_position = move

    start_row, start_col = start_position
    piece = new_board.grid[start_row][start_col]

    
    end_row, end_col = end_position
    new_board.grid[start_row][start_col] = ''
    new_board.grid[end_row][end_col] = piece

    new_state = State(new_board, 'black' if state.turn == 'white' else 'white', state.moves_count + 1)

    return new_state

def initialize_board(gameboard: List[Tuple[str, str, Tuple[int, int]]]) -> Board:
    
    grid = [[None for _ in range(8)] for _ in range(8)]

    for piece_info in gameboard:
        piece_name, piece_color, piece_position = piece_info
        row, col = piece_position
        piece = None

        if piece_name == "King":
            piece = King(piece_color, piece_position, grid)
        elif piece_name == "Rook":
            piece = Rook(piece_color, piece_position, grid)
        elif piece_name == "Bishop":
            piece = Bishop(piece_color, piece_position, grid)
        elif piece_name == "Knight":
            piece = Knight(piece_color, piece_position, grid)
        elif piece_name == "Squire":
            piece = Squire(piece_color, piece_position, grid)
        elif piece_name == "Combatant":
            piece = Combatant(piece_color, piece_position, grid)
        # Add other piece types as needed

        # Place the piece on the board
        grid[row][col] = piece

    return Board(grid)


def minimax(state: State, depth: int, alpha: float, beta: float, maximizing_player: bool) -> Tuple[int, Tuple[int, int]]:
    if depth == 0 or is_terminal_node(state):
        return evaluate_board(state, state.turn), None

    legal_moves = get_legal_moves(state)

    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            child_state = make_move(state, move)
            eval, _ = minimax(child_state, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            child_state = make_move(state, move)
            eval, _ = minimax(child_state, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  
        return min_eval, best_move



def studentAgent(gameboard: List[Tuple[str, str, Tuple[int, int]]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    
    internal_board = initialize_board(gameboard)

    current_state = State(internal_board, 'white', 0)  

    max_depth = 3  
    best_move = None

    for depth in range(1, max_depth + 1):
        _, move = minimax_with_depth_limit(current_state, depth)
        best_move = move

    return best_move

def minimax_with_depth_limit(state: State, depth: int) -> Tuple[int, Tuple[int, int]]:
    
    best_move = None

    _, best_move = minimax(state, depth-1, float('-inf'), float('inf'), True)

    return evaluate_board(state, state.turn), best_move



if __name__ == "__main__":
    # Creating a board
    chessboard = Board()

    public_1 = [("King", 'white', (7,7)),
            ("King", 'black', (0,0)),
            ("Rook", 'white', (6,1)),
            ("Rook", 'white', (5,1)),
            ("Rook", 'black', (6,5)),
            ("Rook", 'black', (6,6))]
    
    path = studentAgent(public_1)
    print(path)

    public_2 = [("King", 'white', (2,3)),
            ("King", 'black', (0,4)),
            ("Combatant", 'white', (1,4)),
            ("Combatant", 'white', (2,5)),
            ("Combatant", 'black', (7,0))]
    
    path = studentAgent(public_2)
    print(path)

    public_3 = [("King", 'white', (3,4)),
            ("King", 'black', (3,2)),
            ("Rook", "white", (2,4)),
            ("Rook", "white", (1,1)),
            ("Combatant", "white", (2,0)),
            ("Squire", "white", (1,0)),
            ("Rook", "black", (4,2)),
            ("Bishop", "black", (7,7)),
            ("Knight", "black", (2,2))]
    
    path = studentAgent(public_3)
    print(path)

    public_4 = [('King', 'white', (0, 5)), 
            ('King', 'black', (7, 1)), 
            ('Rook', 'white', (7, 7)), 
            ('Squire', 'white', (5, 0)), 
            ('Knight', 'white', (6, 4)), 
            ('Squire', 'black', (7, 2)), 
            ('Bishop', 'black', (6, 3)), 
            ('Rook', 'black', (1, 7)), 
            ('Combatant', 'black', (6, 0)), 
            ('Combatant', 'black', (5, 1)), 
            ('Combatant', 'black', (6, 2))]
    
    path = studentAgent(public_4)
    print(path)

    