import chess


class Computer:

    def __init__(self, color, minmax_depth=4):
        self.color = color
        self.depth = minmax_depth

    def value_board(board):
        """
        Compute the value of a board state for the computer. This will use
        heuristics for piece value and board layout. Ultimately, want to
        incorporate a learned value function as well.
        Class method.
        """
        return 0

    def generate_move(self, board):
        """
        Use a minimax algorithm in combination with the value function to
        generate the next move for the computer. Return this move
        """
        return list(board.state.legal_moves)[0]
