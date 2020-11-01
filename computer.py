import chess


class Computer:

    def __init__(self, color, minmax_depth=3):
        self.color = color
        self.depth = minmax_depth

    def value_board(self, board):
        """
        Compute the value of a board state for the computer. This will use
        heuristics for piece value and board layout. Ultimately, want to
        incorporate a learned value function as well.
        """
        if board.state.is_game_over():
            if (board.color_won(self.color)):
                return float("inf")
            return float("-inf")

        score = 0
        piece_value_dict = {'p': 10, 'n': 30, 'b': 30, 'r': 50, 'q': 90, 'k': 900}
        pieces_map = board.state.piece_map()
        for piece in pieces_map.values():
            piece_value = piece_value_dict[piece.symbol().lower()] * (piece.color == self.color)
            score += piece_value
        return score

    def generate_move(self, board):
        """
        Use a minimax algorithm in combination with the value function to
        generate the next move for the computer. Return this move
        """
        def minimax(board, depth, cpu_turn):

            if depth == 0 or board.state.is_game_over():
                return self.value_board(board)

            if cpu_turn:
                # Want to maximize value
                max_score = float("-inf")
                best_move = None
                for move in board.state.legal_moves:
                    if not move:
                        continue
                    board.state.push(move)
                    score = minimax(board, depth-1, not cpu_turn)
                    if score > max_score:
                        max_score = score
                        best_move = move
                    board.state.pop()
                # Need to return the best move as well if in the outer most depth
                return max_score if depth != self.depth else (max_score, best_move)

            else:
                # Assume optimal play by opponent to minimize value
                min_score = float("inf")
                for move in board.state.legal_moves:
                    if not move:
                        continue
                    board.state.push(move)
                    min_score = min(min_score, minimax(board, depth-1, not cpu_turn))
                    board.state.pop()
                return min_score

        max_score, best_move = minimax(board, self.depth, True)
        return best_move
