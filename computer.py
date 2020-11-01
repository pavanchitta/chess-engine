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
        print("EVALUATING BOARD")
        if board.state.is_game_over():
            if (board.color_won(self.color)):
                return float("inf")
            return float("-inf")

        score = 0
        piece_value_dict = {'p': 10, 'n': 30, 'b': 30, 'r': 50, 'q': 90, 'k': 900}
        pieces_map = board.state.piece_map()
        for piece in pieces_map.values():
            piece_value = piece_value_dict[piece.symbol().lower()] * (1 if piece.color == self.color else -1)
            score += piece_value
        print("BOARD SCORE: ", score)
        return score

    def generate_move(self, board):
        """
        Use a minimax algorithm in combination with the value function to
        generate the next move for the computer. Return this move
        """
        def minimax(board, depth, alpha, beta, cpu_turn):

            if depth == 0 or board.state.is_game_over():
                return self.value_board(board)

            if cpu_turn:
                # Want to maximize value
                max_score = float("-inf")
                best_move = list(board.state.legal_moves)[0]
                for move in board.state.legal_moves:
                    board.state.push(move)
                    score = minimax(board, depth-1, alpha, beta, not cpu_turn)
                    if score > max_score:
                        max_score = score
                        best_move = move
                    board.state.pop()
                    alpha = max(max_score, alpha)
                    if (alpha >= beta):
                        break

                # Need to return the best move as well if in the outer most depth
                return max_score if depth != self.depth else (max_score, best_move)

            else:
                # Assume optimal play by opponent to minimize value
                min_score = float("inf")
                for move in board.state.legal_moves:
                    board.state.push(move)
                    min_score = min(min_score, minimax(board, depth-1, alpha, beta, not cpu_turn))
                    board.state.pop()
                    beta = min(min_score, beta)
                    if (beta <= alpha):
                        break
                return min_score

        if board.state.is_game_over():
            return chess.Move.null()
        max_score, best_move = minimax(board, self.depth, float("-inf"), float("inf"), True)
        print("Best Computer Move score: ", max_score)
        assert(best_move != None)
        return best_move
