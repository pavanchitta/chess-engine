####
# Script to process data
####

import numpy as np
import chess.pgn

PGN_FILEPATH = '/Users/pavanchitta/Downloads/ficsgamesdb_202001_standard2000_nomovetimes_163804.pgn'


def process_result(res):
    if res.split('-')[0] == '1':
        return 1
    elif res.split('-')[1] == '1':
        return 0
    return 0.5


def load_data(pgn_file_path=PGN_FILEPATH, ngames=-1):
    """
    Load the data and process it to feed to ML models
    Args:
    pgn_file_path: Path to the pgn file containing all the games
    ngames: Limit on number of games to load data for
    """
    X = []
    y = []
    sample_weights = []
    pgn = open(pgn_file_path)
    game = chess.pgn.read_game(pgn)
    n = 1
    delta = 0.99  # discount factor
    while game and (ngames == -1 or n <= ngames):

        board = game.board()
        result = process_result(game.headers['Result'])
        num_moves = len(list(game.mainline_moves()))
        for idx, move in enumerate(game.mainline_moves()):
            dist = num_moves - idx
            weight = delta ** dist
            board.push(move)
            X.append(process_board_channels(board))
            y.append(result)
            sample_weights.append(weight)

        game = chess.pgn.read_game(pgn)
        n += 1
    return np.array(X), np.array(y), np.array(sample_weights)

def process_board_channels(board):
    """
    Process a python-chess Board object into a 3D 8x8x6 array representation
    """
    X = np.zeros((8, 8, 6))
    piece_dict = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
    pieces_map = board.piece_map()
    for square, piece in pieces_map.items():
        piece_channel = piece_dict[piece.symbol().lower()] 
        row = square // 8
        col = square % 8
        X[row, col, piece_channel] = 10 if piece.color == chess.WHITE else -10
    return X


def process_board(board):
    """
    Process a python-chess Board object into a 2D 8x8 array representation
    """
    X = np.zeros((8, 8))
    piece_dict = {'p': 1, 'n': 2, 'b': 3, 'r': 5, 'q': 9, 'k': 90}
    pieces_map = board.piece_map()
    for square, piece in pieces_map.items():
        piece_value = piece_dict[piece.symbol().lower()] * (1 if piece.color == chess.WHITE else -1)
        row = square // 8
        col = square % 8
        X[row, col] = piece_value
    return X
