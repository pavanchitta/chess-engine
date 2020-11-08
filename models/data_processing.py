####
# Script to process data
####
import os
import numpy as np
import chess.pgn
import time
import sys
from tqdm import tqdm

import git
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.append(f'{homedir}')

PGN_FILEPATH = '/Users/pavanchitta/Downloads/chess-data/all_ratings'

CHUNK_SIZE = 100000  # To prevent memory overflow, store files in CHUNK_SIZE


def process_result(res):
    if res.split('-')[0] == '1':
        return 1
    elif res.split('-')[1] == '1':
        return 0
    return 0.5


def load_data(pgn_file_path=PGN_FILEPATH, ngames=-1, use_cache=False, name='', use_chunks=False):
    """
    Load the data and process it to feed to ML models
    Args:
    pgn_file_path: Path to the pgn file containing all the games
    ngames: Limit on number of games to load data for
    """
    if use_cache:
        cached_data_files = list(os.listdir(f'{homedir}/data'))
        if (f'data{name}_{int(ngames)}_games.npz' in cached_data_files):
            print('===== Loading data from cache ======')
            data = np.load(f'{homedir}/data/data{name}_{int(ngames)}_games.npz')
            X = data['X']
            y = data['y']
            sample_weights = data['sample_weights']
            return X, y, sample_weights
    X = []
    y = []
    sample_weights = []
    n = 1
    delta = 1  # discount factor
    move_thresh = 40  # Only consider board positions within move_thresh (1-side) of the game end
    chunk_idx = 0
    print(f"==== Loading {ngames} games worth of Data ======")

    with tqdm(total=100, file=sys.stdout) as pbar:

        for pgn_file in os.listdir(pgn_file_path):
            # print(pgn_file)
            pgn = open(os.path.join(pgn_file_path, pgn_file))
            try:
                game = chess.pgn.read_game(pgn)
            except Exception as e:
                print(e)
                continue
            board = None
            while game and (ngames == -1 or n <= ngames):
                # if board:
                #     print(board)
                board = game.board()
                result = process_result(game.headers['Result'])
                if result == 0.5:
                    game = chess.pgn.read_game(pgn)
                    continue
                num_moves = len(list(game.mainline_moves()))
                for idx, move in enumerate(game.mainline_moves()):
                    dist = (num_moves - idx) / 2
                    weight = delta ** dist
                    board.push(move)
                    # print(board)
                    if (dist > 40):
                        continue
                    X.append(process_board_channels_w_turn(board))
                    # print(X[-1][:, :, 0])
                    # return
                    y.append(result)
                    sample_weights.append(weight)

                game = chess.pgn.read_game(pgn)
                n += 1
                if (n % 10 == 0):
                    pbar.update(10/ngames * 100)

            if (n % int(1e5) == 0) and use_chunks:
                chunk_idx = n // int(1e5)
                np.savez_compressed(open(f"{homedir}/data/data{name}_{int(ngames)}_thresh{move_thresh}_games_{chunk_idx}.npz", 'wb'),
                         X=X, y=y, sample_weights=sample_weights)
                X = []
                y = []
                sample_weights = []
            if (ngames != -1 and n > ngames):
                break

    # Save files for future use
    if len(X) > 0:
        np.savez_compressed(open(f"{homedir}/data/data{name}_{int(ngames)}_thresh{move_thresh}_games_{chunk_idx+1}.npz", 'wb'),
                 X=X, y=y, sample_weights=sample_weights)

    return np.array(X), np.array(y), np.array(sample_weights)


def process_board_channels_w_turn(board):
    """
    Process a python-chess Board object into a 3D 8x8x6 array representation
    """
    X = np.zeros((8, 8, 7))
    piece_dict = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
    pieces_map = board.piece_map()
    turn = 1 if board.turn == chess.WHITE else -1
    for square, piece in pieces_map.items():
        piece_channel = piece_dict[piece.symbol().lower()] 
        row = square // 8
        col = square % 8
        X[row, col, piece_channel] = 1 if piece.color == chess.WHITE else -1
    X[:, :, 6] = turn
    return X


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
    Process a python-chess Board object into a 2D 8x8x1 array representation
    """
    X = np.zeros((8, 8))
    piece_dict = {'p': 1, 'n': 2, 'b': 3, 'r': 5, 'q': 9, 'k': 90}
    pieces_map = board.piece_map()
    for square, piece in pieces_map.items():
        piece_value = piece_dict[piece.symbol().lower()] * (1 if piece.color == chess.WHITE else -1)
        row = square // 8
        col = square % 8
        X[row, col] = piece_value
    return np.expand_dims(X, axis=-1)
