import chess


class Board:
    """
    This will wrap a python-chess board to provide a represenation for the board.
    For now, this is very minimal and only functions to contain state, which can
    be fetched and operated on directly.
    """
    def __init__(self, state=None):
        self.state = state if state else chess.Board()
