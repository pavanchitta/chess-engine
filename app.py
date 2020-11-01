# app.py
import chess
import chess.svg
from run_game import Game
from computer import Computer
from flask import Flask, render_template, request, Response
import time
app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def create_chess():
    print(game.board.state.legal_moves)
    return '''
        <img src="/board">
        <p><form action="/move"><input type=text name=move><input type=submit value="Make move"></form></p>
        <p><form action="/reset"><input type=submit value="Reset"></form></p>
    '''


@app.route("/reset")
def reset():
    print("Resetting game")
    game.restart_game()
    return create_chess()


@app.route("/move")
def move():
    move_str = request.args.get('move', default="")
    move = chess.Move.from_uci(move_str)
    if move in game.board.state.legal_moves:
        game.board.state.push(move)
        # Get computer move
        cpu_move = game.computer.generate_move(game.board)
        game.board.state.push(cpu_move)
    return create_chess()


@app.route("/board")
def board():
    return Response(chess.svg.board(game.board.state, size=350), mimetype='image/svg+xml')

if __name__ == "__main__":
    player_is_white = input("Enter: w for White, b for Black: ") == 'w'
    game = Game(player_is_white)
    app.run(debug=True)
