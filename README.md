This repo implements a simple chess AI

- Build a representation for the board and pieces and board state by using a light-weight wrapper on top of python-chess API
- Uses a minimax algorithm with alpha-beta pruning to search the game tree when considering moves, and leaf node board positions
are evaluated using a combination of heuristic-based and learning-based techniques.
- Implemented a simple interactive web-app UI to play against the CPU using Flask

To launch app on localhost, run python app.py

![alt text](https://github.com/pavanchitta/chess-engine/blob/master/ui.png?raw=true)



