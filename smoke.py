"""Small runnable demonstration; not a strength benchmark."""
import chess
import agent

board = chess.Board()
move = agent.get_move(board.fen(), 1000)
assert chess.Move.from_uci(move) in board.legal_moves
print(f"Legal demonstration move: {move}")
