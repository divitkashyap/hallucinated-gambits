"""Conservative lookup of popular moves from the frozen human-game corpus."""

import json
import pkgutil

import chess


def load_book() -> dict[str, list]:
    # Loader access works both from an extracted submission and directly in a
    # ZIP during local validation. Read only our own packaged data at import.
    raw = pkgutil.get_data(__name__, "human_book.json")
    if raw is None:
        raise ValueError("Missing packaged human opening book")
    return json.loads(raw)


def select_move(board: chess.Board, book: dict[str, list], used: int) -> str | None:
    if board.fullmove_number > 16 or board.halfmove_clock >= 8 or used >= 8:
        return None
    key = " ".join(board.fen(en_passant="legal").split()[:4])
    row = book.get(key)
    if row is None:
        return None
    uci, count, total = row
    if count < 3 or count / total < .45:
        return None
    move = chess.Move.from_uci(uci)
    return uci if move in board.legal_moves else None
