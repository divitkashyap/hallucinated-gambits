"""Experimental Numba chess core.

This module is intentionally isolated from the active competition agent.  Its first
job is correctness: compact integer moves, legal move generation, and state updates
that can be checked exhaustively against python-chess before search is ported.
"""

from __future__ import annotations

from typing import Final

import chess
import numpy as np
from numba import njit

try:
    from .config import FAST_LEGALITY
except ImportError:
    from config import FAST_LEGALITY

EMPTY: Final = 0
PAWN: Final = 1
KNIGHT: Final = 2
BISHOP: Final = 3
ROOK: Final = 4
QUEEN: Final = 5
KING: Final = 6

WHITE: Final = 1
BLACK: Final = -1

SIDE: Final = 0
CASTLING: Final = 1
EP_SQUARE: Final = 2
HALFMOVE: Final = 3
FULLMOVE: Final = 4
HASH_KEY: Final = 5

WHITE_KING: Final = 1
WHITE_QUEEN: Final = 2
BLACK_KING: Final = 4
BLACK_QUEEN: Final = 8

CAPTURE: Final = 1 << 15
EN_PASSANT: Final = 1 << 16
CASTLE: Final = 1 << 17
DOUBLE_PUSH: Final = 1 << 18

MAX_MOVES: Final = 256
UNDO_SIZE: Final = 8
UNDO_CAPTURED: Final = 0
UNDO_CAPTURE_SQUARE: Final = 1
UNDO_CASTLING: Final = 2
UNDO_EP_SQUARE: Final = 3
UNDO_HALFMOVE: Final = 4
UNDO_FULLMOVE: Final = 5
UNDO_MOVED_PIECE: Final = 6
UNDO_HASH_KEY: Final = 7

KNIGHT_DELTAS: Final = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
KING_DELTAS: Final = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
BISHOP_DELTAS: Final = ((1, 1), (1, -1), (-1, -1), (-1, 1))
ROOK_DELTAS: Final = ((1, 0), (0, -1), (-1, 0), (0, 1))
SLIDER_DELTAS: Final = BISHOP_DELTAS + ROOK_DELTAS


def _zobrist_numbers(count: int) -> np.ndarray:
    """Create stable 63-bit SplitMix64 constants without runtime randomness."""
    mask = (1 << 64) - 1
    x = 0xA17C9E3779B97F4A
    values = np.empty(count, dtype=np.int64)
    for index in range(count):
        x = (x + 0x9E3779B97F4A7C15) & mask
        value = x
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        values[index] = value & ((1 << 63) - 1)
    return values


_ZOBRIST: Final = _zobrist_numbers(12 * 64 + 1 + 16 + 64)
ZOBRIST_PIECES: Final = _ZOBRIST[: 12 * 64].reshape(12, 64)
ZOBRIST_SIDE: Final = _ZOBRIST[12 * 64]
ZOBRIST_CASTLING: Final = _ZOBRIST[12 * 64 + 1 : 12 * 64 + 17]
ZOBRIST_EP: Final = _ZOBRIST[12 * 64 + 17 :]


def encode_move(
    from_square: int,
    to_square: int,
    promotion: int = 0,
    flags: int = 0,
) -> int:
    return from_square | (to_square << 6) | (promotion << 12) | flags


def move_to_uci(move: int) -> str:
    source = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    suffix = "" if promotion == 0 else {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}[promotion]
    return chess.square_name(source) + chess.square_name(target) + suffix


def position_from_fen(fen: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert a FEN to the fixed arrays consumed by the compiled core."""
    source = chess.Board(fen)
    board = np.zeros(64, dtype=np.int8)
    for square, piece in source.piece_map().items():
        board[square] = piece.piece_type if piece.color else -piece.piece_type

    castling = 0
    if source.has_kingside_castling_rights(chess.WHITE):
        castling |= WHITE_KING
    if source.has_queenside_castling_rights(chess.WHITE):
        castling |= WHITE_QUEEN
    if source.has_kingside_castling_rights(chess.BLACK):
        castling |= BLACK_KING
    if source.has_queenside_castling_rights(chess.BLACK):
        castling |= BLACK_QUEEN
    state = np.array(
        [
            WHITE if source.turn else BLACK,
            castling,
            -1 if source.ep_square is None else source.ep_square,
            source.halfmove_clock,
            source.fullmove_number,
            0,
        ],
        dtype=np.int64,
    )
    state[HASH_KEY] = compute_hash(board, state)
    return board, state


@njit(cache=False)
def _pack_move(source: int, target: int, promotion: int, flags: int) -> np.int32:
    return np.int32(source | (target << 6) | (promotion << 12) | flags)


@njit(cache=False)
def _append_move(
    moves: np.ndarray,
    count: int,
    source: int,
    target: int,
    promotion: int,
    flags: int,
) -> int:
    moves[count] = _pack_move(source, target, promotion, flags)
    return count + 1


@njit(cache=False)
def _piece_hash_index(piece: int) -> int:
    return piece - 1 if piece > 0 else 6 + (-piece - 1)


@njit(cache=False)
def compute_hash(board: np.ndarray, state: np.ndarray) -> np.int64:
    """Compute the full key used to validate incremental hash updates."""
    key = np.int64(0)
    for square in range(64):
        piece = int(board[square])
        if piece != EMPTY:
            key ^= ZOBRIST_PIECES[_piece_hash_index(piece), square]
    if state[SIDE] == BLACK:
        key ^= ZOBRIST_SIDE
    key ^= ZOBRIST_CASTLING[int(state[CASTLING])]
    if state[EP_SQUARE] >= 0:
        key ^= ZOBRIST_EP[int(state[EP_SQUARE])]
    return key


@njit(cache=False)
def is_square_attacked(board: np.ndarray, square: int, attacker: int) -> bool:
    target_file = square & 7
    target_rank = square >> 3

    pawn_source_rank = target_rank - attacker
    if 0 <= pawn_source_rank < 8:
        for pawn_source_file in (target_file - 1, target_file + 1):
            if 0 <= pawn_source_file < 8:
                source = pawn_source_rank * 8 + pawn_source_file
                if board[source] == attacker * PAWN:
                    return True

    for file_delta, rank_delta in KNIGHT_DELTAS:
        source_file = target_file + file_delta
        source_rank = target_rank + rank_delta
        if (
            0 <= source_file < 8
            and 0 <= source_rank < 8
            and board[source_rank * 8 + source_file] == attacker * KNIGHT
        ):
            return True

    for file_delta, rank_delta in KING_DELTAS:
        source_file = target_file + file_delta
        source_rank = target_rank + rank_delta
        if (
            0 <= source_file < 8
            and 0 <= source_rank < 8
            and board[source_rank * 8 + source_file] == attacker * KING
        ):
            return True

    for file_delta, rank_delta in BISHOP_DELTAS:
        source_file = target_file + file_delta
        source_rank = target_rank + rank_delta
        while 0 <= source_file < 8 and 0 <= source_rank < 8:
            piece = board[source_rank * 8 + source_file]
            if piece != EMPTY:
                if piece == attacker * BISHOP or piece == attacker * QUEEN:
                    return True
                break
            source_file += file_delta
            source_rank += rank_delta

    for file_delta, rank_delta in ROOK_DELTAS:
        source_file = target_file + file_delta
        source_rank = target_rank + rank_delta
        while 0 <= source_file < 8 and 0 <= source_rank < 8:
            piece = board[source_rank * 8 + source_file]
            if piece != EMPTY:
                if piece == attacker * ROOK or piece == attacker * QUEEN:
                    return True
                break
            source_file += file_delta
            source_rank += rank_delta
    return False


@njit(cache=False)
def _generate_pseudo_legal(board: np.ndarray, state: np.ndarray, moves: np.ndarray) -> int:
    side = int(state[SIDE])
    count = 0
    for source in range(64):
        piece = int(board[source])
        if piece == EMPTY or (piece > 0) != (side > 0):
            continue
        piece_type = abs(piece)
        source_file = source & 7
        source_rank = source >> 3

        if piece_type == PAWN:
            target = source + 8 * side
            promotion_rank = 7 if side == WHITE else 0
            start_rank = 1 if side == WHITE else 6
            if 0 <= target < 64 and board[target] == EMPTY:
                if target >> 3 == promotion_rank:
                    for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                        count = _append_move(moves, count, source, target, promotion, 0)
                else:
                    count = _append_move(moves, count, source, target, 0, 0)
                    double_target = source + 16 * side
                    if source_rank == start_rank and board[double_target] == EMPTY:
                        count = _append_move(moves, count, source, double_target, 0, DOUBLE_PUSH)
            for file_delta in (-1, 1):
                target_file = source_file + file_delta
                target_rank = source_rank + side
                if not (0 <= target_file < 8 and 0 <= target_rank < 8):
                    continue
                target = target_rank * 8 + target_file
                target_piece = int(board[target])
                is_ep = target == state[EP_SQUARE]
                if target_piece * side < 0 or is_ep:
                    flags = CAPTURE | (EN_PASSANT if is_ep else 0)
                    if target_rank == promotion_rank:
                        for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                            count = _append_move(moves, count, source, target, promotion, flags)
                    else:
                        count = _append_move(moves, count, source, target, 0, flags)
            continue

        if piece_type == KNIGHT:
            for file_delta, rank_delta in KNIGHT_DELTAS:
                target_file = source_file + file_delta
                target_rank = source_rank + rank_delta
                if 0 <= target_file < 8 and 0 <= target_rank < 8:
                    target = target_rank * 8 + target_file
                    target_piece = int(board[target])
                    if target_piece * side <= 0:
                        flags = CAPTURE if target_piece != EMPTY else 0
                        count = _append_move(moves, count, source, target, 0, flags)
            continue

        if piece_type == KING:
            for file_delta, rank_delta in KING_DELTAS:
                target_file = source_file + file_delta
                target_rank = source_rank + rank_delta
                if 0 <= target_file < 8 and 0 <= target_rank < 8:
                    target = target_rank * 8 + target_file
                    target_piece = int(board[target])
                    if target_piece * side <= 0:
                        flags = CAPTURE if target_piece != EMPTY else 0
                        count = _append_move(moves, count, source, target, 0, flags)
            rights = int(state[CASTLING])
            enemy = -side
            if side == WHITE and source == chess.E1:
                if (
                    rights & WHITE_KING
                    and board[chess.F1] == EMPTY
                    and board[chess.G1] == EMPTY
                    and board[chess.H1] == ROOK
                    and not is_square_attacked(board, chess.E1, enemy)
                    and not is_square_attacked(board, chess.F1, enemy)
                    and not is_square_attacked(board, chess.G1, enemy)
                ):
                    count = _append_move(moves, count, source, chess.G1, 0, CASTLE)
                if (
                    rights & WHITE_QUEEN
                    and board[chess.D1] == EMPTY
                    and board[chess.C1] == EMPTY
                    and board[chess.B1] == EMPTY
                    and board[chess.A1] == ROOK
                    and not is_square_attacked(board, chess.E1, enemy)
                    and not is_square_attacked(board, chess.D1, enemy)
                    and not is_square_attacked(board, chess.C1, enemy)
                ):
                    count = _append_move(moves, count, source, chess.C1, 0, CASTLE)
            elif side == BLACK and source == chess.E8:
                if (
                    rights & BLACK_KING
                    and board[chess.F8] == EMPTY
                    and board[chess.G8] == EMPTY
                    and board[chess.H8] == -ROOK
                    and not is_square_attacked(board, chess.E8, enemy)
                    and not is_square_attacked(board, chess.F8, enemy)
                    and not is_square_attacked(board, chess.G8, enemy)
                ):
                    count = _append_move(moves, count, source, chess.G8, 0, CASTLE)
                if (
                    rights & BLACK_QUEEN
                    and board[chess.D8] == EMPTY
                    and board[chess.C8] == EMPTY
                    and board[chess.B8] == EMPTY
                    and board[chess.A8] == -ROOK
                    and not is_square_attacked(board, chess.E8, enemy)
                    and not is_square_attacked(board, chess.D8, enemy)
                    and not is_square_attacked(board, chess.C8, enemy)
                ):
                    count = _append_move(moves, count, source, chess.C8, 0, CASTLE)
            continue

        if piece_type == BISHOP:
            first_direction = 0
            last_direction = 4
        elif piece_type == ROOK:
            first_direction = 4
            last_direction = 8
        else:
            first_direction = 0
            last_direction = 8
        for direction_index in range(first_direction, last_direction):
            file_delta, rank_delta = SLIDER_DELTAS[direction_index]
            target_file = source_file + file_delta
            target_rank = source_rank + rank_delta
            while 0 <= target_file < 8 and 0 <= target_rank < 8:
                target = target_rank * 8 + target_file
                target_piece = int(board[target])
                if target_piece * side > 0:
                    break
                flags = CAPTURE if target_piece != EMPTY else 0
                count = _append_move(moves, count, source, target, 0, flags)
                if target_piece != EMPTY:
                    break
                target_file += file_delta
                target_rank += rank_delta
    return count


@njit(cache=False)
def make_move(
    board: np.ndarray,
    state: np.ndarray,
    move: int,
    undo: np.ndarray,
) -> None:
    """Apply one encoded move in place and fill a fixed-size undo record."""
    source = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    piece = int(board[source])
    side = int(state[SIDE])
    capture_square = target - 8 * side if move & EN_PASSANT else target
    captured = int(board[capture_square])

    undo[UNDO_CAPTURED] = captured
    undo[UNDO_CAPTURE_SQUARE] = capture_square
    undo[UNDO_CASTLING] = state[CASTLING]
    undo[UNDO_EP_SQUARE] = state[EP_SQUARE]
    undo[UNDO_HALFMOVE] = state[HALFMOVE]
    undo[UNDO_FULLMOVE] = state[FULLMOVE]
    undo[UNDO_MOVED_PIECE] = piece
    undo[UNDO_HASH_KEY] = state[HASH_KEY]

    hash_key = np.int64(state[HASH_KEY])
    hash_key ^= ZOBRIST_PIECES[_piece_hash_index(piece), source]
    if captured != EMPTY:
        hash_key ^= ZOBRIST_PIECES[_piece_hash_index(captured), capture_square]
    hash_key ^= ZOBRIST_CASTLING[int(state[CASTLING])]
    if state[EP_SQUARE] >= 0:
        hash_key ^= ZOBRIST_EP[int(state[EP_SQUARE])]
    hash_key ^= ZOBRIST_SIDE

    board[source] = EMPTY
    placed_piece = side * promotion if promotion else piece
    board[target] = placed_piece
    hash_key ^= ZOBRIST_PIECES[_piece_hash_index(placed_piece), target]
    if move & EN_PASSANT:
        board[capture_square] = EMPTY
    if move & CASTLE:
        if target == chess.G1:
            hash_key ^= ZOBRIST_PIECES[_piece_hash_index(ROOK), chess.H1]
            hash_key ^= ZOBRIST_PIECES[_piece_hash_index(ROOK), chess.F1]
            board[chess.F1] = board[chess.H1]
            board[chess.H1] = EMPTY
        elif target == chess.C1:
            hash_key ^= ZOBRIST_PIECES[_piece_hash_index(ROOK), chess.A1]
            hash_key ^= ZOBRIST_PIECES[_piece_hash_index(ROOK), chess.D1]
            board[chess.D1] = board[chess.A1]
            board[chess.A1] = EMPTY
        elif target == chess.G8:
            hash_key ^= ZOBRIST_PIECES[_piece_hash_index(-ROOK), chess.H8]
            hash_key ^= ZOBRIST_PIECES[_piece_hash_index(-ROOK), chess.F8]
            board[chess.F8] = board[chess.H8]
            board[chess.H8] = EMPTY
        else:
            hash_key ^= ZOBRIST_PIECES[_piece_hash_index(-ROOK), chess.A8]
            hash_key ^= ZOBRIST_PIECES[_piece_hash_index(-ROOK), chess.D8]
            board[chess.D8] = board[chess.A8]
            board[chess.A8] = EMPTY

    rights = int(state[CASTLING])
    if abs(piece) == KING:
        rights &= ~(WHITE_KING | WHITE_QUEEN) if side == WHITE else ~(BLACK_KING | BLACK_QUEEN)
    if source == chess.A1 or target == chess.A1:
        rights &= ~WHITE_QUEEN
    if source == chess.H1 or target == chess.H1:
        rights &= ~WHITE_KING
    if source == chess.A8 or target == chess.A8:
        rights &= ~BLACK_QUEEN
    if source == chess.H8 or target == chess.H8:
        rights &= ~BLACK_KING
    state[CASTLING] = rights
    state[EP_SQUARE] = source + 8 * side if move & DOUBLE_PUSH else -1
    hash_key ^= ZOBRIST_CASTLING[rights]
    if state[EP_SQUARE] >= 0:
        hash_key ^= ZOBRIST_EP[int(state[EP_SQUARE])]
    if abs(piece) == PAWN or captured != EMPTY or move & EN_PASSANT:
        state[HALFMOVE] = 0
    else:
        state[HALFMOVE] += 1
    if side == BLACK:
        state[FULLMOVE] += 1
    state[SIDE] = -side
    state[HASH_KEY] = hash_key


@njit(cache=False)
def unmake_move(
    board: np.ndarray,
    state: np.ndarray,
    move: int,
    undo: np.ndarray,
) -> None:
    """Restore the exact board and state that existed before ``make_move``."""
    source = move & 63
    target = (move >> 6) & 63
    side = -int(state[SIDE])

    if move & CASTLE:
        if target == chess.G1:
            board[chess.H1] = board[chess.F1]
            board[chess.F1] = EMPTY
        elif target == chess.C1:
            board[chess.A1] = board[chess.D1]
            board[chess.D1] = EMPTY
        elif target == chess.G8:
            board[chess.H8] = board[chess.F8]
            board[chess.F8] = EMPTY
        else:
            board[chess.A8] = board[chess.D8]
            board[chess.D8] = EMPTY

    board[source] = undo[UNDO_MOVED_PIECE]
    if move & EN_PASSANT:
        board[target] = EMPTY
        board[int(undo[UNDO_CAPTURE_SQUARE])] = undo[UNDO_CAPTURED]
    else:
        board[target] = undo[UNDO_CAPTURED]

    state[SIDE] = side
    state[CASTLING] = undo[UNDO_CASTLING]
    state[EP_SQUARE] = undo[UNDO_EP_SQUARE]
    state[HALFMOVE] = undo[UNDO_HALFMOVE]
    state[FULLMOVE] = undo[UNDO_FULLMOVE]
    state[HASH_KEY] = undo[UNDO_HASH_KEY]


@njit(cache=False)
def _needs_legality_probe(move: int, king_square: int, in_check: bool) -> bool:
    """Only king moves, en passant, checks and possible pins need make/unmake.

    A non-king move cannot expose its king unless its source lies on a king
    ray. En passant is exceptional because it removes a second occupied square.
    Ray alignment deliberately overestimates pins, retaining the proven slow
    path for every ambiguous case. Pseudo-generation already checks castling.
    """
    source = move & 63
    if in_check or source == king_square or move & EN_PASSANT:
        return True
    file_delta = abs((source & 7) - (king_square & 7))
    rank_delta = abs((source >> 3) - (king_square >> 3))
    return file_delta == 0 or rank_delta == 0 or file_delta == rank_delta


@njit(cache=False)
def generate_legal_moves(
    board: np.ndarray,
    state: np.ndarray,
    pseudo: np.ndarray,
    legal: np.ndarray,
    undo: np.ndarray,
) -> int:
    pseudo_count = _generate_pseudo_legal(board, state, pseudo)
    legal_count = 0
    side = int(state[SIDE])
    king_square = -1
    for square in range(64):
        if board[square] == side * KING:
            king_square = square
            break
    if king_square < 0:
        return 0
    in_check = is_square_attacked(board, king_square, -side) if FAST_LEGALITY else True
    for index in range(pseudo_count):
        move = int(pseudo[index])
        if FAST_LEGALITY and not _needs_legality_probe(move, king_square, in_check):
            legal[legal_count] = move
            legal_count += 1
            continue
        source = move & 63
        target = (move >> 6) & 63
        moved_king = board[source] == side * KING
        make_move(board, state, move, undo)
        checked_square = target if moved_king else king_square
        legal_move = not is_square_attacked(board, checked_square, -side)
        unmake_move(board, state, move, undo)
        if legal_move:
            legal[legal_count] = move
            legal_count += 1
    return legal_count


@njit(cache=False)
def has_legal_move(
    board: np.ndarray,
    state: np.ndarray,
    pseudo: np.ndarray,
    undo: np.ndarray,
) -> bool:
    """Detect terminal nodes without constructing every legal move."""
    count = _generate_pseudo_legal(board, state, pseudo)
    side = int(state[SIDE])
    king_square = -1
    for square in range(64):
        if board[square] == side * KING:
            king_square = square
            break
    if king_square < 0:
        return False
    in_check = is_square_attacked(board, king_square, -side) if FAST_LEGALITY else True
    for index in range(count):
        move = int(pseudo[index])
        if FAST_LEGALITY and not _needs_legality_probe(move, king_square, in_check):
            return True
        source = move & 63
        target = (move >> 6) & 63
        moved_king = board[source] == side * KING
        make_move(board, state, move, undo)
        legal = not is_square_attacked(board, target if moved_king else king_square, -side)
        unmake_move(board, state, move, undo)
        if legal:
            return True
    return False


@njit(cache=False)
def perft(
    board: np.ndarray,
    state: np.ndarray,
    depth: int,
    move_buffers: np.ndarray,
    pseudo_buffers: np.ndarray,
    undo_buffers: np.ndarray,
    ply: int = 0,
) -> np.int64:
    """Count legal leaf nodes using the reversible path that search will use."""
    if depth == 0:
        return np.int64(1)
    move_count = generate_legal_moves(
        board, state, pseudo_buffers[ply], move_buffers[ply], undo_buffers[ply]
    )
    if depth == 1:
        return np.int64(move_count)

    nodes = np.int64(0)
    for index in range(move_count):
        move = int(move_buffers[ply, index])
        make_move(board, state, move, undo_buffers[ply])
        nodes += perft(board, state, depth - 1, move_buffers, pseudo_buffers, undo_buffers, ply + 1)
        unmake_move(board, state, move, undo_buffers[ply])
    return nodes


def legal_moves_uci(fen: str) -> set[str]:
    board, state = position_from_fen(fen)
    pseudo = np.empty(MAX_MOVES, dtype=np.int32)
    legal = np.empty(MAX_MOVES, dtype=np.int32)
    undo = np.empty(UNDO_SIZE, dtype=np.int64)
    count = generate_legal_moves(board, state, pseudo, legal, undo)
    return {move_to_uci(int(legal[index])) for index in range(count)}

@njit(cache=False)
def generate_legal_tactical(
    board: np.ndarray,
    state: np.ndarray,
    pseudo: np.ndarray,
    legal: np.ndarray,
    undo: np.ndarray,
) -> int:
    pseudo_count = _generate_pseudo_legal(board, state, pseudo)
    legal_count = 0
    side = int(state[SIDE])
    king_square = -1
    for square in range(64):
        if board[square] == side * KING:
            king_square = square
            break
    if king_square < 0:
        return 0
    in_check = is_square_attacked(board, king_square, -side) if FAST_LEGALITY else True
    for index in range(pseudo_count):
        move = int(pseudo[index])
        if not (move & CAPTURE or (move >> 12) & 7):
            continue
        if FAST_LEGALITY and not _needs_legality_probe(move, king_square, in_check):
            legal[legal_count] = move
            legal_count += 1
            continue
        source = move & 63
        target = (move >> 6) & 63
        moved_king = board[source] == side * KING
        make_move(board, state, move, undo)
        checked_square = target if moved_king else king_square
        legal_move = not is_square_attacked(board, checked_square, -side)
        unmake_move(board, state, move, undo)
        if legal_move:
            legal[legal_count] = move
            legal_count += 1
    return legal_count
