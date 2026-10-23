import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

STOCKFISH_PATH = os.path.join(ROOT_DIR, "models", "stockfish.exe")

import chess
import chess.engine


def get_fen_from_image() -> str:
    from image_to_fen import get_fen_from_image as extract_fen

    fen, _ = extract_fen()
    return fen


def fen_to_best_moves(fen: str) -> tuple[str, str]:
    try:
        board = chess.Board(fen)
    except ValueError as error:
        raise ValueError(f"Invalid FEN: {error}") from error

    if not board.is_valid():
        raise ValueError("Invalid chess position in FEN.")

    moves: dict[chess.Color, str] = {}
    with chess.engine.SimpleEngine.popen_uci(str(STOCKFISH_PATH)) as engine:
        for color in (chess.WHITE, chess.BLACK):
            position = board.copy(stack=False)
            position.turn = color
            if not position.legal_moves:
                raise ValueError(
                    f"No legal moves for {'White' if color == chess.WHITE else 'Black'}."
                )

            result = engine.play(position, chess.engine.Limit(time=2.0))
            if result.move is None:
                raise RuntimeError(
                    f"Stockfish returned no move for "
                    f"{'White' if color == chess.WHITE else 'Black'}."
                )
            moves[color] = result.move.uci()

    return moves[chess.WHITE], moves[chess.BLACK]


def main() -> tuple[str, str]:
    fen = get_fen_from_image()
    print(f"Predicted FEN: {fen}")
    white_move, black_move = fen_to_best_moves(fen)
    print(f"Best move for White: {white_move}")
    print(f"Best move for Black: {black_move}")
    return white_move, black_move


if __name__ == "__main__":
    main()
