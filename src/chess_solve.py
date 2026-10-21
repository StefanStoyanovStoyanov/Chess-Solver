import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import IMAGE_PATH, OUTPUT_PATH
from draw_best_move import draw_best_move
from fen_to_best_move import fen_to_best_moves
from image_to_fen import get_fen_from_image


def main() -> int:
    try:
        print("[1/3] Extracting FEN from image...")
        fen, _ = get_fen_from_image(IMAGE_PATH)
        print(f"Predicted FEN: {fen}")

        print("[2/3] Getting best moves from FEN via Stockfish...")
        white_move, black_move = fen_to_best_moves(fen)
        print(f"Best move for White: {white_move}")
        print(f"Best move for Black: {black_move}")

        print("[3/3] Drawing best moves on board...")
        output_path = draw_best_move(
            white_move=white_move,
            black_move=black_move,
            image_path=IMAGE_PATH,
            output_path=OUTPUT_PATH,
        )
        print(f"Saved: {output_path}")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
