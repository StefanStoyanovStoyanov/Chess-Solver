import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chess

from config import IMAGE_PATH, OUTPUT_PATH
from draw_best_move import draw_best_move
from fen_to_best_move import fen_to_best_moves
from image_to_fen import get_fen_from_image


def get_valid_orientation_candidates(fen: str) -> list[tuple[str, str, bool]]:
    board = chess.Board(fen)
    candidates = [("white_bottom", board, False)]

    rotated = chess.Board(None)
    for square, piece in board.piece_map().items():
        rotated.set_piece_at(square ^ 63, piece)
    rotated.turn = board.turn
    candidates.append(("black_bottom", rotated, True))

    return [
        (name, candidate.fen(), black_on_bottom)
        for name, candidate, black_on_bottom in candidates
        if candidate.is_valid()
    ]


def main(image_path: str = IMAGE_PATH, output_dir: str | None = None) -> int:
    try:
        output_dir = output_dir or os.path.dirname(OUTPUT_PATH)
        os.makedirs(output_dir, exist_ok=True)

        print("Extracting FEN from image...")
        fen, _ = get_fen_from_image(image_path)
        print(f"Predicted FEN: {fen}")

        candidates = get_valid_orientation_candidates(fen)
        if not candidates:
            raise ValueError("Neither board orientation is a valid chess position.")

        print("Getting best moves for valid board orientations...")
        for orientation, candidate_fen, black_on_bottom in candidates:
            print(f"Checking {orientation.replace('_', ' ')}...")
            white_move, black_move = fen_to_best_moves(candidate_fen)
            print(f"Best move for White: {white_move}")
            print(f"Best move for Black: {black_move}")

            print("Drawing best moves on board...")
            output_path = os.path.join(
                output_dir, f"solved_chessboard_{orientation}.png"
            )
            output_path = draw_best_move(
                white_move=white_move,
                black_move=black_move,
                image_path=image_path,
                output_path=output_path,
                black_on_bottom=black_on_bottom,
            )
            print(f"Saved: {output_path}")

        skipped = {"white_bottom", "black_bottom"} - {
            orientation for orientation, _, _ in candidates
        }
        for orientation in sorted(skipped):
            print(f"Skipped invalid {orientation.replace('_', ' ')} orientation.")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find best chess moves in an image and save annotated boards."
    )
    parser.add_argument(
        "--image",
        default=IMAGE_PATH,
        help="Path to the chessboard image to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(OUTPUT_PATH),
        help="Directory where solved orientation images will be saved.",
    )
    arguments = parser.parse_args()
    raise SystemExit(main(image_path=arguments.image, output_dir=arguments.output_dir))
