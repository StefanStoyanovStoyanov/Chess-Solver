import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

IMAGE_PATH = os.path.join(ROOT_DIR, "images", "chessboard.png")
OUTPUT_PATH = os.path.join(ROOT_DIR, "images", "solved_chessboard.png")

import numpy as np
import chess
from PIL import ImageDraw

import chessboard_finder
import load_image


def crop_chessboard(image, corners):
    return image.crop(tuple(int(coordinate) for coordinate in corners))


def draw_arrow(draw, start, end, color, width, arrowhead):
    import math

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    length = math.hypot(dx, dy)
    draw.line([start, end], fill=color, width=width)

    head_length = min(arrowhead, length * 0.4)
    for side in (-1, 1):
        head_angle = angle + side * math.radians(30)
        point = (
            end[0] - head_length * math.cos(head_angle),
            end[1] - head_length * math.sin(head_angle),
        )
        draw.line([end, point], fill=color, width=width)


def draw_best_move(
    white_move: str | None = None,
    black_move: str | None = None,
    output_path: str | None = None,
    image_path: str = IMAGE_PATH,
    black_on_bottom: bool = False,
) -> str:
    image = load_image.load_image(image_path)
    corners = chessboard_finder.findChessboardCorners(
        np.asarray(image.convert("L"), dtype=np.float32)
    )
    if corners is None:
        raise ValueError(f"Couldn't find a chessboard in image: {image_path}")

    board_image = crop_chessboard(image.convert("RGB"), corners)
    draw = ImageDraw.Draw(board_image)
    square_width = board_image.width / 8
    square_height = board_image.height / 8

    for move, color in ((white_move, (255, 0, 0)), (black_move, (0, 0, 255))):
        if move is None:
            continue

        chess_move = chess.Move.from_uci(move)
        source = chess.square_name(chess_move.from_square)
        destination = chess.square_name(chess_move.to_square)
        source_file = ord(source[0]) - ord("a")
        source_rank = int(source[1]) - 1
        destination_file = ord(destination[0]) - ord("a")
        destination_rank = int(destination[1]) - 1
        if not black_on_bottom:
            source_rank = 7 - source_rank
            destination_rank = 7 - destination_rank
        else:
            source_file = 7 - source_file
            destination_file = 7 - destination_file
        start = (
            (source_file + 0.5) * square_width,
            (source_rank + 0.5) * square_height,
        )
        end = (
            (destination_file + 0.5) * square_width,
            (destination_rank + 0.5) * square_height,
        )
        line_width = max(2, int(min(square_width, square_height) // 10))
        draw_arrow(
            draw,
            start,
            end,
            color=color,
            width=line_width,
            arrowhead=int(min(square_width, square_height) / 2.5),
        )

    draw.rectangle(
        [(0, 0), (board_image.width - 1, board_image.height - 1)],
        outline=(0, 255, 0),
        width=2,
    )

    output_path = output_path or OUTPUT_PATH
    board_image.save(output_path)
    return output_path


if __name__ == "__main__":
    from chess_solve import main

    raise SystemExit(main())
