import os
import re
import sys
from contextlib import closing

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

IMAGE_PATH = os.path.join(ROOT_DIR, "images", "chessboard.png")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "image_to_fen_model.pb")

import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import chessboard_finder
import load_image


def shorten_fen(fen: str) -> str:
    return re.sub(r"1+", lambda m: str(len(m.group(0))), fen)


def load_graph(model_filepath: str) -> tf.Graph:
    with tf.io.gfile.GFile(model_filepath, "rb") as model_file:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(model_file.read())

    with tf.compat.v1.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name="tcb")
    return graph


class ChessboardPredictor:
    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = MODEL_PATH
        graph = load_graph(model_path)
        self.sess = tf.compat.v1.Session(graph=graph)
        self.x = graph.get_tensor_by_name("tcb/Input:0")
        self.keep_prob = graph.get_tensor_by_name("tcb/KeepProb:0")
        self.prediction = graph.get_tensor_by_name("tcb/prediction:0")
        self.probabilities = graph.get_tensor_by_name("tcb/probabilities:0")

    def get_prediction(self, tiles: np.ndarray) -> tuple[str, np.ndarray]:
        if tiles is None or len(tiles) == 0:
            raise ValueError("Couldn't parse chessboard tiles.")

        validation_set = np.swapaxes(np.reshape(tiles, [32 * 32, 64]), 0, 1)
        guess_prob, guessed = self.sess.run(
            [self.probabilities, self.prediction],
            feed_dict={self.x: validation_set, self.keep_prob: 1.0},
        )

        tile_certainties = np.array(
            [
                probabilities[prediction]
                for probabilities, prediction in zip(guess_prob, guessed)
            ]
        ).reshape([8, 8])[::-1, :]
        label_names = " KQRBNPkqrbnp"
        piece_names = ["1" if label == 0 else label_names[label] for label in guessed]
        fen = "/".join(
            ["".join(piece_names[i * 8 : (i + 1) * 8]) for i in reversed(range(8))]
        )
        return fen, tile_certainties

    def make_prediction(self, image_path: str) -> list[str | float | None]:
        image = load_image.load_image(image_path)
        image = load_image.resize_image(image)
        if image is None:
            raise ValueError(f"Image is too large to process: {image_path}")

        tiles, _ = chessboard_finder.findGrayscaleTilesInImage(image)
        if tiles is None:
            raise ValueError(f"Couldn't find a chessboard in image: {image_path}")

        fen, tile_certainties = self.get_prediction(tiles)
        return [shorten_fen(fen), float(tile_certainties.min()), None]

    def close(self) -> None:
        self.sess.close()


def get_fen_from_image(image_path: str = IMAGE_PATH) -> tuple[str, np.ndarray]:
    image = load_image.load_image(image_path)
    image = load_image.resize_image(image)
    if image is None:
        raise ValueError(f"Image is too large to process: {image_path}")

    tiles, _ = chessboard_finder.findGrayscaleTilesInImage(image)
    if tiles is None:
        raise ValueError(f"Couldn't find a chessboard in image: {image_path}")

    with closing(ChessboardPredictor()) as predictor:
        board_placement, tile_certainties = predictor.get_prediction(tiles)

    fen = f"{shorten_fen(board_placement)} w - - 0 1"
    return fen, tile_certainties


def main() -> None:
    fen, tile_certainties = get_fen_from_image()
    certainty = tile_certainties.min()
    print("Per-tile certainty:")
    print(tile_certainties)
    print(
        f"Certainty range [{tile_certainties.min()} - {tile_certainties.max()}], Avg: {tile_certainties.mean()}"
    )
    print(f"---\nPredicted FEN:\n{fen}")
    print(f"Final Certainty: {certainty * 100:.1f}%")


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=3)
    main()
