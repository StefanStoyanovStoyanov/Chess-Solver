import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
MODEL_PATH = os.path.join(ASSETS_DIR, "image_to_fen_model.pb")
STOCKFISH_PATH = os.path.join(ASSETS_DIR, "stockfish.exe")
IMAGE_PATH = os.path.join(ASSETS_DIR, "chessboard.png")
OUTPUT_PATH = os.path.join(ASSETS_DIR, "solved_chessboard.png")
