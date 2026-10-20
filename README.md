# Chess Solver

Chess Solver reads a chessboard from an image, predicts its position as FEN, asks Stockfish for a strong move for each side, and saves a copy of the board with both moves drawn on it.

## How it works

1. Detects and crops the chessboard in the input image.
2. Uses the TensorFlow model in `assets/model.pb` to identify the pieces and build a FEN position.
3. Uses the Stockfish engine in `assets/stockfish.exe` to calculate moves for White and Black.
4. Saves the detected board with White's move shown in red and Black's move in blue.

## Requirements

- Windows (the included Stockfish executable is `stockfish.exe`)
- Python and pip
- The files in `assets/`: `model.pb` and `stockfish.exe` are required to run; `chessboard.png` is the sample input image

Install the Python dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

TensorFlow must be installable for your Python version and Windows environment. If installation fails, use a Python/TensorFlow version combination supported by TensorFlow.

## Run

From the repository root:

```powershell
python src\chess_solve.py
```

By default, the program reads `assets/chessboard.png` and writes the annotated board to `assets/solved_chessboard.png`. Replace `assets/chessboard.png` with your own image to analyze a different position; the input and output paths are defined in `config.py`.

If the board is oriented with Black at the bottom, use:

```powershell
python src\chess_solve.py --invert_fen
```

The program prints the predicted FEN and the selected moves in UCI notation as it runs. It exits with an error message if it cannot load the image, detect the board, or calculate the moves.

## Analyze a FEN directly

To skip image recognition and calculate moves from a FEN position:

```powershell
python src\fen_to_best_move.py --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

This prints a move for White and a move for Black. The engine uses a two-second search limit for each move.

## Project layout

```text
.
├── assets/
│   ├── chessboard.png          # Default input image
│   ├── model.pb                # Piece-recognition model
│   ├── solved_chessboard.png   # Default output image
│   └── stockfish.exe           # Chess engine
├── src/
│   ├── chess_solve.py          # End-to-end workflow
│   ├── chessboard_finder.py    # Board detection and tile extraction
│   ├── draw_best_move.py       # Annotates moves on the board
│   ├── fen_to_best_move.py     # Finds moves from a FEN position
│   ├── image_to_fen.py         # Predicts piece placement from an image
│   └── load_image.py           # Image loading and resizing
├── config.py                   # Asset and image paths
└── requirements.txt            # Python dependencies
```

## Notes

- Use a clear image in which the full board is visible and the board squares are distinguishable. Detection or piece recognition may be inaccurate for cropped, obstructed, rotated, or low-quality images.
- The prediction model identifies piece placement; it does not infer whose turn it is or castling/en-passant state from the image. The end-to-end image workflow uses a default FEN turn of White.
- The FEN command accepts a complete FEN string, including its side-to-move field.