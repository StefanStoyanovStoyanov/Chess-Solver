import numpy as np
import PIL.Image


def nonmax_suppress_1d(arr, winsize=5):
    _arr = arr.copy()
    for i in range(_arr.size):
        left_neighborhood = arr[max(0, i - winsize) : i] if i > 0 else 0
        right_neighborhood = (
            arr[i + 1 : min(arr.size - 1, i + winsize)] if i < _arr.size - 2 else 0
        )
        if arr[i] < np.max(left_neighborhood) or arr[i] <= np.max(right_neighborhood):
            _arr[i] = 0
    return _arr


def findChessboardCorners(img_arr_gray, noise_threshold=8000):
    gx, gy = np.gradient(img_arr_gray)
    gx_pos = gx.copy()
    gx_pos[gx_pos < 0] = 0
    gx_neg = -gx.copy()
    gx_neg[gx_neg < 0] = 0

    gy_pos = gy.copy()
    gy_pos[gy_pos < 0] = 0
    gy_neg = -gy.copy()
    gy_neg[gy_neg < 0] = 0

    hough_gx = gx_pos.sum(axis=1) * gx_neg.sum(axis=1)
    hough_gy = gy_pos.sum(axis=0) * gy_neg.sum(axis=0)

    if (
        min(hough_gx.std() / hough_gx.size, hough_gy.std() / hough_gy.size)
        < noise_threshold
    ):
        return None

    hough_gx = nonmax_suppress_1d(hough_gx) / hough_gx.max()
    hough_gy = nonmax_suppress_1d(hough_gy) / hough_gy.max()

    hough_gx[hough_gx < 0.2] = 0
    hough_gy[hough_gy < 0.2] = 0

    pot_lines_x = np.where(hough_gx)[0]
    pot_lines_y = np.where(hough_gy)[0]
    pot_lines_x_vals = hough_gx[pot_lines_x]
    pot_lines_y_vals = hough_gy[pot_lines_y]

    seqs_x = getAllSequences(pot_lines_x)
    seqs_y = getAllSequences(pot_lines_y)

    if len(seqs_x) == 0 or len(seqs_y) == 0:
        return None

    seqs_x_vals = [pot_lines_x_vals[[v in seq for v in pot_lines_x]] for seq in seqs_x]
    seqs_y_vals = [pot_lines_y_vals[[v in seq for v in pot_lines_y]] for seq in seqs_y]

    for i in range(len(seqs_x)):
        seq = seqs_x[i]
        seq_val = seqs_x_vals[i]

        if len(seq) > 9:
            while len(seq) > 7:
                if seq_val[0] > seq_val[-1]:
                    seq = seq[:-1]
                    seq_val = seq_val[:-1]
                else:
                    seq = seq[1:]
                    seq_val = seq_val[1:]

        seqs_x[i] = seq
        seqs_x_vals[i] = seq_val

    for i in range(len(seqs_y)):
        seq = seqs_y[i]
        seq_val = seqs_y_vals[i]

        while len(seq) > 9:
            if seq_val[0] > seq_val[-1]:
                seq = seq[:-1]
                seq_val = seq_val[:-1]
            else:
                seq = seq[1:]
                seq_val = seq_val[1:]

        seqs_y[i] = seq
        seqs_y_vals[i] = seq_val

    scores_x = np.array([np.mean(v) for v in seqs_x_vals])
    scores_y = np.array([np.mean(v) for v in seqs_y_vals])

    best_seq_x = seqs_x[scores_x.argmax()]
    best_seq_y = seqs_y[scores_y.argmax()]

    sub_seqs_x = [best_seq_x[k : k + 7] for k in range(len(best_seq_x) - 7 + 1)]
    sub_seqs_y = [best_seq_y[k : k + 7] for k in range(len(best_seq_y) - 7 + 1)]

    dx = np.median(np.diff(best_seq_x))
    dy = np.median(np.diff(best_seq_y))
    corners = np.zeros(4, dtype=int)

    corners[0] = int(best_seq_y[0] - dy)
    corners[1] = int(best_seq_x[0] - dx)
    corners[2] = int(best_seq_y[-1] + dy)
    corners[3] = int(best_seq_x[-1] + dx)

    gray_img_crop = PIL.Image.fromarray(img_arr_gray).crop(corners)

    k = 8
    quad = np.ones([k, k])
    kernel = np.vstack([np.hstack([quad, -quad]), np.hstack([-quad, quad])])
    kernel = np.tile(kernel, (4, 4))
    kernel = kernel / np.linalg.norm(kernel)

    k = 0
    final_corners = None
    best_score = None

    for i in range(len(sub_seqs_x)):
        for j in range(len(sub_seqs_y)):
            k = k + 1

            sub_corners = np.array(
                [
                    sub_seqs_y[j][0] - corners[0] - dy,
                    sub_seqs_x[i][0] - corners[1] - dx,
                    sub_seqs_y[j][-1] - corners[0] + dy,
                    sub_seqs_x[i][-1] - corners[1] + dx,
                ],
                dtype=int,
            )

            this_score = 0

            if (
                sub_corners[0] > 0
                and sub_corners[1] > 0
                and sub_corners[2] < np.min(gray_img_crop.size)
                and sub_corners[3] < np.min(gray_img_crop.size)
            ):
                this_score = 100
            else:
                this_score = -10

            if (best_score is None) or (this_score > best_score):
                best_score = this_score
                final_corners = (
                    corners[0] + sub_corners[0],
                    corners[1] + sub_corners[1],
                    corners[0] + sub_corners[2],
                    corners[1] + sub_corners[3],
                )
    return final_corners


def getAllSequences(seq, min_seq_len=7, err_px=5):
    seqs = []
    for i in range(seq.size - 1):
        for j in range(i + 1, seq.size):
            d = seq[j] - seq[i]
            if d < err_px or d > 100:
                continue

            s = [seq[i], seq[j]]
            n = s[-1] + d
            while np.abs((seq - n)).min() < err_px:
                n = seq[np.abs((seq - n)).argmin()]
                s.append(n)
                n = s[-1] + d

            if len(s) >= min_seq_len:
                s = np.array(s)
                seqs.append(s)
    return seqs


def getChessTilesColor(img, corners):
    if img.ndim != 3 or img.shape[2] != 3:
        print("Need RGB color image input")
        return None
    height, width, _ = img.shape

    padl_x = max(0, -corners[0])
    padl_y = max(0, -corners[1])
    padr_x = max(0, corners[2] - width)
    padr_y = max(0, corners[3] - height)

    img_padded = np.pad(img, ((padl_y, padr_y), (padl_x, padr_x), (0, 0)), mode="edge")

    chessboard_img = img_padded[
        (padl_y + corners[1]) : (padl_y + corners[3]),
        (padl_x + corners[0]) : (padl_x + corners[2]),
        :,
    ]

    chessboard_img_resized = (
        np.asarray(
            PIL.Image.fromarray(chessboard_img).resize([256, 256], PIL.Image.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )

    tiles = np.zeros([32, 32, 3 * 64], dtype=np.float32)
    for rank in range(8):
        for file in range(8):
            tiles[:, :, 3 * (rank * 8 + file) : 3 * (rank * 8 + file + 1)] = (
                chessboard_img_resized[
                    (7 - rank) * 32 : ((7 - rank) + 1) * 32, file * 32 : (file + 1) * 32
                ]
            )

    return tiles


def getChessBoardGray(img, corners):
    height, width = img.shape

    padl_x = max(0, -corners[0])
    padl_y = max(0, -corners[1])
    padr_x = max(0, corners[2] - width)
    padr_y = max(0, corners[3] - height)

    img_padded = np.pad(img, ((padl_y, padr_y), (padl_x, padr_x)), mode="edge")

    chessboard_img = img_padded[
        (padl_y + corners[1]) : (padl_y + corners[3]),
        (padl_x + corners[0]) : (padl_x + corners[2]),
    ]

    chessboard_img_resized = (
        np.asarray(
            PIL.Image.fromarray(chessboard_img).resize([256, 256], PIL.Image.BILINEAR),
            dtype=np.uint8,
        )
        / 255.0
    )
    return chessboard_img_resized


def getChessTilesGray(img, corners):
    chessboard_img_resized = getChessBoardGray(img, corners)
    return getTiles(chessboard_img_resized)


def getTiles(processed_gray_img):
    tiles = np.zeros([32, 32, 64], dtype=np.float32)
    for rank in range(8):
        for file in range(8):
            tiles[:, :, (rank * 8 + file)] = processed_gray_img[
                (7 - rank) * 32 : ((7 - rank) + 1) * 32, file * 32 : (file + 1) * 32
            ]

    return tiles


def findGrayscaleTilesInImage(img):
    if img is None:
        return None, None

    img_arr = np.asarray(img.convert("L"), dtype=np.float32)

    corners = findChessboardCorners(img_arr)
    if corners is None:
        return None, None

    tiles = getChessTilesGray(img_arr, corners)

    return tiles, corners
