import os
import re
import json
import glob
import cv2
import numpy as np

# -------------------------
# Settings (relative to src/)
# -------------------------
DATASET_DIR = os.path.join("..", "data", "calib")
OUT_JSON = "outputs/calibration_outputs.json"

MARKERS_X = 5
MARKERS_Y = 7
DICT_NAME = "DICT_4X4_50"
MARKER_LENGTH_M = 0.020
MARKER_SEP_M = 0.007

DETECT_MAX_WIDTH = 0          # 0 = no resize (safest)
MIN_POINTS_PER_VIEW = 12

STEREO_FLAGS = cv2.CALIB_FIX_INTRINSIC
RECTIFY_ALPHA = 1.0
RECTIFY_FLAGS = cv2.CALIB_ZERO_DISPARITY
# -------------------------


def aruco_dict(name: str):
    name = name.strip().upper()
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dict: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_markers_fullres(gray, dct, max_w: int):
    """
    Detect ArUco markers; optionally run detection on a resized image.
    Returns corners in full-resolution coordinates (for calibration).
    """
    h, w = gray.shape[:2]
    if max_w and w > max_w:
        scale = max_w / float(w)
        small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        sx, sy = w / float(small.shape[1]), h / float(small.shape[0])
    else:
        small, sx, sy = gray, 1.0, 1.0

    corners, ids, _ = cv2.aruco.detectMarkers(small, dct)
    if ids is None or len(ids) == 0:
        return None, None

    corners_full = []
    for c in corners:
        cf = c.copy()
        cf[..., 0] *= sx
        cf[..., 1] *= sy
        corners_full.append(cf)

    return corners_full, ids.astype(np.int32)


def last_int_in_name(path: str):
    nums = re.findall(r"(\d+)", os.path.basename(path))
    return int(nums[-1]) if nums else None


def collect_images(dataset_dir: str):
    """
    Returns: {cam_id: {idx: filepath}}
    idx is the last integer group found in the filename.
    """
    cam_dirs = sorted(d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d))
    if len(cam_dirs) < 2:
        raise RuntimeError(f"Need at least 2 camera folders in {dataset_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    cam_map = {}
    for d in cam_dirs:
        cam_id = os.path.basename(d)
        idx_map = {}
        for p in sorted(glob.glob(os.path.join(d, "*.*"))):
            if os.path.isdir(p) or os.path.splitext(p)[1].lower() not in exts:
                continue
            idx = last_int_in_name(p)
            if idx is not None:
                idx_map[idx] = p
        if idx_map:
            cam_map[cam_id] = idx_map

    if len(cam_map) < 2:
        raise RuntimeError("Could not find two valid camera image sets with numeric filenames.")
    return cam_map


def board_points(board, corners, ids):
    """
    Convert detected markers to matched object/image points for the board.
    """
    obj, img = cv2.aruco.getBoardObjectAndImagePoints(board, corners, ids)
    if obj is None or img is None:
        return None, None, 0
    obj = np.asarray(obj, dtype=np.float32).reshape(-1, 3)
    img = np.asarray(img, dtype=np.float32).reshape(-1, 2)
    return obj, img, int(img.shape[0])


def keep_common_ids(corners, ids, common_ids):
    """
    Keep only markers with ids in common_ids, sorted by id.
    """
    ids_list = ids.flatten().tolist()
    pairs = [(int(i), c) for i, c in zip(ids_list, corners) if int(i) in common_ids]
    if not pairs:
        return None, None
    pairs.sort(key=lambda t: t[0])
    ids_out = np.array([[p[0]] for p in pairs], dtype=np.int32)
    corners_out = [p[1] for p in pairs]
    return corners_out, ids_out


def calibrate_intrinsics(obj_list, img_list, image_size):
    """
    Intrinsics calibration with fixed aspect ratio (square pixel assumption).
    Helps avoid unstable fx/fy that can break rectification.
    """
    w, h = image_size
    K0 = np.array([[1000.0, 0.0, w / 2.0],
                   [0.0, 1000.0, h / 2.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    D0 = np.zeros((5, 1), dtype=np.float64)

    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    rms, K, D, *_ = cv2.calibrateCamera(obj_list, img_list, image_size, K0, D0, flags=flags, criteria=crit)
    return float(rms), K, D


def main():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module not found. Install opencv-contrib-python.")

    dct = aruco_dict(DICT_NAME)
    board = cv2.aruco.GridBoard((MARKERS_X, MARKERS_Y), MARKER_LENGTH_M, MARKER_SEP_M, dct)

    cam_map = collect_images(DATASET_DIR)
    cam_ids = sorted(cam_map.keys())
    camA, camB = cam_ids[0], cam_ids[1]

    common_idx = sorted(set(cam_map[camA]) & set(cam_map[camB]))
    if not common_idx:
        raise RuntimeError("No common frame indices found between the two camera folders.")

    # Full-resolution size from first image
    img0 = cv2.imread(cam_map[camA][common_idx[0]], cv2.IMREAD_UNCHANGED)
    if img0 is None:
        raise RuntimeError("Cannot read first image to determine image size.")
    h0, w0 = img0.shape[:2]
    image_size = (w0, h0)  # (W, H)

    objA_list, imgA_list = [], []
    objB_list, imgB_list = [], []
    stereo_obj, stereo_imgA, stereo_imgB = [], [], []

    for idx in common_idx:
        imA = cv2.imread(cam_map[camA][idx], cv2.IMREAD_UNCHANGED)
        imB = cv2.imread(cam_map[camB][idx], cv2.IMREAD_UNCHANGED)
        if imA is None or imB is None:
            continue

        gA, gB = to_gray(imA), to_gray(imB)
        cA, idsA = detect_markers_fullres(gA, dct, DETECT_MAX_WIDTH)
        cB, idsB = detect_markers_fullres(gB, dct, DETECT_MAX_WIDTH)

        # Per-camera views (intrinsics)
        if idsA is not None:
            oA, iA, nA = board_points(board, cA, idsA)
            if nA >= MIN_POINTS_PER_VIEW:
                objA_list.append(oA); imgA_list.append(iA)

        if idsB is not None:
            oB, iB, nB = board_points(board, cB, idsB)
            if nB >= MIN_POINTS_PER_VIEW:
                objB_list.append(oB); imgB_list.append(iB)

        # Stereo views: enforce same marker IDs in both images
        if idsA is None or idsB is None:
            continue

        common_ids = set(map(int, idsA.flatten())) & set(map(int, idsB.flatten()))
        if not common_ids:
            continue

        cA2, idsA2 = keep_common_ids(cA, idsA, common_ids)
        cB2, idsB2 = keep_common_ids(cB, idsB, common_ids)
        if cA2 is None or cB2 is None:
            continue

        oA2, iA2, nA2 = board_points(board, cA2, idsA2)
        oB2, iB2, nB2 = board_points(board, cB2, idsB2)
        if nA2 != nB2 or nA2 < MIN_POINTS_PER_VIEW:
            continue

        stereo_obj.append(oA2)
        stereo_imgA.append(iA2)
        stereo_imgB.append(iB2)

    used_stereo = len(stereo_obj)
    if used_stereo < 10:
        raise RuntimeError("Too few usable stereo pairs. Capture more/better calibration pairs.")

    print("Cameras (calib order):", camA, camB)
    print("Common pairs:", len(common_idx))
    print(f"Usable views: camA={len(objA_list)} camB={len(objB_list)} | stereo={used_stereo}")

    # Intrinsics
    rmsA, K1, D1 = calibrate_intrinsics(objA_list, imgA_list, image_size)
    rmsB, K2, D2 = calibrate_intrinsics(objB_list, imgB_list, image_size)

    # Stereo calibration (intrinsics fixed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    rms_st, K1o, D1o, K2o, D2o, R, T, E, F = cv2.stereoCalibrate(
        stereo_obj, stereo_imgA, stereo_imgB,
        K1, D1, K2, D2,
        image_size,
        criteria=crit,
        flags=STEREO_FLAGS
    )

    # Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1o, D1o, K2o, D2o,
        image_size, R, T,
        flags=RECTIFY_FLAGS,
        alpha=RECTIFY_ALPHA
    )

    out = {
        "img_size": [int(image_size[0]), int(image_size[1])],
        "markersX": int(MARKERS_X),
        "markersY": int(MARKERS_Y),
        "markerLength": float(MARKER_LENGTH_M),
        "markerSeparation": float(MARKER_SEP_M),
        "dict": DICT_NAME,
        "cam1_id": camA,
        "cam2_id": camB,

        "K1": K1o.reshape(-1).tolist(),
        "D1": D1o.reshape(-1).tolist(),
        "K2": K2o.reshape(-1).tolist(),
        "D2": D2o.reshape(-1).tolist(),
        "R": R.reshape(-1).tolist(),
        "T": T.reshape(-1).tolist(),

        "R1": R1.reshape(-1).tolist(),
        "R2": R2.reshape(-1).tolist(),
        "P1": P1.reshape(-1).tolist(),
        "P2": P2.reshape(-1).tolist(),
        "Q": Q.reshape(-1).tolist(),
        "roi1": [int(x) for x in roi1],
        "roi2": [int(x) for x in roi2],

        "rms_cam1": float(rmsA),
        "rms_cam2": float(rmsB),
        "rms_stereo": float(rms_st),
        "used_stereo_pairs": int(used_stereo),

        "detect_max_width": int(DETECT_MAX_WIDTH),
        "min_points_per_view": int(MIN_POINTS_PER_VIEW),
        "stereo_flags": int(STEREO_FLAGS),
        "rectify_alpha": float(RECTIFY_ALPHA),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved:", OUT_JSON)
    print("Stereo RMS:", float(rms_st))
    print("Baseline |T| (meters):", float(np.linalg.norm(T)))


if __name__ == "__main__":
    main()

