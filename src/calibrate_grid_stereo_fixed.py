import os
import re
import json
import glob
import cv2
import numpy as np

# =========================
# SETTINGS (relative to src/)
# =========================
DATASET_DIR = os.path.join("data", "calib")          # data/calib/<camid>/*.png|jpg...
OUT_JSON    = "stereo_calib_grid_fixed.json"

# Your board:
MARKERS_X = 5
MARKERS_Y = 7
DICT_NAME = "DICT_4X4_50"
MARKER_LENGTH_M = 0.020
MARKER_SEP_M    = 0.007

# Detection speed option:
# 0 = no resize (safest)
DETECT_MAX_WIDTH = 0

# Minimum matched points to accept a view/pair
MIN_POINTS_PER_VIEW = 12

# Stereo calibrate flags (keep intrinsics fixed after separate calibration)
STEREO_FLAGS = cv2.CALIB_FIX_INTRINSIC

# Rectification options
RECTIFY_ALPHA = 1.0  # 1.0 keeps more FOV (less crop), 0.0 crops more but cleaner
RECTIFY_FLAGS = cv2.CALIB_ZERO_DISPARITY
# =========================


def get_aruco_dict(name: str):
    name = name.strip().upper()
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dict: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def ensure_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def maybe_resize_for_detect(gray, max_w: int):
    """
    Resize only for detection; returns resized image and scale factors (sx, sy)
    to map corners back to full-res.
    """
    h, w = gray.shape[:2]
    if max_w is None or max_w <= 0 or w <= max_w:
        return gray, 1.0, 1.0

    scale = max_w / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # scale factors to go from small -> full-res
    sx = w / float(new_w)
    sy = h / float(new_h)
    return small, sx, sy


def extract_index(path: str):
    """Extract last integer group from filename to match pairs across cameras."""
    base = os.path.basename(path)
    nums = re.findall(r"(\d+)", base)
    if not nums:
        return None
    return int(nums[-1])


def collect_cam_images(dataset_dir: str):
    """
    Returns dict: cam_id -> { idx -> filepath }
    idx is parsed from the last integer found in the filename.
    """
    cam_dirs = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]
    cam_dirs = sorted(cam_dirs)
    if len(cam_dirs) < 2:
        raise RuntimeError(f"Need at least 2 camera folders in {dataset_dir}")

    cam_map = {}
    for d in cam_dirs:
        cam_id = os.path.basename(d)
        files = sorted(glob.glob(os.path.join(d, "*.*")))

        idx_map = {}
        for p in files:
            if os.path.isdir(p):
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                continue
            idx = extract_index(p)
            if idx is None:
                continue
            idx_map[idx] = p

        if len(idx_map) > 0:
            cam_map[cam_id] = idx_map

    if len(cam_map) < 2:
        raise RuntimeError("Could not find 2 valid camera image sets with numeric filenames.")
    return cam_map


def detect_markers_fullres(gray, aruco_dict, max_w):
    """
    Detect markers and return corners in FULL-RES coords.
    Returns corners_full (list), ids (Nx1 int32).
    """
    small, sx, sy = maybe_resize_for_detect(gray, max_w)

    corners, ids, _ = cv2.aruco.detectMarkers(small, aruco_dict)
    if ids is None or len(ids) == 0:
        return None, None

    corners_full = []
    for c in corners:
        cf = c.copy()
        cf[..., 0] *= sx
        cf[..., 1] *= sy
        corners_full.append(cf)

    ids = ids.astype(np.int32)
    return corners_full, ids


def filter_and_sort_by_common_ids(corners_full, ids, common_ids_set):
    """
    Keep only markers whose ids are in common_ids_set, then sort by id.
    Returns (corners_list, ids_array) suitable for OpenCV ArUco functions.
    """
    ids_list = [int(x) for x in ids.flatten().tolist()]
    pairs = [(i, c) for (i, c) in zip(ids_list, corners_full) if i in common_ids_set]
    pairs.sort(key=lambda t: t[0])

    if len(pairs) == 0:
        return None, None

    ids_out = np.array([[p[0]] for p in pairs], dtype=np.int32)
    corners_out = [p[1] for p in pairs]
    return corners_out, ids_out


def board_points_from_markers(board, corners_full, ids):
    """
    Convert detected markers into matched (objPoints, imgPoints) for the board.
    """
    obj, img = cv2.aruco.getBoardObjectAndImagePoints(board, corners_full, ids)
    if obj is None or img is None:
        return None, None, 0
    obj = np.asarray(obj, dtype=np.float32).reshape(-1, 3)
    img = np.asarray(img, dtype=np.float32).reshape(-1, 2)
    return obj, img, int(img.shape[0])


def calibrate_single_camera(obj_list, img_list, image_size):
    """
    Calibrate camera intrinsics with FIX_ASPECT_RATIO (square pixels assumption).
    This prevents crazy fx/fy mismatch that breaks rectification.
    """
    w, h = image_size  # (W, H)

    # Initial guess: fx = fy, principal point at image center
    K_init = np.array([
        [1000.0,   0.0, w / 2.0],
        [  0.0, 1000.0, h / 2.0],
        [  0.0,   0.0,     1.0]
    ], dtype=np.float64)
    D_init = np.zeros((5, 1), dtype=np.float64)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS |
        cv2.CALIB_FIX_ASPECT_RATIO
    )

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, image_size, K_init, D_init,
        flags=flags, criteria=criteria
    )
    return rms, K, D



def main():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco module not found. Install opencv-contrib-python.")

    aruco_dict = get_aruco_dict(DICT_NAME)
    board = cv2.aruco.GridBoard((MARKERS_X, MARKERS_Y), MARKER_LENGTH_M, MARKER_SEP_M, aruco_dict)

    cam_map = collect_cam_images(DATASET_DIR)

    # Use stable ordering
    cam_ids = sorted(list(cam_map.keys()))
    camA, camB = cam_ids[0], cam_ids[1]

    # Match indices across cameras
    common_idx = sorted(list(set(cam_map[camA].keys()) & set(cam_map[camB].keys())))
    if len(common_idx) == 0:
        raise RuntimeError("No common frame indices found between the two camera folders.")

    print("✅ Cameras (calib order):", camA, camB)
    print("✅ Common pairs:", len(common_idx))

    # Determine image size from first image (FULL RES)
    img0 = cv2.imread(cam_map[camA][common_idx[0]], cv2.IMREAD_UNCHANGED)
    if img0 is None:
        raise RuntimeError("Cannot read first image to get image size.")
    h0, w0 = img0.shape[:2]
    image_size = (w0, h0)  # IMPORTANT: (W, H)

    # Lists for intrinsics (per camera)
    objA_list, imgA_list = [], []
    objB_list, imgB_list = [], []

    # Lists for stereo (must have matched points count across cameras per pair)
    stereo_obj_list, stereo_imgA_list, stereo_imgB_list = [], [], []

    usableA = 0
    usableB = 0
    used_stereo = 0

    for idx in common_idx:
        pA = cam_map[camA][idx]
        pB = cam_map[camB][idx]

        imA = cv2.imread(pA, cv2.IMREAD_UNCHANGED)
        imB = cv2.imread(pB, cv2.IMREAD_UNCHANGED)
        if imA is None or imB is None:
            continue

        gA = ensure_gray(imA)
        gB = ensure_gray(imB)

        cornersA_full, idsA = detect_markers_fullres(gA, aruco_dict, DETECT_MAX_WIDTH)
        cornersB_full, idsB = detect_markers_fullres(gB, aruco_dict, DETECT_MAX_WIDTH)

        # Per-camera board points (for intrinsics)
        if idsA is not None:
            objA, imgA, nA = board_points_from_markers(board, cornersA_full, idsA)
            if nA >= MIN_POINTS_PER_VIEW:
                objA_list.append(objA)
                imgA_list.append(imgA)
                usableA += 1

        if idsB is not None:
            objB, imgB, nB = board_points_from_markers(board, cornersB_full, idsB)
            if nB >= MIN_POINTS_PER_VIEW:
                objB_list.append(objB)
                imgB_list.append(imgB)
                usableB += 1

        # Stereo: enforce SAME markers in both frames (intersection of IDs)
        if idsA is None or idsB is None:
            continue

        setA = set(int(x) for x in idsA.flatten().tolist())
        setB = set(int(x) for x in idsB.flatten().tolist())
        common_ids = setA & setB

        if len(common_ids) == 0:
            continue

        cornersA_f, idsA_f = filter_and_sort_by_common_ids(cornersA_full, idsA, common_ids)
        cornersB_f, idsB_f = filter_and_sort_by_common_ids(cornersB_full, idsB, common_ids)

        if cornersA_f is None or cornersB_f is None:
            continue

        objA2, imgA2, nA2 = board_points_from_markers(board, cornersA_f, idsA_f)
        objB2, imgB2, nB2 = board_points_from_markers(board, cornersB_f, idsB_f)

        if objA2 is None or imgA2 is None or objB2 is None or imgB2 is None:
            continue

        # Now counts must match
        if nA2 != nB2:
            continue
        if nA2 < MIN_POINTS_PER_VIEW:
            continue

        # Use SAME object points list; board coords are identical when ids are identical & sorted
        stereo_obj_list.append(objA2)
        stereo_imgA_list.append(imgA2)
        stereo_imgB_list.append(imgB2)
        used_stereo += 1

    print(f"✅ Detection: camA usable views={usableA}, camB usable views={usableB}")
    print(f"✅ Stereo usable pairs={used_stereo} / {len(common_idx)}")

    if used_stereo < 10:
        raise RuntimeError("Too few stereo pairs usable. Capture more/better calibration pairs.")

    # --- Calibrate intrinsics per camera
    rmsA, K1, D1 = calibrate_single_camera(objA_list, imgA_list, image_size)
    rmsB, K2, D2 = calibrate_single_camera(objB_list, imgB_list, image_size)

    print("\n=== Intrinsics sanity ===")
    print("camA fx,fy:", float(K1[0, 0]), float(K1[1, 1]))
    print("camB fx,fy:", float(K2[0, 0]), float(K2[1, 1]))
    print("Note: fx and fy should be in the same ballpark. Large mismatch means the dataset/detection still has issues.\n")

    # --- Stereo calibration with fixed intrinsics
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    rms_stereo, K1o, D1o, K2o, D2o, R, T, E, F = cv2.stereoCalibrate(
        stereo_obj_list,
        stereo_imgA_list,
        stereo_imgB_list,
        K1, D1, K2, D2,
        image_size,
        criteria=criteria,
        flags=STEREO_FLAGS
    )

    # --- Rectification (store convenience outputs)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1o, D1o, K2o, D2o,
        image_size,
        R, T,
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
        "rms_stereo": float(rms_stereo),
        "used_stereo_pairs": int(used_stereo),

        "detect_max_width": int(DETECT_MAX_WIDTH),
        "min_points_per_view": int(MIN_POINTS_PER_VIEW),
        "stereo_flags": int(STEREO_FLAGS),
        "rectify_alpha": float(RECTIFY_ALPHA),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("✅ Saved:", OUT_JSON)
    print("✅ Stereo RMS:", float(rms_stereo))
    print("✅ Baseline |T| (meters):", float(np.linalg.norm(T)))


if __name__ == "__main__":
    main()
