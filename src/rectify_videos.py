import os
import json
import glob
import cv2
import numpy as np

# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

RUN_DIR = os.path.join(PROJECT_ROOT, "data", "video", "20251218_002232")
CALIB_JSON = os.path.join(PROJECT_ROOT, "outputs", "calibration_outputs.json")

OUT_DIR = os.path.join(RUN_DIR, "rectified_fix")
FOURCC = "MJPG"
WRITE_DEBUG = True

# alpha=1 keeps more FOV (less "zoom"/crop) but may show black borders.
ALPHA = 1.0
FLAGS = cv2.CALIB_ZERO_DISPARITY
# =========================


def _mat3x3(flat9):
    return np.array(flat9, dtype=np.float64).reshape(3, 3)


def load_calib(path: str):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)

    W, H = j["img_size"]  # [1920,1200]
    K1 = _mat3x3(j["K1"])
    D1 = np.array(j["D1"], dtype=np.float64).reshape(-1, 1)
    K2 = _mat3x3(j["K2"])
    D2 = np.array(j["D2"], dtype=np.float64).reshape(-1, 1)

    R = _mat3x3(j["R"])
    T = np.array(j["T"], dtype=np.float64).reshape(3, 1)

    return (W, H), K1, D1, K2, D2, R, T


def build_maps(W, H, K, D, Rrect, P):
    newK = P[:3, :3]  # 3x3
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, Rrect, newK, (W, H), m1type=cv2.CV_16SC2
    )
    return map1, map2


def draw_epipolar_lines(img, step=80):
    out = img.copy()
    h = out.shape[0]
    for y in range(step // 2, h, step):
        cv2.line(out, (0, y), (out.shape[1] - 1, y), (0, 255, 0), 1, cv2.LINE_AA)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    vids = sorted(glob.glob(os.path.join(RUN_DIR, "*_color.avi")))
    if len(vids) < 2:
        raise RuntimeError(f"Did not find 2 '*_color.avi' videos in: {RUN_DIR}")

    left_path = vids[0]
    right_path = vids[1]

    left_id = os.path.basename(left_path).replace("_color.avi", "")
    right_id = os.path.basename(right_path).replace("_color.avi", "")

    print(" Using videos:")
    print("   cam1 =", left_path)
    print("   cam2 =", right_path)

    (Wc, Hc), K1, D1, K2, D2, R, T = load_calib(CALIB_JSON)

    # IMPORTANT: recompute stereo rectification properly from K/D/R/T and correct image_size=(W,H)
    image_size = (Wc, Hc)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, flags=FLAGS, alpha=ALPHA
    )

    map1L, map2L = build_maps(Wc, Hc, K1, D1, R1, P1)
    map1R, map2R = build_maps(Wc, Hc, K2, D2, R2, P2)

    capL = cv2.VideoCapture(left_path)
    capR = cv2.VideoCapture(right_path)
    if not capL.isOpened():
        raise RuntimeError(f"Cannot open: {left_path}")
    if not capR.isOpened():
        raise RuntimeError(f"Cannot open: {right_path}")

    fpsL = capL.get(cv2.CAP_PROP_FPS) or 0
    fpsR = capR.get(cv2.CAP_PROP_FPS) or 0
    fps = float(fpsL if fpsL > 0 else (fpsR if fpsR > 0 else 20.0))

    outL = os.path.join(OUT_DIR, f"{left_id}_rect.avi")
    outR = os.path.join(OUT_DIR, f"{right_id}_rect.avi")
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)

    vwL = cv2.VideoWriter(outL, fourcc, fps, (Wc, Hc))
    vwR = cv2.VideoWriter(outR, fourcc, fps, (Wc, Hc))
    if not vwL.isOpened() or not vwR.isOpened():
        raise RuntimeError("VideoWriter failed. Try FOURCC='XVID'.")

    vwDbg = None
    outDbg = os.path.join(OUT_DIR, "rect_debug_side_by_side.avi")
    if WRITE_DEBUG:
        vwDbg = cv2.VideoWriter(outDbg, fourcc, fps, (Wc * 2, Hc))
        if not vwDbg.isOpened():
            raise RuntimeError("Debug VideoWriter failed. Try FOURCC='XVID'.")

    idx = 0
    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            break

        idx += 1

        # Ensure correct size
        if (frameL.shape[1], frameL.shape[0]) != (Wc, Hc):
            frameL = cv2.resize(frameL, (Wc, Hc), interpolation=cv2.INTER_AREA)
        if (frameR.shape[1], frameR.shape[0]) != (Wc, Hc):
            frameR = cv2.resize(frameR, (Wc, Hc), interpolation=cv2.INTER_AREA)

        rectL = cv2.remap(frameL, map1L, map2L, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rectR = cv2.remap(frameR, map1R, map2R, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        vwL.write(rectL)
        vwR.write(rectR)

        if vwDbg is not None:
            dbgL = draw_epipolar_lines(rectL, step=80)
            dbgR = draw_epipolar_lines(rectR, step=80)
            side = np.hstack([dbgL, dbgR])
            cv2.putText(side, f"frame {idx}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            vwDbg.write(side)

    capL.release()
    capR.release()
    vwL.release()
    vwR.release()
    if vwDbg is not None:
        vwDbg.release()

    print("\n Rectification done (fixed).")
    print("Saved:", outL)
    print("Saved:", outR)
    if WRITE_DEBUG:
        print("Saved:", outDbg)
    print(f"FPS used: {fps:.2f}")


if __name__ == "__main__":
    main()
