import os
import json
import glob
import csv
import cv2
import numpy as np

# =========================
# SETTINGS (relative to src/)
# =========================
RUN_DIR = os.path.join("..","data", "video", "20251218_002232")
RECT_DIR = os.path.join(RUN_DIR, "rectified_fix")
CALIB_JSON = "stereo_calib_grid_fixed.json"

OUT_DIR = os.path.join(RUN_DIR, "tracking")  # overwrite same folder each run

WRITE_DEBUG_VIDEOS = True
FOURCC = "MJPG"

# Rectified epipolar tolerance
Y_TOL_PX = 15  # this is stricter; try 20-30 if rectification isn't perfect

# ROI tracking
USE_ROI_TRACKING = True
ROI_HALF_SIZE = 220
ROI_HALF_MAX = 1400
ROI_EXPAND_ON_MISS = True

# If we miss too many pairs, go back to full-frame reacquire
PAIR_MISS_TO_REACQUIRE = 6

# Ball size constraints
MIN_RADIUS = 3
MAX_RADIUS = 90

# HSV strict (the one that used to work near)
HSV_LOWER = (5, 80, 80)
HSV_UPPER = (25, 255, 255)

USE_SECOND_RANGE = True
HSV2_LOWER = (0, 120, 80)
HSV2_UPPER = (5, 255, 255)

# HSV relaxed (far / low saturation)
HSV_RELAX_LOWER = (3, 50, 50)
HSV_RELAX_UPPER = (30, 255, 255)
HSV2_RELAX_LOWER = (0, 70, 50)
HSV2_RELAX_UPPER = (6, 255, 255)

# Hough fallback (set False if it locks onto wrong circles)
USE_HOUGH_FALLBACK = True

# =========================


def mat3x4(flat12):
    return np.array(flat12, dtype=np.float64).reshape(3, 4)


def load_P_mats(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    P1 = mat3x4(j["P1"])
    P2 = mat3x4(j["P2"])
    return P1, P2


def find_rect_videos(rect_dir: str):
    vids = sorted(glob.glob(os.path.join(rect_dir, "*_rect.avi")))
    vids = [v for v in vids if "debug" not in os.path.basename(v).lower()]
    if len(vids) < 2:
        raise RuntimeError(f"Did not find 2 '*_rect.avi' videos in: {rect_dir}")
    return vids[0], vids[1]


def clamp_roi(x, y, w, h, half):
    x0 = max(0, int(x - half))
    y0 = max(0, int(y - half))
    x1 = min(w, int(x + half))
    y1 = min(h, int(y + half))
    return x0, y0, x1, y1


def triangulate_point(P1, P2, ptL, ptR):
    xL, yL = ptL
    xR, yR = ptR
    pts1 = np.array([[xL], [yL]], dtype=np.float64)
    pts2 = np.array([[xR], [yR]], dtype=np.float64)
    X_h = cv2.triangulatePoints(P1, P2, pts1, pts2)[:, 0]
    if abs(X_h[3]) < 1e-9:
        return None
    X = X_h[0:3] / X_h[3]
    return float(X[0]), float(X[1]), float(X[2])


def detect_orange_ball(frame_bgr, last_xy=None, roi_half=None, mode="TRACK"):
    """
    v4-style detection:
      1) HSV strict -> best contour circle
      2) HSV relaxed -> best contour circle
      3) Hough fallback in grayscale (optional)

    Returns: (found, (cx,cy), radius, method, roi_box)
    roi_box = (x0,y0,x1,y1) where we searched
    """
    h, w = frame_bgr.shape[:2]

    if roi_half is None:
        roi_half = max(w, h)

    # ROI
    x0, y0, x1, y1 = 0, 0, w, h
    roi = frame_bgr
    if USE_ROI_TRACKING and (last_xy is not None) and (mode == "TRACK"):
        x0, y0, x1, y1 = clamp_roi(last_xy[0], last_xy[1], w, h, roi_half)
        roi = frame_bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    def make_mask(lower, upper, lower2=None, upper2=None):
        m1 = cv2.inRange(hsv, lower, upper)
        if USE_SECOND_RANGE and lower2 is not None and upper2 is not None:
            m2 = cv2.inRange(hsv, lower2, upper2)
            return cv2.bitwise_or(m1, m2)
        return m1

    def clean_mask(mask, gentle=False):
        if gentle:
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            k = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        else:
            mask = cv2.medianBlur(mask, 5)
            k = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        return mask

    def best_contour_circle(mask, circ_min=0.35):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        best = None
        best_score = -1.0

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 12:
                continue

            (x, y), r = cv2.minEnclosingCircle(c)
            r = float(r)
            if r < MIN_RADIUS or r > MAX_RADIUS:
                continue

            per = cv2.arcLength(c, True)
            if per <= 1e-6:
                continue

            circ = 4.0 * np.pi * area / (per * per)
            if circ < circ_min:
                continue

            score = circ * area
            if score > best_score:
                best_score = score
                best = (x, y, r)

        return best

    # Pass 1: HSV strict
    m1 = make_mask(HSV_LOWER, HSV_UPPER, HSV2_LOWER, HSV2_UPPER)
    m1 = clean_mask(m1, gentle=False)
    b = best_contour_circle(m1, circ_min=0.40)
    if b is not None:
        cx, cy, r = b
        return True, (float(cx + x0), float(cy + y0)), float(r), "hsv_strict", (x0, y0, x1, y1)

    # Pass 2: HSV relaxed
    m2 = make_mask(HSV_RELAX_LOWER, HSV_RELAX_UPPER, HSV2_RELAX_LOWER, HSV2_RELAX_UPPER)
    m2 = clean_mask(m2, gentle=True)
    b = best_contour_circle(m2, circ_min=0.30)
    if b is not None:
        cx, cy, r = b
        return True, (float(cx + x0), float(cy + y0)), float(r), "hsv_relaxed", (x0, y0, x1, y1)

    # Pass 3: Hough fallback
    if USE_HOUGH_FALLBACK:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=20,
            param1=120, param2=18,
            minRadius=MIN_RADIUS, maxRadius=MAX_RADIUS
        )
        if circles is not None:
            circles = np.squeeze(circles).astype(np.float64)
            if circles.ndim == 1:
                circles = circles.reshape(1, 3)
            cx, cy, r = circles[0]
            return True, (float(cx + x0), float(cy + y0)), float(r), "hough_gray", (x0, y0, x1, y1)

    return False, None, None, "none", (x0, y0, x1, y1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    left_vid, right_vid = find_rect_videos(RECT_DIR)
    print("✅ Rect videos:")
    print("   L:", left_vid)
    print("   R:", right_vid)

    P1, P2 = load_P_mats(CALIB_JSON)

    capL = cv2.VideoCapture(left_vid)
    capR = cv2.VideoCapture(right_vid)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError("Cannot open rectified videos.")

    fps = capL.get(cv2.CAP_PROP_FPS) or 20.0
    W = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Debug writers
    vwL = vwR = None
    if WRITE_DEBUG_VIDEOS:
        fourcc = cv2.VideoWriter_fourcc(*FOURCC)
        vwL = cv2.VideoWriter(os.path.join(OUT_DIR, "left_debug.avi"), fourcc, float(fps), (W, H))
        vwR = cv2.VideoWriter(os.path.join(OUT_DIR, "right_debug.avi"), fourcc, float(fps), (W, H))
        if not vwL.isOpened() or not vwR.isOpened():
            raise RuntimeError("Debug VideoWriter failed. Try FOURCC='XVID'.")

    # CSV outputs
    ball2d_path = os.path.join(OUT_DIR, "ball_2d.csv")
    traj3d_path = os.path.join(OUT_DIR, "trajectory_3d.csv")

    f2d = open(ball2d_path, "w", newline="", encoding="utf-8")
    w2d = csv.writer(f2d)
    w2d.writerow(["frame", "mode",
                  "foundL", "xL", "yL", "rL", "methL",
                  "foundR", "xR", "yR", "rR", "methR",
                  "abs_y", "valid_pair"])

    f3d = open(traj3d_path, "w", newline="", encoding="utf-8")
    w3d = csv.writer(f3d)
    w3d.writerow(["frame", "valid_3d", "X", "Y", "Z"])

    mode = "ACQUIRE"
    lastL = None
    lastR = None
    pair_miss = 0

    frame = 0
    ok_pairs = 0
    ok_3d = 0

    while True:
        okL, imgL = capL.read()
        okR, imgR = capR.read()
        if not okL or not okR:
            break
        frame += 1

        # ROI expand logic (TRACK)
        roi_half = ROI_HALF_SIZE
        if mode == "TRACK" and ROI_EXPAND_ON_MISS and pair_miss >= 2:
            roi_half = min(ROI_HALF_MAX, ROI_HALF_SIZE * (2 ** min(pair_miss - 2, 4)))

        # Detect
        foundL, ptL, rL, mL, roiL = detect_orange_ball(
            imgL, last_xy=lastL, roi_half=roi_half, mode=mode
        )
        foundR, ptR, rR, mR, roiR = detect_orange_ball(
            imgR, last_xy=lastR, roi_half=roi_half, mode=mode
        )

        valid_pair = 0
        abs_y = ""

        if foundL and foundR:
            abs_y = abs(ptL[1] - ptR[1])
            if abs_y <= Y_TOL_PX:
                valid_pair = 1
                ok_pairs += 1
                lastL = ptL
                lastR = ptR
                pair_miss = 0
                mode = "TRACK"
            else:
                pair_miss += 1
        else:
            pair_miss += 1

        # Reacquire if too many misses
        if mode == "TRACK" and pair_miss >= PAIR_MISS_TO_REACQUIRE:
            mode = "ACQUIRE"
            lastL = None
            lastR = None
            pair_miss = 0

        # Write 2D CSV
        w2d.writerow([
            frame, mode,
            int(foundL), f"{ptL[0]:.3f}" if foundL else "", f"{ptL[1]:.3f}" if foundL else "", f"{rL:.2f}" if foundL else "", mL,
            int(foundR), f"{ptR[0]:.3f}" if foundR else "", f"{ptR[1]:.3f}" if foundR else "", f"{rR:.2f}" if foundR else "", mR,
            f"{abs_y:.3f}" if abs_y != "" else "",
            int(valid_pair)
        ])

        # 3D (we keep it but you can ignore)
        if valid_pair:
            xyz = triangulate_point(P1, P2, ptL, ptR)
            if xyz is not None:
                X, Y, Z = xyz
                ok_3d += 1
                w3d.writerow([frame, 1, f"{X:.6f}", f"{Y:.6f}", f"{Z:.6f}"])
            else:
                w3d.writerow([frame, 0, "", "", ""])
        else:
            w3d.writerow([frame, 0, "", "", ""])

        # Debug draw
        if WRITE_DEBUG_VIDEOS:
            dL = imgL.copy()
            dR = imgR.copy()

            # ROI rectangles (so you SEE where it searches)
            x0, y0, x1, y1 = roiL
            cv2.rectangle(dL, (x0, y0), (x1, y1), (255, 255, 255), 2)
            x0, y0, x1, y1 = roiR
            cv2.rectangle(dR, (x0, y0), (x1, y1), (255, 255, 255), 2)

            if foundL:
                cv2.circle(dL, (int(ptL[0]), int(ptL[1])), int(max(3, rL)), (0, 255, 255), 2)
                cv2.circle(dL, (int(ptL[0]), int(ptL[1])), 2, (0, 0, 255), -1)
            if foundR:
                cv2.circle(dR, (int(ptR[0]), int(ptR[1])), int(max(3, rR)), (0, 255, 255), 2)
                cv2.circle(dR, (int(ptR[0]), int(ptR[1])), 2, (0, 0, 255), -1)

            cv2.putText(dL, f"frame={frame} mode={mode} miss={pair_miss} L={mL}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dR, f"frame={frame} mode={mode} miss={pair_miss} R={mR}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if valid_pair:
                cv2.putText(dL, f"PAIR OK |y|={abs_y:.1f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(dR, f"PAIR OK |y|={abs_y:.1f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(dL, "PAIR NO", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(dR, "PAIR NO", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            vwL.write(dL)
            vwR.write(dR)

    capL.release()
    capR.release()
    if vwL is not None:
        vwL.release()
    if vwR is not None:
        vwR.release()
    f2d.close()
    f3d.close()

    print("\n✅ Done.")
    print("2D CSV:", ball2d_path)
    print("3D CSV:", traj3d_path)
    if WRITE_DEBUG_VIDEOS:
        print("Debug vids:", os.path.join(OUT_DIR, "left_debug.avi"), "and", os.path.join(OUT_DIR, "right_debug.avi"))
    print(f"Stats: frames={frame}, valid_pairs={ok_pairs}, valid_3d={ok_3d}")


if __name__ == "__main__":
    main()
