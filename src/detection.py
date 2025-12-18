"""
Stereo Orange Ball Detection & Centroid Tracking

INPUTS:
- Left camera video (left_ball.mp4)
- Right camera video (right_ball.mp4)
- HSV color range for the orange ball (tune for lighting)

OUTPUTS:
- CSV file with synchronized detections:
  frame, xL, yL, xR, yR

NOTES:
- This file performs stereo detection ONLY (no calibration, no triangulation)
- Assumes videos are time-synchronized and same FPS
- Missing detections are written as -1, -1
"""

import cv2
import numpy as np
import csv

# ===================== USER PARAMETERS =====================
LEFT_VIDEO_PATH = "left_ball.mp4"
RIGHT_VIDEO_PATH = "right_ball.mp4"
OUTPUT_CSV = "stereo_centroids.csv"

# HSV range for orange ball (TUNE THESE)
LOWER_ORANGE = np.array([5, 150, 150])
UPPER_ORANGE = np.array([15, 255, 255])

MIN_CONTOUR_AREA = 200   # reject noise
SHOW_DEBUG = True        # show visualization

# ===========================================================

# Open videos
capL = cv2.VideoCapture(LEFT_VIDEO_PATH)
capR = cv2.VideoCapture(RIGHT_VIDEO_PATH)

if not capL.isOpened() or not capR.isOpened():
    raise IOError("Could not open one or both videos")

# Prepare CSV output
with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "xL", "yL", "xR", "yR"])

    frame_idx = 0

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()

        if not retL or not retR:
            break

        def detect_centroid(frame):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return None, None, mask

            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                return None, None, mask

            M = cv2.moments(c)
            if M["m00"] == 0:
                return None, None, mask

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy, mask

        # Detect in left and right frames
        xL, yL, maskL = detect_centroid(frameL)
        xR, yR, maskR = detect_centroid(frameR)

        # Handle missing detections
        out_xL = xL if xL is not None else -1
        out_yL = yL if yL is not None else -1
        out_xR = xR if xR is not None else -1
        out_yR = yR if yR is not None else -1

        writer.writerow([frame_idx, out_xL, out_yL, out_xR, out_yR])

        # Visualization
        if SHOW_DEBUG:
            if xL is not None:
                cv2.circle(frameL, (xL, yL), 5, (0, 255, 0), -1)
            if xR is not None:
                cv2.circle(frameR, (xR, yR), 5, (0, 255, 0), -1)

            top = np.hstack([
                cv2.cvtColor(maskL, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(maskR, cv2.COLOR_GRAY2BGR)
            ])
            bottom = np.hstack([frameL, frameR])
            combined = np.vstack([top, bottom])

            cv2.imshow("Masks (top) | Detections (bottom)", combined)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

capL.release()
capR.release()
cv2.destroyAllWindows()

print(f"Stereo detection finished. Results saved to {OUTPUT_CSV}")