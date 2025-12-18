import cv2
import numpy as np
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== File paths ======
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CSV_INPUT  = os.path.join(PROJECT_ROOT, "outputs", "stereo_centroids.csv")
JSON_CALIB = os.path.join(PROJECT_ROOT, "outputs", "calibration_outputs.json")
CSV_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "xyz_frame.csv")

# ====== Load calibration ======
with open(JSON_CALIB) as f:
    calib = json.load(f)

# Projection matrices P1 and P2
P1 = np.array(calib["P1"]).reshape(3, 4)
P2 = np.array(calib["P2"]).reshape(3, 4)

# ====== Read CSV and triangulate ======
output_rows = []

with open(CSV_INPUT) as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame = int(row["frame"])
        xL = float(row["xL"])
        yL = float(row["yL"])
        xR = float(row["xR"])
        yR = float(row["yR"])

        # Skip missing detections
        if xL == -1 or xR == -1:
            continue

        # Points need to be 2xN in float32
        ptL = np.array([[xL], [yL]], dtype=np.float32)
        ptR = np.array([[xR], [yR]], dtype=np.float32)

        # Triangulate
        point_4d = cv2.triangulatePoints(P1, P2, ptL, ptR)

        # Convert from homogeneous to 3D
        X = point_4d[0][0] / point_4d[3][0]
        Y = point_4d[1][0] / point_4d[3][0]
        Z = point_4d[2][0] / point_4d[3][0]

        Z = -Z

        output_rows.append([frame, X, Y, Z])

# ====== Save to CSV ======
with open(CSV_OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "X", "Y", "Z"])
    writer.writerows(output_rows)

print(f"Triangulated 3D points saved to {CSV_OUTPUT}")
