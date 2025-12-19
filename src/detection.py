import cv2
import numpy as np
import csv
import os

LEFT_VIDEO_PATH = "data/video/20251218_002232/rectified/267601CA2FC7_rect.avi"
RIGHT_VIDEO_PATH = "data/video/20251218_002232/rectified/267601CA2FC6_rect.avi"
OUTPUT_CSV = "outputs/stereo_centroids.csv"
# this range of orange allowed for only the ball and its reflection to be detected on the table 
LOWER_ORANGE = np.array([12, 150, 100]) 
UPPER_ORANGE = np.array([23, 255, 255])
MIN_CONTOUR_AREA = 200
SHOW_DEBUG = True
FRAME_START = 1
DISPLAY_WIDTH = 800  # width for each video on screen


os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

capL = cv2.VideoCapture(LEFT_VIDEO_PATH)
capR = cv2.VideoCapture(RIGHT_VIDEO_PATH)

if not capL.isOpened() or not capR.isOpened():
    raise IOError("Could not open one or both videos")

capL.set(cv2.CAP_PROP_POS_FRAMES, FRAME_START)
capR.set(cv2.CAP_PROP_POS_FRAMES, FRAME_START)
frame_idx = FRAME_START

# The following method was my way of solving for the reflection of the ball being almost the same colour as the ball
# since this was the case, simply creating a mask for colour was not going to be anough
# therefore what i did was increase the range above for orange to make sure the reflection detection is as smooth as possible
# Then, create a bounding box over the whole detected contour.
# Since the reflection and the ball grew proportionally according to where they are in the Z- axis, and
# the reflection and the ball had very close to the same sizes, it was appropriate to calculate the centroid by
# getting the bounding box, dividing it in half, and calculating the centroid in only the top half of it, which would be the ball. 

def detect_top_half_centroid(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE) # idea for using a mask from cv assignments
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None, None, None, mask

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_CONTOUR_AREA:
        return None, None, None, None, None, None, mask

    x, y, w, h = cv2.boundingRect(c)
    top_half_mask = np.zeros_like(mask)
    top_half_mask[y:y+h//2, x:x+w] = mask[y:y+h//2, x:x+w]

    M = cv2.moments(top_half_mask)
    if M["m00"] == 0:
        return None, None, x, y, w, h, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy, x, y, w, h, mask

with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["frame", "xL", "yL", "xR", "yR"])

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break

        xL, yL, xLb, yLb, wL, hL, maskL = detect_top_half_centroid(frameL)
        xR, yR, xRb, yRb, wR, hR, maskR = detect_top_half_centroid(frameR)

        # Write CSV
        out_xL = xL if xL is not None else -1
        out_yL = yL if yL is not None else -1
        out_xR = xR if xR is not None else -1
        out_yR = yR if yR is not None else -1
        writer.writerow([frame_idx, out_xL, out_yL, out_xR, out_yR])

        if SHOW_DEBUG:
            # Resize masks to DISPLAY_WIDTH
            hL_new = int(maskL.shape[0] * DISPLAY_WIDTH / maskL.shape[1])
            hR_new = int(maskR.shape[0] * DISPLAY_WIDTH / maskR.shape[1])
            maskL_disp = cv2.resize(maskL, (DISPLAY_WIDTH, hL_new))
            maskR_disp = cv2.resize(maskR, (DISPLAY_WIDTH, hR_new))

            # Convert masks to BGR so we can draw colored centroids and boxes
            maskL_disp = cv2.cvtColor(maskL_disp, cv2.COLOR_GRAY2BGR)
            maskR_disp = cv2.cvtColor(maskR_disp, cv2.COLOR_GRAY2BGR)

            # Draw centroid and bounding box on masks
            if xL is not None:
                cv2.circle(maskL_disp, (int(xL * DISPLAY_WIDTH / maskL.shape[1]),
                                        int(yL * hL_new / maskL.shape[0])), 5, (0, 255, 0), -1)
                cv2.rectangle(maskL_disp,
                              (int(xLb * DISPLAY_WIDTH / maskL.shape[1]), int(yLb * hL_new / maskL.shape[0])),
                              (int((xLb + wL) * DISPLAY_WIDTH / maskL.shape[1]), int((yLb + hL) * hL_new / maskL.shape[0])),
                              (0, 0, 255), 2)

            if xR is not None:
                cv2.circle(maskR_disp, (int(xR * DISPLAY_WIDTH / maskR.shape[1]),
                                        int(yR * hR_new / maskR.shape[0])), 5, (0, 255, 0), -1)
                cv2.rectangle(maskR_disp,
                              (int(xRb * DISPLAY_WIDTH / maskR.shape[1]), int(yRb * hR_new / maskR.shape[0])),
                              (int((xRb + wR) * DISPLAY_WIDTH / maskR.shape[1]), int((yRb + hR) * hR_new / maskR.shape[0])),
                              (0, 0, 255), 2)

            # Side by side
            combined = np.hstack([maskL_disp, maskR_disp])
            cv2.imshow("Masks side by side", combined)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        frame_idx += 1

capL.release()
capR.release()
cv2.destroyAllWindows()
print(f"Stereo detection finished. Results saved to {OUTPUT_CSV}")
