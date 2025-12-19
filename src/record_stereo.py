import os
import csv
import time
import cv2
import numpy as np
from stereo_image_source import ImageSource

# Record a short stereo "video" by grabbing pairs and then encoding to AVI at the end.

# -------------------------
# Settings
# -------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUT_DIR_BASE = os.path.join(PROJECT_ROOT, "data", "video")

DURATION_S = 12 # How long to record
SHOW_PREVIEW = False # If True, show live preview windows ( reduces the  performance so we put it false )

# Manual camera settings (if ImageSource supports these args)
EXPOSURE_US = 20000
GAIN = 0.0
FORCE_FULL_FRAME = True

TIMEOUT_S = 0.35 # Timeout waiting for frames after trigger
FOURCC = "MJPG" # code foe avi writing 

BAYER_PATTERN = cv2.COLOR_BayerRG2BGR
MAX_BUFFER_FRAMES = 7000 # Safety limit so we don’t keep buffering 
# -------------------------

 #Make a new timestamp folder name so each run is separate and we save all the videos 
def unique_run_dir(base_dir: str) -> str:
    """Create a new timestamped output folder without overwriting existing runs."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, stamp)
    if not os.path.exists(run_dir):
        return run_dir
    k = 1
    while True:
        run_dir = os.path.join(base_dir, f"{stamp}_{k:02d}")
        if not os.path.exists(run_dir):
            return run_dir
        k += 1


def main():
    os.makedirs(OUT_DIR_BASE, exist_ok=True)
    run_dir = unique_run_dir(OUT_DIR_BASE) # Create a unique folder for this recording 
    os.makedirs(run_dir, exist_ok=True)

    # Open ImageSource 
    try:
        src = ImageSource(
            use_auto_exposure=False,
            use_auto_gain=False,
            exposure_time_us=EXPOSURE_US,
            gain=GAIN,
            force_full_frame=FORCE_FULL_FRAME,
            verbose=True,
        )
    except TypeError:
        src = ImageSource()

    # get_images() compatibility wrapper (keyword/positional/none)
    def get_images(timeout_s: float):
        try:
            return src.get_images(timeout_s=timeout_s)
        except TypeError:
            try:
                return src.get_images(timeout_s)
            except TypeError:
                return src.get_images()

    # Startup grab: to find camera IDs and frame size
    src.trigger_cameras()
    images = get_images(TIMEOUT_S)
    if not images or len(images) < 2:
        print("Error: failed to get frames from 2 cameras on startup.")
        try:
            src.close()
        except Exception:
            pass
        return

    cam_ids = sorted(str(cam_id) for cam_id, _, _ in images)
    cam0, cam1 = cam_ids[0], cam_ids[1]

    sample = images[0][2]
    H, W = sample.shape[:2]

    print("Cameras:", cam_ids)
    print(f"Frame size: {W}x{H} | sample shape={sample.shape} dtype={sample.dtype}")

    # Buffers for raw frames (we store raw frames first, encode later)
    buf0, buf1 = [], []
    pair_times = [] # For FPS estimation
    ts_rows = []  # Rows to write in timestamps.csv
    frame_idx = 0  # Counter for accepted frames
    stop = False

    t_end = time.perf_counter() + float(DURATION_S)

    # capture loop
    while not stop and time.perf_counter() < t_end and frame_idx < MAX_BUFFER_FRAMES:
        src.trigger_cameras()
        trig_wall = time.time()

        images = get_images(TIMEOUT_S)
        frames = {str(cid): (float(ts), img) for cid, ts, img in images} if images else {}
        cap_wall = time.time()

        ok = int((cam0 in frames) and (cam1 in frames))
        
        # Make sure both images have the same siz
        if ok:
            ts0, img0 = frames[cam0]
            ts1, img1 = frames[cam1]

            # Keep size consistent for VideoWriter
            if img0.shape[:2] != (H, W):
                img0 = cv2.resize(img0, (W, H), interpolation=cv2.INTER_AREA)
            if img1.shape[:2] != (H, W):
                img1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_AREA)

             # Store copies in RAM (fast capture, encoding happens later)
            buf0.append(img0.copy())
            buf1.append(img1.copy())
            pair_times.append(time.perf_counter())

            frame_idx += 1
            ts_rows.append([
                frame_idx, f"{cap_wall:.6f}", f"{trig_wall:.6f}",
                cam0, f"{ts0:.9f}", cam1, f"{ts1:.9f}", 1
            ])

            if SHOW_PREVIEW:
                g0 = img0 if img0.ndim == 2 else cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                g1 = img1 if img1.ndim == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                cv2.imshow(cam0, g0)
                cv2.imshow(cam1, g1)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop = True
        else:
            ts0 = frames[cam0][0] if cam0 in frames else float("nan")
            ts1 = frames[cam1][0] if cam1 in frames else float("nan")
            ts_rows.append([
                frame_idx, f"{cap_wall:.6f}", f"{trig_wall:.6f}",
                cam0, (f"{ts0:.9f}" if np.isfinite(ts0) else "nan"),
                cam1, (f"{ts1:.9f}" if np.isfinite(ts1) else "nan"),
                0
            ])

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    # Close camera source
    try:
        src.close()
    except Exception:
        pass

    # Always write timestamps.csv
    ts_csv = os.path.join(run_dir, "timestamps.csv")
    with open(ts_csv, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow([
            "frame_idx", "cap_wall_time", "trigger_wall_time",
            "cam0_id", "cam0_ts", "cam1_id", "cam1_ts", "ok_pair"
        ])
        wcsv.writerows(ts_rows)

    # how many accepted stereo pers we have 
    n = len(buf0)
    if n == 0:
        print("Error: no stereo pairs captured. Only timestamps saved:", ts_csv)
        return

    # FPS based on accepted pairs
    if len(pair_times) >= 2:
        elapsed = max(1e-6, pair_times[-1] - pair_times[0])
        fps_out = max(1.0, (n - 1) / elapsed)
    else:
        fps_out = 1.0

    # Setup VideoWriter for each camera
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    out0 = os.path.join(run_dir, f"{cam0}_color.avi")
    out1 = os.path.join(run_dir, f"{cam1}_color.avi")

    vw0 = cv2.VideoWriter(out0, fourcc, float(fps_out), (W, H))
    vw1 = cv2.VideoWriter(out1, fourcc, float(fps_out), (W, H))
    if not vw0.isOpened() or not vw1.isOpened():
        raise RuntimeError("VideoWriter failed. Try FOURCC='XVID'.")

    # Convert raw frames to BGR only during encoding
    for i in range(n):
        a, b = buf0[i], buf1[i]

        if a.ndim == 2:
            try:
                a = cv2.cvtColor(a, BAYER_PATTERN)
            except cv2.error:
                a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        if b.ndim == 2:
            try:
                b = cv2.cvtColor(b, BAYER_PATTERN)
            except cv2.error:
                b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

        # Write frame to the output video
        vw0.write(a)
        vw1.write(b)

    vw0.release()
    vw1.release()

    print("Saved run:", run_dir)
    print("Video 0:", out0)
    print("Video 1:", out1)
    print("timestamps.csv:", ts_csv)


if __name__ == "__main__":
    main()

