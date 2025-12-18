import os
import csv
import time
import cv2
import numpy as np

from stereo_image_source import ImageSource

# =========================
# SETTINGS
# =========================
OUT_DIR_BASE = os.path.join("..", "data", "video")

DURATION_S   = 12          # Real time capture duration (seconds)
SHOW_PREVIEW = False       # Preview can reduce FPS a lot (keep False for best performance)

EXPOSURE_US = 20000
GAIN        = 0.0
FORCE_FULL_FRAME = True

TIMEOUT_S   = 0.35         # Smaller timeout reduces long stalls

FOURCC = "MJPG"            # AVI + MJPG is robust

# If input frames are 2D (H,W) and come from a color camera, they are usually Bayer.
# Pick a Bayer pattern; if colors look wrong, change to BG/GR/GB.
BAYER_PATTERN = cv2.COLOR_BayerRG2BGR  # try RG first

# Buffer safety
MAX_BUFFER_FRAMES = 7000
# =========================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def make_unique_run_dir(base_dir: str) -> str:
    """Create a unique timestamped output folder (no overwrite)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, stamp)
    if not os.path.exists(run_dir):
        return run_dir
    k = 1
    while True:
        candidate = os.path.join(base_dir, f"{stamp}_{k:02d}")
        if not os.path.exists(candidate):
            return candidate
        k += 1


def safe_get_images(src, timeout_s: float):
    """Handle possible ImageSource.get_images() signature variants."""
    try:
        return src.get_images(timeout_s=timeout_s)
    except TypeError:
        try:
            return src.get_images(timeout_s)
        except TypeError:
            return src.get_images()


def open_imagesource():
    """Open ImageSource with manual exposure/gain if supported."""
    try:
        return ImageSource(
            use_auto_exposure=False,
            use_auto_gain=False,
            exposure_time_us=EXPOSURE_US,
            gain=GAIN,
            force_full_frame=FORCE_FULL_FRAME,
            verbose=True
        )
    except TypeError:
        return ImageSource()


def raw_to_bgr(raw: np.ndarray, bayer_pattern: int) -> np.ndarray:
    """
    Convert captured frame to BGR for VideoWriter.
    This is called AFTER capture (offline) to keep capture fast.

    Cases:
      - raw is already BGR (H,W,3): return as-is
      - raw is 2D:
          * could be Bayer (color sensor) -> debayer
          * could be Mono8 -> gray->BGR (will look grayscale, not true color)
    """
    if raw.ndim == 3 and raw.shape[2] == 3:
        return raw

    if raw.ndim != 2:
        raise ValueError(f"Unexpected image shape: {raw.shape}")

    # Try debayer first (for color sensors outputting Bayer)
    # If it is actually Mono8, the result will look "weird"; then switch to gray->BGR by changing mode.
    # In practice for color Basler, 2D is almost always Bayer.
    try:
        return cv2.cvtColor(raw, bayer_pattern)
    except cv2.error:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)


def main():
    ensure_dir(OUT_DIR_BASE)
    run_dir = make_unique_run_dir(OUT_DIR_BASE)
    ensure_dir(run_dir)

    src = open_imagesource()

    # --- Initial grab to discover camera IDs and image size
    src.trigger_cameras()
    images = safe_get_images(src, TIMEOUT_S)
    if not images or len(images) < 2:
        print("❌ Failed to get frames from 2 cameras on startup.")
        try:
            src.close()
        except Exception:
            pass
        return

    cam_ids = sorted([str(cam_id) for cam_id, ts, img in images])
    cam0, cam1 = cam_ids[0], cam_ids[1]
    print("✅ Cameras:", cam_ids)

    sample = images[0][2]
    H, W = sample.shape[:2]
    print(f"✅ Frame size: {W}x{H}")
    print(f"✅ Sample frame shape: {sample.shape} dtype={sample.dtype}")
    print("Tip: If sample is 2D and colors look wrong, change BAYER_PATTERN (RG/BG/GR/GB).")

    # --- Buffers: store RAW frames only (fast)
    buf0_raw, buf1_raw = [], []
    pair_times_pc = []

    # timestamps log: log attempts, store frames only for ok_pair
    ts_rows = []
    frame_idx = 0
    stop = False

    print(f"\nCapturing RAW for {DURATION_S}s (real time). Encoding COLOR after capture.\n")

    t_start_pc = time.perf_counter()
    t_end_pc = t_start_pc + float(DURATION_S)

    while True:
        now_pc = time.perf_counter()
        if now_pc >= t_end_pc or stop:
            break
        if frame_idx >= MAX_BUFFER_FRAMES:
            print("⚠️ Reached MAX_BUFFER_FRAMES, stopping early.")
            break

        src.trigger_cameras()
        trig_wall = time.time()

        images = safe_get_images(src, TIMEOUT_S)
        frames = {str(cam_id): (float(ts), img) for cam_id, ts, img in images} if images else {}
        cap_wall = time.time()

        ok_pair = int((cam0 in frames) and (cam1 in frames))

        if ok_pair:
            ts0, img0 = frames[cam0]
            ts1, img1 = frames[cam1]

            # Ensure same size
            if (img0.shape[1], img0.shape[0]) != (W, H):
                img0 = cv2.resize(img0, (W, H), interpolation=cv2.INTER_AREA)
            if (img1.shape[1], img1.shape[0]) != (W, H):
                img1 = cv2.resize(img1, (W, H), interpolation=cv2.INTER_AREA)

            buf0_raw.append(img0.copy())
            buf1_raw.append(img1.copy())
            pair_times_pc.append(time.perf_counter())

            frame_idx += 1
            ts_rows.append([
                frame_idx,
                f"{cap_wall:.6f}",
                f"{trig_wall:.6f}",
                cam0, f"{ts0:.9f}",
                cam1, f"{ts1:.9f}",
                1
            ])

            if SHOW_PREVIEW:
                # Preview minimal work: show grayscale to keep FPS high
                g0 = img0 if img0.ndim == 2 else cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                g1 = img1 if img1.ndim == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                cv2.imshow(cam0, g0)
                cv2.imshow(cam1, g1)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("Stopped by user.")
                    stop = True
        else:
            ts0 = frames[cam0][0] if (cam0 in frames) else float("nan")
            ts1 = frames[cam1][0] if (cam1 in frames) else float("nan")
            ts_rows.append([
                frame_idx,
                f"{cap_wall:.6f}",
                f"{trig_wall:.6f}",
                cam0, f"{ts0:.9f}" if np.isfinite(ts0) else "nan",
                cam1, f"{ts1:.9f}" if np.isfinite(ts1) else "nan",
                0
            ])

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    try:
        src.close()
    except Exception:
        pass

    # Always save timestamps.csv
    ts_csv = os.path.join(run_dir, "timestamps.csv")
    with open(ts_csv, "w", newline="", encoding="utf-8") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow([
            "frame_idx",
            "cap_wall_time",
            "trigger_wall_time",
            "cam0_id", "cam0_ts",
            "cam1_id", "cam1_ts",
            "ok_pair"
        ])
        wcsv.writerows(ts_rows)

    n = len(buf0_raw)
    if n == 0:
        print("❌ No stereo pairs captured. Only timestamps saved:", ts_csv)
        return

    # Compute FPS based on accepted pair times (stable)
    if len(pair_times_pc) >= 2:
        elapsed_pairs = max(1e-6, pair_times_pc[-1] - pair_times_pc[0])
        fps_out = max(1.0, (n - 1) / elapsed_pairs)
    else:
        fps_out = 1.0

    print(f"✅ Captured stereo pairs: {n}")
    print(f"✅ Writing COLOR videos with FPS = {fps_out:.2f}")

    # Encode AFTER capture
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    out0 = os.path.join(run_dir, f"{cam0}_color.avi")
    out1 = os.path.join(run_dir, f"{cam1}_color.avi")

    vw0 = cv2.VideoWriter(out0, fourcc, float(fps_out), (W, H))
    vw1 = cv2.VideoWriter(out1, fourcc, float(fps_out), (W, H))
    if not vw0.isOpened() or not vw1.isOpened():
        raise RuntimeError("VideoWriter failed. Try FOURCC='XVID'.")

    for i in range(n):
        bgr0 = raw_to_bgr(buf0_raw[i], BAYER_PATTERN)
        bgr1 = raw_to_bgr(buf1_raw[i], BAYER_PATTERN)
        vw0.write(bgr0)
        vw1.write(bgr1)

    vw0.release()
    vw1.release()

    print("\n✅ Saved run:", run_dir)
    print("✅ Video 0:", out0)
    print("✅ Video 1:", out1)
    print("✅ timestamps.csv:", ts_csv)

    print("\nIf colors look wrong, change BAYER_PATTERN:")
    print(" - cv2.COLOR_BayerBG2BGR")
    print(" - cv2.COLOR_BayerGR2BGR")
    print(" - cv2.COLOR_BayerGB2BGR")


if __name__ == "__main__":
    main()
