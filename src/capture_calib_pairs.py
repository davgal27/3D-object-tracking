import os
import cv2
from stereo_image_source import ImageSource

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# -------------------------
# Configuration
# -------------------------

# Where we will save the calibration images (one folder per camera id)
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "calib")

# Manual camera settings 
EXPOSURE_US = 20000
GAIN = 0.0
FORCE_FULL_FRAME = True

# How long we wait to receive frames
TIMEOUT_S = 2.0

# 0 = no limit
TARGET_PAIRS = 0
# -------------------------

# Create a folder if it does not exist
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

   # Clear queued frames so we always use the newest images
def drain_queue(src) -> None:
    """Drop queued frames so each grab uses the most recent images."""
    q = getattr(src, "q", None)
    if q is None:
        return
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass

# Open ImageSource with manual exposure/gain
def open_source():
    """Open ImageSource with settings; fall back if the version does not support kwargs."""
    try:
        return ImageSource(
            use_auto_exposure=False,
            use_auto_gain=False,
            exposure_time_us=EXPOSURE_US,
            gain=GAIN,
            force_full_frame=FORCE_FULL_FRAME,
            verbose=True,
        )
    except TypeError:
        return ImageSource()


def get_images(src, timeout_s: float):
    """Support different ImageSource.get_images() signatures."""
    try:
        return src.get_images(timeout_s=timeout_s)
    except TypeError:
        try:
            return src.get_images(timeout_s)
        except TypeError:
            return src.get_images()

# Trigger the cameras and return one frame per camera id
def grab_one_per_camera(src, timeout_s: float):
    """
    Trigger cameras and grab one frame per camera.
    Returns: {cam_id_str: (timestamp, image)}
    """
    drain_queue(src)
    src.trigger_cameras()

    frames = {}
    for cam_id, ts, img in get_images(src, timeout_s):
        frames[str(cam_id)] = (ts, img)
    return frames


def main():
    ensure_dir(OUT_DIR)
    src = open_source() # Open the stereo camera source
   
    # Try one grab to see if we have at least two cameras
    frames = grab_one_per_camera(src, TIMEOUT_S)
    if len(frames) < 2:
        print("Error: did not receive frames from at least two cameras.")
        try:
            src.close()
        except Exception:
            pass
        return

    cam_ids = sorted(frames.keys())
    print("Cameras (cam_id):", cam_ids)

    cam_dirs = {}
    for cid in cam_ids:
        d = os.path.join(OUT_DIR, cid)
        ensure_dir(d)
        cam_dirs[cid] = d

        # Create a preview window for each camera so we can see what we take 
        cv2.namedWindow(cid, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cid, 960, 540)
    
    idx = 1  # Image pair counter
    print("\nControls:")
    print("  s      -> save stereo pair (one image per camera)")
    print("  q/ESC  -> quit\n")

    while True:
        frames = grab_one_per_camera(src, TIMEOUT_S)

        for cid in cam_ids:
            if cid not in frames:
                continue

            ts, img = frames[cid]

            # Sharpening for preview (does not affect saved images)
            blur = cv2.GaussianBlur(img, (0, 0), 1.5)
            img_show = cv2.addWeighted(img, 1.6, blur, -0.6, 0)

            focus = cv2.Laplacian(img, cv2.CV_64F).var()
            mn, mx = int(img.min()), int(img.max())  # Show brightness range to detect over/under exposure

            #some info on the preview image
            cv2.putText(
                img_show,
                f"idx={idx:03d}  focus={focus:.0f}  min/max={mn}/{mx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                255,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(cid, img_show)

        # Read keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        if key == ord("s"): # for saving 
            frames = grab_one_per_camera(src, TIMEOUT_S)

            missing = [cid for cid in cam_ids if cid not in frames]
            if missing:
                print("Error: missing cameras during save:", missing)
                continue

            ts_ref = max(frames[cid][0] for cid in cam_ids)
            stamp = f"{ts_ref:.3f}"

            # Save one png per camera with same idx + timestamp
            for cid in cam_ids:
                ts, img = frames[cid]
                fn = os.path.join(cam_dirs[cid], f"{idx:03d}_{stamp}.png")
                if cv2.imwrite(fn, img):
                    print("Saved:", fn)
                else:
                    print("Error: failed to write:", fn)

            idx += 1
            if TARGET_PAIRS and idx > TARGET_PAIRS:
                print("Reached target number of pairs. Exiting.")
                break

    try:
        src.close()
    except Exception:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

