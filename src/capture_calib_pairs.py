import os
import time
import cv2

from stereo_image_source import ImageSource

# =========================
# ΡΥΘΜΙΣΕΙΣ
# =========================
OUT_DIR = os.path.join("..", "data", "calib")
EXPOSURE_US = 20000
GAIN = 0.0
FORCE_FULL_FRAME = True
TIMEOUT_S = 2.0

# πόσα stereo pairs θες να γράψεις (0 = χωρίς limit)
TARGET_PAIRS = 0
# =========================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def drain_queue(src):
    """Άδειασε την ουρά για να μη βλέπεις παλιά frames."""
    q = getattr(src, "q", None)
    if q is None:
        return
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass


def safe_get_images(src, timeout_s: float):
    """
    Υποστηρίζει και τις 3 μορφές:
      - get_images(timeout_s=...)
      - get_images(timeout_s)   (positional)
      - get_images()            (παλιά έκδοση)
    """
    try:
        return src.get_images(timeout_s=timeout_s)
    except TypeError:
        pass

    try:
        return src.get_images(timeout_s)
    except TypeError:
        pass

    return src.get_images()


def open_imagesource():
    """
    Υποστηρίζει ImageSource με kwargs ή χωρίς.
    """
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
        # Παλιά έκδοση ImageSource: δεν δέχεται kwargs
        return ImageSource()


def discover_cam_ids(frames_dict):
    return sorted(frames_dict.keys())


def grab_one_per_camera(src, timeout_s: float):
    """
    Trigger -> πάρε 1 frame ανά κάμερα.
    Επιστρέφει dict: {cam_id_str: (ts, img)}
    """
    drain_queue(src)
    src.trigger_cameras()

    images = safe_get_images(src, timeout_s)
    frames = {}
    for cam_id, ts, img in images:
        frames[str(cam_id)] = (ts, img)

    return frames


def main():
    ensure_dir(OUT_DIR)

    src = open_imagesource()

    # Πάρε 1 frame για να μάθουμε cam_ids
    frames = grab_one_per_camera(src, TIMEOUT_S)
    if len(frames) < 2:
        print("❌ Δεν πήρα frames από 2 κάμερες. Έλεγξε καλώδια/USB/driver.")
        try:
            src.close()
        except Exception:
            pass
        return

    cam_ids = discover_cam_ids(frames)
    print("✅ Cameras (cam_id):", cam_ids)

    # folders ανά cam_id
    cam_dirs = {}
    for cid in cam_ids:
        d = os.path.join(OUT_DIR, cid)
        ensure_dir(d)
        cam_dirs[cid] = d

        cv2.namedWindow(cid, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cid, 960, 540)

    idx = 1
    print("\nControls:")
    print("  s  -> save stereo pair (1 εικόνα ανά cam_id)")
    print("  q/ESC -> quit\n")

    while True:
        frames = grab_one_per_camera(src, TIMEOUT_S)

        # Preview
        for cid in cam_ids:
            if cid not in frames:
                continue
            ts, img = frames[cid]

            # sharpening preview
            blur = cv2.GaussianBlur(img, (0, 0), 1.5)
            img_show = cv2.addWeighted(img, 1.6, blur, -0.6, 0)

            fm = cv2.Laplacian(img, cv2.CV_64F).var()
            mn, mx = int(img.min()), int(img.max())

            cv2.putText(
                img_show,
                f"idx={idx:03d}  focus={fm:.0f}  min/max={mn}/{mx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA
            )
            cv2.imshow(cid, img_show)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        if key == ord("s"):
            # πάρε φρέσκο ζευγάρι την ώρα του save
            frames = grab_one_per_camera(src, TIMEOUT_S)

            missing = [cid for cid in cam_ids if cid not in frames]
            if missing:
                print("❌ Missing cameras on save:", missing, " -> ξαναπάτα s")
                continue

            ts_ref = max(frames[cid][0] for cid in cam_ids)
            stamp = f"{ts_ref:.3f}"

            for cid in cam_ids:
                ts, img = frames[cid]
                fn = os.path.join(cam_dirs[cid], f"{idx:03d}_{stamp}.png")
                ok = cv2.imwrite(fn, img)
                if ok:
                    print("✅ Saved:", fn)
                else:
                    print("Failed to write:", fn)

            idx += 1
            if TARGET_PAIRS and idx > TARGET_PAIRS:
                print("✅ Reached target pairs. Exiting.")
                break

    # cleanup
    try:
        src.close()
    except Exception:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
