# infer_toycar_yoloe_seg_video.py
import os
import sys
import shutil
import tempfile
from pathlib import Path
from collections import deque
import time

import cv2
import torch
import numpy as np
from ultralytics import YOLOE  # YOLOE (promptable segmentation)

# --------- SERIAL / ARDUINO SETTINGS ---------
try:
    import serial
except ImportError:
    serial = None

ARDUINO_PORT = "COM8"  # <-- CHANGE THIS to your Arduino port
ARDUINO_BAUD = 115200
SERIAL_TIMEOUT = 1.0  # seconds

# ==== USER SETTINGS ====
WEIGHTS_PATH = r"codes\AiModelTrain\yoloe-11s-seg (1).pt"  # promptable seg weights
IMGSZ = 640
CONF = 0.25
IOU = 0.5
DEVICE = "cpu"  # 0 for GPU/CUDA, "mps" for Apple Silicon GPU, or "cpu"
PROJECT = "runs/infer"
NAME = "toy_cars"
SAVE_ANNOTATED = True

# Path to your source (image OR video) - kept, but video uses webcam in infer_on_video ON PURPOSE
SOURCE_PATH = r"codes\WhatsApp Video 2025-12-30 at 12.05.23 AM.mp4"

# Path to irregular road regions config (polygons)
REGION_CONFIG_PATH = "codes/road_regions.json"

# ======================= SPEED / STABILITY SETTINGS (NUMBER 2) =======================
STRIDE = 1   # run inference every N frames (higher = faster, but less responsive)
WINDOW = 1   # aggregate decisions across last N inferred frames

# ======================= PROMPTS =======================
# Add your emergency prompts here if you want an emergency override.
# If you DON'T have emergency prompts, you can leave it empty or remove it.
ALIASES = {
    "car": [
        "arduino rc car",
        "arduino robot car",
        "diy arduino rc car",
        "robot car kit",
        "toy car",
        "small rc car",
        "mini robot car",
    ],
    "emergency_vehicle": [
        "toy ambulance",
        "toy police car",
        "toy fire truck",
        "ambulance",
        "police car",
        "fire truck",
    ],
}
# ===== Flatten prompts and build maps =====
CLASS_PROMPTS = []
CANON_NAME_MAP = {}   # idx -> canonical name, e.g. "arduino_car", "emergency_vehicle"
PROMPT_NAME_MAP = {}  # idx -> full prompt text

_idx = 0
for canonical_name, prompts in ALIASES.items():
    for p in prompts:
        CLASS_PROMPTS.append(p)
        CANON_NAME_MAP[_idx] = canonical_name
        PROMPT_NAME_MAP[_idx] = p
        _idx += 1


# ======================= SERIAL HELPERS =======================
def init_serial():
    """Initialize serial connection to Arduino — CRASH if it fails."""
    if serial is None:
        raise RuntimeError("pyserial is not installed — cannot use Arduino serial communication.")

    try:
        ser = serial.Serial(
            ARDUINO_PORT,
            ARDUINO_BAUD,
            timeout=SERIAL_TIMEOUT,
            write_timeout=SERIAL_TIMEOUT
        )
        time.sleep(2)  # give Arduino time to reset
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"[INFO] Opened serial port {ARDUINO_PORT} @ {ARDUINO_BAUD}")
        print(f"[INFO] ser.is_open = {ser.is_open}")
        return ser
    except Exception as e:
        raise RuntimeError(f"[FATAL] Could not open serial port {ARDUINO_PORT}: {e}")


def send_mode_to_arduino(ser, mode):
    if ser is None:
        return
    try:
        msg = f"{mode}"
        ser.write(msg.encode("ascii"))
        ser.flush()
        print(f"[SERIAL] Sent mode: {mode}")
    except Exception as e:
        print(f"[WARN] Failed to send mode over serial: {e}")


# ======================= UTILITIES =======================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def deep_purge_mobileclip(verbose=True):
    roots = set()
    roots.update([
        Path.home() / ".cache",
        Path.home() / "Library" / "Caches",
        Path(os.environ.get("TMPDIR", "/tmp")),
    ])
    for base in [sys.prefix, sys.base_prefix, sys.exec_prefix]:
        roots.add(Path(base))
    for p in sys.path:
        try:
            pp = Path(p)
            if "site-packages" in p or "dist-packages" in p:
                roots.add(pp)
        except Exception:
            pass

    patterns = ["**/mobileclip*.ts", "**/mobileclip*/*.ts", "**/mobileclip*"]
    removed = 0
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            for hit in root.glob(pat):
                try:
                    if hit.is_file():
                        hit.unlink()
                        removed += 1
                        if verbose:
                            print(f"[CACHE] removed file: {hit}")
                    elif hit.is_dir():
                        shutil.rmtree(hit, ignore_errors=True)
                        removed += 1
                        if verbose:
                            print(f"[CACHE] removed dir : {hit}")
                except Exception as e:
                    if verbose:
                        print(f"[CACHE] warn: could not remove {hit}: {e}")
    if verbose and removed == 0:
        print("[CACHE] no mobileclip artifacts found to remove")


def prepare_mobileclip_asset(force_fresh_cache=False, validate_load=False):
    if force_fresh_cache:
        tmp_cache = tempfile.mkdtemp(prefix="ultra_cache_")
        os.environ["ULTRALYTICS_CACHE_DIR"] = tmp_cache
        os.environ["TORCH_HOME"] = tmp_cache
        print(f"[CACHE] Using fresh cache: {tmp_cache}")

    deep_purge_mobileclip(verbose=True)

    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except Exception:
        from ultralytics.utils import downloads as _dl
        attempt_download_asset = _dl.attempt_download_asset

    asset_name = "mobileclip_blt.ts"
    asset_path = attempt_download_asset(asset_name)
    print(f"[CACHE] MobileCLIP asset path: {asset_path}")

    if validate_load:
        try:
            _ = torch.jit.load(asset_path, map_location="cpu")
            print("[CACHE] MobileCLIP TorchScript validated OK.")
        except Exception as e:
            print("[WARN] Validation failed, removing asset and retrying once...\n", e)
            try:
                Path(asset_path).unlink(missing_ok=True)
            except Exception:
                pass
            asset_path = attempt_download_asset(asset_name)
            _ = torch.jit.load(asset_path, map_location="cpu")
            print("[CACHE] MobileCLIP TorchScript validated OK after re-download.")


def get_text_embeddings_with_retry(model, texts):
    prepare_mobileclip_asset(force_fresh_cache=False, validate_load=False)
    try:
        return model.get_text_pe(texts)
    except Exception as e:
        print("[WARN] get_text_pe failed after prepare; hard reset caches and retry once...\n", e)
        prepare_mobileclip_asset(force_fresh_cache=False, validate_load=False)
        return model.get_text_pe(texts)


# ======================= ROAD REGION HELPERS =======================
def load_road_config(config_path: str = REGION_CONFIG_PATH):
    """
    Expected format:
    {
      "frame_width": W,
      "frame_height": H,
      "roads": [
        {"name": "road_1", "points": [[x1, y1], [x2, y2], ...]},
        ...
      ]
    }
    """
    import json
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Road region config '{config_path}' not found. "
            f"Create it first with the region-definition script."
        )
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg


def scale_polygon(points, src_w, src_h, dst_w, dst_h):
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    return [(int(px * sx), int(py * sy)) for (px, py) in points]


def get_road_regions(frame, cfg):
    h, w = frame.shape[:2]
    src_w = cfg["frame_width"]
    src_h = cfg["frame_height"]

    regions = []
    for road in cfg["roads"]:
        name = road["name"]
        pts_src = [(p[0], p[1]) for p in road["points"]]
        pts = scale_polygon(pts_src, src_w, src_h, w, h)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = max(min(xs), 0), min(max(xs), w - 1)
        y1, y2 = max(min(ys), 0), min(max(ys), h - 1)

        regions.append({
            "name": name,
            "poly": pts,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return regions


def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    if n < 3:
        return False

    px1, py1 = poly[0]
    for i in range(n + 1):
        px2, py2 = poly[i % n]
        if y > min(py1, py2):
            if y <= max(py1, py2):
                if x <= max(px1, px2):
                    if py1 != py2:
                        xinters = (y - py1) * (px2 - px1) / float(py2 - py1) + px1
                    if px1 == px2 or x <= xinters:
                        inside = not inside
        px1, py1 = px2, py2
    return inside


# ======================= FAST INFERENCE ON VIDEO (NUMBER 1 + NUMBER 2) =======================
def infer_on_video(model: YOLOE, video_path: str, out_dir: Path, ser=None):
    """
    NUMBER 1: Run YOLOE ONCE per inference step on the FULL FRAME (not 4 crops).
              Then assign detections to roads via point-in-polygon.

    NUMBER 2: Increase effective FPS by:
              - STRIDE: only run inference every STRIDE frames
              - WINDOW aggregation:
                    * car count per road = max over last WINDOW inferred frames
                    * emergency = OR over last WINDOW inferred frames
              - Reuse last annotated frame for skipped frames
    """
    print(f"\n[INFO] Inference on video (webcam index 1 ON PURPOSE): {video_path}")
    if not os.path.isfile(video_path):
        print(f"[WARN] Video file path does not exist (ok if webcam). Path: {video_path}")

    # Load road configuration
    try:
        road_cfg = load_road_config(REGION_CONFIG_PATH)
        print(f"[INFO] Loaded road regions from {REGION_CONFIG_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not load road config: {e}")
        return

    ensure_dir(out_dir)

    cap = cv2.VideoCapture(0)  # <-- ON PURPOSE (as you said)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam index 1.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 25.0

    base = Path(video_path).stem if video_path else "webcam"
    out_path = out_dir / f"{base}_toycar.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print(f"[INFO] Writing annotated video to: {out_path.resolve()}")
    print("[INFO] Press 'q' in the video window to stop early.")

    # HISTORY for aggregation
    history_counts = deque(maxlen=WINDOW)      # each item: dict road_name->count
    history_emergency = deque(maxlen=WINDOW)   # each item: bool

    last_annotated_frame = None
    last_mode_sent = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No more frames (or camera disconnected).")
            break

        frame_idx += 1

        do_infer = (frame_idx % STRIDE == 0)

        # If skipping inference, reuse last annotated frame
        if not do_infer and last_annotated_frame is not None:
            out_writer.write(last_annotated_frame)
            cv2.imshow("Arduino Car Detection - Video (4 Irregular Roads)", last_annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 'q' pressed, stopping early.")
                break
            continue

        # Build road regions for this frame
        road_regions = get_road_regions(frame, road_cfg)
        road_counts = {r["name"]: 0 for r in road_regions}

        # Run YOLOE ONCE on full frame (NUMBER 1)
        results = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            save=False,
            verbose=False,
        )

        # Start annotated frame
        annotated_for_save = frame.copy()

        # Default: no emergency this frame (set true if any emergency detection appears)
        emergency_this_frame = False

        if results and len(results) > 0:
            r = results[0]

            # For drawing, use prompt labels
            r.names = PROMPT_NAME_MAP
            ann = r.plot()  # draw all detections once
            if ann is not None:
                annotated_for_save = ann

            # Now compute counts per road by checking each detection center against polygons
            if r.boxes is not None and len(r.boxes) > 0:
                boxes_xyxy = r.boxes.xyxy.detach().cpu().numpy()  # (N,4)
                clses = r.boxes.cls.detach().cpu().numpy().astype(int)  # (N,)

                for (bx1, by1, bx2, by2), cls_id in zip(boxes_xyxy, clses):
                    cx = (bx1 + bx2) / 2.0
                    cy = (by1 + by2) / 2.0

                    # emergency flag if canonical class is "emergency_vehicle"
                    canon = CANON_NAME_MAP.get(int(cls_id), "")
                    if canon == "emergency_vehicle":
                        emergency_this_frame = True

                    # increment first road polygon that contains it
                    for region in road_regions:
                        if point_in_polygon(cx, cy, region["poly"]):
                            road_counts[region["name"]] += 1
                            break

        # Push into history (NUMBER 2 aggregation)
        history_counts.append(road_counts)
        history_emergency.append(emergency_this_frame)

        # Aggregate:
        road_names = [r["name"] for r in road_regions]
        agg_counts = {
            rn: max(d.get(rn, 0) for d in history_counts) if history_counts else 0
            for rn in road_names
        }
        agg_emergency = any(history_emergency) if history_emergency else False

        # Draw polygons + aggregated counts on annotated frame
        for region in road_regions:
            poly = region["poly"]
            pts_arr = np.array(poly, dtype=np.int32)
            cv2.polylines(annotated_for_save, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=2)

            label_x, label_y = poly[0]
            cv2.putText(
                annotated_for_save,
                f'{region["name"]}: {agg_counts.get(region["name"], 0)}',
                (label_x + 5, label_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Decide mode (same logic, but based on aggregated counts)
        counts_in_order = [agg_counts[name] for name in road_names]

        # Default
        mode_to_send = 0

        # Emergency has highest priority
        if agg_emergency:
            mode_to_send = 5
        else:
            # Check if all roads have equal counts
            if len(set(counts_in_order)) == 1:
                mode_to_send = 0
            else:
                max_count = max(counts_in_order)
                roads_with_max = [i for i, c in enumerate(counts_in_order) if c == max_count]

                # If only ONE road has the maximum, select that road
                if len(roads_with_max) == 1:
                    # +1 because road index starts from 0
                    mode_to_send = roads_with_max[0] + 1
                else:
                    # Tie between multiple roads
                    mode_to_send = 0

        # Send only if changed
        if mode_to_send != last_mode_sent:
            send_mode_to_arduino(ser, mode_to_send)
            last_mode_sent = mode_to_send

        print(
            f"[FRAME {frame_idx}] "
            f"counts={road_counts} | agg={agg_counts} | "
            f"emergency={agg_emergency} | mode={mode_to_send}"
        )

        # Write + show
        out_writer.write(annotated_for_save)
        cv2.imshow("Arduino Car Detection - Video (4 Irregular Roads)", annotated_for_save)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' pressed, stopping early.")
            break


        # Cache for reuse on skipped frames
        last_annotated_frame = annotated_for_save.copy()

        if frame_idx % 50 == 0:
            print(f"[INFO] Processed {frame_idx} frames... (stride={STRIDE}, window={WINDOW})")

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Video processing complete.")


# ======================= (OPTIONAL) IMAGE INFERENCE KEPT =======================
def infer_on_image(model: YOLOE, image_path: str, out_dir: Path):
    print(f"\n[INFO] Inference on image: {image_path}")
    if not os.path.isfile(image_path):
        print(f"[WARN] File does not exist: {image_path}")
        return

    ensure_dir(out_dir)

    results = model.predict(
        source=image_path,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        save=False,
        verbose=True,
    )

    if not results:
        print("[INFO] No results returned.")
        return

    base = Path(image_path).stem
    ext = Path(image_path).suffix or ".jpg"
    saved_any = False

    for i, r in enumerate(results):
        r.names = PROMPT_NAME_MAP
        annotated_for_save = r.plot()
        if annotated_for_save is None:
            continue

        suffix = "" if len(results) == 1 else f"_{i}"
        out_path = out_dir / f"{base}_toycar{suffix}{ext}"

        cv2.imwrite(str(out_path), annotated_for_save)
        print(f"[INFO] Saved annotated image: {out_path}")
        saved_any = True

        cv2.imshow("Toy Car Detection - Image (PROMPT LABELS)", annotated_for_save)
        print("[INFO] Press any key in the image window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not saved_any:
        print("[INFO] No detections above threshold; nothing saved.")


# ======================= MAIN =======================
def main():
    # 1) Open serial to Arduino
    try:
        ser = init_serial()
    except Exception as e:
        print(e)
        sys.exit(1)

    # 2) Load promptable seg model once
    print("[INFO] Loading YOLOE promptable segmentation model...")
    model = YOLOE(WEIGHTS_PATH)

    # 3) Set promptable classes
    print("[INFO] Setting prompt classes:")
    for t in CLASS_PROMPTS:
        print("   -", t)

    text_pe = get_text_embeddings_with_retry(model, CLASS_PROMPTS)
    model.set_classes(CLASS_PROMPTS, text_pe)

    # (Best-effort) set names globally to canonical labels
    try:
        if hasattr(model, "names"):
            model.names = CANON_NAME_MAP
        if hasattr(model, "model") and hasattr(model.model, "names"):
            model.model.names = CANON_NAME_MAP
    except Exception as e:
        print("[WARN] Could not set model names globally:", e)

    # Output directories
    output_root_images = Path(PROJECT) / NAME / "images"
    output_root_videos = Path(PROJECT) / NAME / "videos"
    ensure_dir(output_root_images)
    ensure_dir(output_root_videos)
    print(f"[INFO] Annotated images will be saved under: {output_root_images.resolve()}")
    print(f"[INFO] Annotated videos will be saved under: {output_root_videos.resolve()}")

    if not SOURCE_PATH:
        print("[ERROR] SOURCE_PATH is empty. Please set it to an image or video file.")
        return

    lower = SOURCE_PATH.lower()
    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".wmv")
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    if lower.endswith(video_exts):
        infer_on_video(model, SOURCE_PATH, output_root_videos, ser=ser)
    elif lower.endswith(image_exts):
        infer_on_image(model, SOURCE_PATH, output_root_images)
    else:
        print(f"[WARN] Could not determine file type from extension for: {SOURCE_PATH}")
        print("[INFO] Assuming video.")
        infer_on_video(model, SOURCE_PATH, output_root_videos, ser=ser)

    if ser is not None:
        ser.close()
        print("[INFO] Serial port closed.")


if __name__ == "__main__":
    main()