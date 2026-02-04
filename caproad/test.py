#infer_cameraAi.py
import os
import sys
import shutil
import tempfile
from pathlib import Path
import time
import json

import cv2
import torch
import numpy as np
from ultralytics import YOLOE  # YOLOE (promptable segmentation)
import time
import urllib.request
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import threading
# ---------- shared config ----------
##--------- SERIAL / ARDUINO SETTINGS ---------
try:
    import serial # type: ignore
except Exception:
    serial = None
# --- shared config (put at top of BOTH scripts or in a separate config.py) ---

# Who is allowed to trigger emergency mode 5?
#   "camera" → police car in image can send mode 5
#   "audio"  → siren in microphone can send mode 5
EMERGENCY_SOURCE = "audio"   # change to "audio" when you want YAMNet to control it
ARDUINO_PORT = "COM13"  # <-- CHANGE THIS to your Arduino port
ARDUINO_BAUD = 115200
SERIAL_TIMEOUT = 1.0  # seconds
# =============== YAMNet (audio siren detection) ===============
SAMPLE_RATE = 16000          # YAMNet expects 16kHz
CHUNK_SECONDS = 3            # record 3 seconds at a time
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS

SIREN_KEYWORDS = [
    "siren",
    "police car (siren)",
    "ambulance (siren)",
    "fire engine (siren)",
]
SIREN_THRESHOLD = 0.2        # probability threshold - tweak as needed

print("[INFO] Loading YAMNet model from TensorFlow Hub...")
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)
print("[INFO] YAMNet loaded.")

# Load YAMNet class names
labels_path = "caproad\yamnet_class_map.csv"
try:
    with open(labels_path, "r") as f:
        lines = f.read().strip().split("\n")
except FileNotFoundError:
    print("[INFO] Downloading YAMNet class map...")
    url = "https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv"
    urllib.request.urlretrieve(url, labels_path)
    with open(labels_path, "r") as f:
        lines = f.read().strip().split("\n")

# Third column is display_name
YAMNET_CLASS_NAMES = [l.split(",")[2] for l in lines]
print(f"[INFO] Loaded {len(YAMNET_CLASS_NAMES)} YAMNet classes.")
# ===============================================================
last_mode_sent = None  # to avoid sending the same mode every frame

def init_serial():
    """Initialize serial connection to Arduino — return None if unavailable."""
    if serial is None:
        print("[WARN] pyserial is not installed — Arduino serial will be disabled.")
        return None

    try:
        ser = serial.Serial(
            ARDUINO_PORT,
            ARDUINO_BAUD,
            timeout=SERIAL_TIMEOUT,
            write_timeout=SERIAL_TIMEOUT
        )
        time.sleep(2)  # give Arduino time to reset after opening port
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"[INFO] Opened serial port {ARDUINO_PORT} @ {ARDUINO_BAUD}")
        print(f"[INFO] ser.is_open = {ser.is_open}")
        return ser
    except Exception as e:
        print(f"[WARN] Could not open serial port {ARDUINO_PORT}: {e}")
        return None
# ----------------- YAMNet helpers -----------------
def is_siren_present(mean_scores, class_names, threshold=SIREN_THRESHOLD):
    """Check if any siren-related label passes the threshold."""
    best_label = None
    best_score = 0.0

    for idx, score in enumerate(mean_scores):
        name = class_names[idx]
        lower_name = name.lower()
        if any(k in lower_name for k in [k.lower() for k in SIREN_KEYWORDS]):
            if score > best_score:
                best_score = score
                best_label = name

    if best_label is not None and best_score >= threshold:
        return True, best_label, best_score

    return False, None, 0.0


def detect_siren_in_chunk(audio_chunk):
    """
    audio_chunk: 1D np.array of float32 audio at 16kHz, mono.
    """
    audio_tensor = tf.convert_to_tensor(audio_chunk, dtype=tf.float32)

    # YAMNet expects mono waveform. Shape: [num_samples]
    scores, embeddings, spectrogram = yamnet_model(audio_tensor)
    scores_np = scores.numpy()  # [time_frames, num_classes]

    # Average scores over time
    mean_scores = np.mean(scores_np, axis=0)

    # Check for siren presence
    found, label, score = is_siren_present(mean_scores, YAMNET_CLASS_NAMES)
    return found, label, score


def audio_loop(ser):
    """
    Runs in a background thread.
    If EMERGENCY_SOURCE == 'audio' and a siren is detected,
    send mode 5 to Arduino.
    """
    print("[AUDIO] Starting microphone siren detection loop...")
    print(f"[AUDIO] Recording every {CHUNK_SECONDS} seconds at {SAMPLE_RATE} Hz.")
    print("[AUDIO] Press Ctrl+C in the main process to stop.\n")

    while True:
        try:
            # Record audio chunk
            audio = sd.rec(
                frames=CHUNK_SAMPLES,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()  # block until recording is finished

            # audio shape: (CHUNK_SAMPLES, 1) -> make it (N,)
            audio_mono = audio[:, 0]

            found, label, score = detect_siren_in_chunk(audio_mono)

            if found:
                print(f"[AUDIO] *** SIREN DETECTED *** -> {label} (score={score:.3f})")
                if EMERGENCY_SOURCE == "audio":
                    send_mode_to_arduino(ser, 5)
            else:
                print("[AUDIO] No siren detected in this chunk.")

            time.sleep(0.1)  # small pause, optional

        except Exception as e:
            print(f"[AUDIO][ERROR] {e}")
            time.sleep(1.0)
# --------------------------------------------------
def send_mode_to_arduino(ser, mode):
    """Send a numeric mode to Arduino, but only if serial is available and mode changed."""
    global last_mode_sent
    if ser is None:
        return

    #Only send if changed
    if mode == last_mode_sent:
        return

    try:
        msg = f"{int(mode)}"
        ser.write(msg.encode("ascii"))
        ser.flush()
        last_mode_sent = mode
        print(f"[SERIAL] Sent mode: {mode}")
    except Exception as e:
        print(f"[WARN] Failed to send mode over serial: {e}")

#==== USER SETTINGS ====
WEIGHTS_PATH = r"AiModelTrain\yoloe-11s-seg (1).pt"  # promptable seg weights
IMGSZ = 640
CONF = 0.35
IOU = 0.5
DEVICE = "mps"  # 0 for GPU/CUDA, "mps" for Apple Silicon GPU, or "cpu"
PROJECT = "runs/infer"
NAME = "toy_cars"
SAVE_ANNOTATED = True  # (not heavily used but kept for compatibility)

#Path to your source (image OR video)
SOURCE_PATH = "WhatsApp Video 2025-12-01 at 00.35.53.mp4"

#Path to irregular road regions config (polygons)
REGION_CONFIG_PATH = "road_regions.json"

#======================= PROMPTS (TOY CARS + POLICE TOY CARS) =======================
ALIASES = {
    "toy_car": [
        "upright toy car", "toy car on its wheels", "toy car standing upright",
        "toy car viewed from above", "toy car top view", "toy car side view",
        "toy car seen from the side", "toy car seen from the front", "toy car front view", "toy car angled view",
        "small plastic toy car", "diecast toy car", "miniature toy car", "kid's toy car", "tiny toy car",
        "toy race car", "colored toy car", "toy sports car", "toy sedan", "toy hatchback",
        "white toy car", "red toy car", "blue toy car", "yellow toy car",
        "roof of toy car", "roof of white toy car", "roof of red toy car", "roof of blue toy car", "roof of yellow toy car",
        "black toy car", "green toy car", "pink toy car", "orange toy car", "purple toy car",
        "light blue toy car", "cyan toy car", "orange toy car",
        "arduino rc car",
        "diy arduino rc car",
        "homemade arduino rc car",
        "custom rc car with arduino",
        "arduino-controlled rc car",
        "arduino based toy car",
        "arduino robot car",
        "arduino smart car",
        "2 wheel drive arduino car",
        "4 wheel drive arduino car",
        "arduino rc car top view",
        "arduino rc car side view",
        "arduino rc car front view",
        "arduino rc car rear view",
        "arduino rc car angled view",
        "arduino robot car chassis",
        "arduino car with l298n",
        "rc car with motor driver",
        "arduino car wiring visible",
        "arduino car breadboard mounted",
        "arduino car with cables exposed",
        "arduino car with battery pack",
        "rc robot car with ultrasonic sensor",
        "arduino car with hc-sr04",
        "arduino car with sensor on front",
        "bluetooth controlled arduino car",
        "arduino rc car with hc-05 module",
        "arduino nano robot car",
        "arduino uno robot car",
        "small arduino-powered car",
        "arduino car with wheels and motors",
        "line follower arduino car",
        "arduino car with ir sensors",
        "4wd robot car kit",
        "2wd robot car kit",
        "robot car with acrylic chassis",
        "robot car with yellow motors",
        "diy rc car with wiring visible",
        "electronics visible on car",
        "open-frame rc robot car",
        "mini robotic vehicle",
        "arduino project toy car",
        "rc car with circuit board on top"
    ],
    "police_toy_car": [
        "toy police car",
        "miniature police car",
        "police toy cruiser",
        "small toy police car",
        "toy police car with police decals",
        "toy police car with POLICE text",
        "diecast toy police car",
        "plastic toy police car",
        "police toy car top view",
        "police toy car side view",
        "police toy car front view",
        "police toy car rear view",
        "police toy car angled view",
        "toy police pickup",
    ]
}

# ===== Flatten prompts and build maps =====
CLASS_PROMPTS = []
CANON_NAME_MAP = {}   # idx -> "toy_car" / "police_toy_car"
PROMPT_NAME_MAP = {}  # idx -> full prompt text

_idx = 0
for canonical_name, prompts in ALIASES.items():
    for p in prompts:
        CLASS_PROMPTS.append(p)
        CANON_NAME_MAP[_idx] = canonical_name
        PROMPT_NAME_MAP[_idx] = p
        _idx += 1

# ======================= UTILITIES =======================
def ensure_dir(p: Path):
    """Create a directory (and parents) if it doesn't exist."""
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

    # All patterns are now RELATIVE (no leading "/")
    # and we’ll use rglob to search recursively.
    patterns = [
        "mobileclip.ts",     # e.g. mobileclip_blt.ts
        "mobileclip/.ts",    # any .ts inside a mobileclip folder
        "mobileclip",        # mobileclip dirs/files
    ]

    removed = 0
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            # rglob supports relative patterns and wildcards
            for hit in root.rglob(pat):
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
def prepare_mobileclip_asset(force_fresh_cache=True, validate_load=True):
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
    prepare_mobileclip_asset(force_fresh_cache=True, validate_load=True)
    try:
        return model.get_text_pe(texts)
    except Exception as e:
        print("[WARN] get_text_pe failed after prepare; hard reset caches and retry once...\n", e)
        prepare_mobileclip_asset(force_fresh_cache=True, validate_load=True)
        return model.get_text_pe(texts)

# ======================= ROAD REGION HELPERS (IRREGULAR / POLYGON) =======================
def load_road_config(config_path: str = REGION_CONFIG_PATH):
    """
    Load a JSON file describing irregular road polygons.

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
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Road region config '{config_path}' not found. "
            f"Create it first with the region-definition script."
        )
    with open(config_path, "r") as f:
        cfg = json.load(f)
    return cfg

def scale_polygon(points, src_w, src_h, dst_w, dst_h):
    """Scale polygon points from (src_w,src_h) coordinate system to (dst_w,dst_h)."""
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    return [(int(px * sx), int(py * sy)) for (px, py) in points]

def get_road_regions(frame, cfg):
    """
    Build road regions for this frame from config.

    Returns a list of dicts:
    [
      {
        "name": str,
        "poly": [(x,y), ...],
        "x1": int, "y1": int, "x2": int, "y2": int,   # bounding box of poly
      },
      ...
    ]
    """
    h, w = frame.shape[:2]
    src_w = cfg["frame_width"]
    src_h = cfg["frame_height"]

    regions = []
    for road in cfg["roads"]:
        name = road["name"]
        pts_src = road["points"]
        # points may come as lists -> convert to tuples
        pts_src = [(p[0], p[1]) for p in pts_src]
        pts = scale_polygon(pts_src, src_w, src_h, w, h)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = max(min(xs), 0), min(max(xs), w - 1)
        y1, y2 = max(min(ys), 0), min(max(ys), h - 1)

        # ensure valid bbox (add safety)
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))

        # ensure x2 > x1 and y2 > y1 to avoid empty crops
        if x2 <= x1 or y2 <= y1:
            # expand a little if possible
            x2 = min(x1 + 2, w - 1)
            y2 = min(y1 + 2, h - 1)

        regions.append({
            "name": name,
            "poly": pts,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })
    return regions

def point_in_polygon(x, y, poly):
    """
    Ray-casting algorithm for point in polygon.
    poly: list of (x,y) in full-frame coordinates.
    """
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

# ======================= INFERENCE ON IMAGE =======================
def infer_on_image(model: YOLOE, image_path: str, out_dir: Path):
    """
    Run promptable segmentation on a single image, save annotated copy,
    AND display the annotated image in a window.

    - Saved image: canonical label (toy_car / police_toy_car)
    - Window: actual prompt label ("toy police car with siren lights", etc.)
    """
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
        # 1) Use canonical name for saving
        r.names = CANON_NAME_MAP

        annotated_for_save = r.plot()
        if annotated_for_save is None:
            continue

        suffix = "" if len(results) == 1 else f"_{i}"
        out_path = out_dir / f"{base}_toycar{suffix}{ext}"

        cv2.imwrite(str(out_path), annotated_for_save)
        print(f"[INFO] Saved annotated image: {out_path}")
        saved_any = True

        # 2) For the window: show the actual prompt label
        r.names = PROMPT_NAME_MAP      # now each box label is the full prompt
        debug_vis = r.plot()

        cv2.imshow("Toy Car Detection - Image (PROMPT LABELS)", debug_vis)
        print("[INFO] Press any key in the image window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not saved_any:
        print("[INFO] No toy cars detected above threshold; nothing saved.")

# ======================= INFERENCE ON VIDEO (4 IRREGULAR ROADS) =======================
def infer_on_video(model: YOLOE, video_path, out_dir: Path, ser=None):
    """
    Run promptable segmentation on each frame of a video or webcam.

    BUT: instead of sending a mode every frame, we process batches of 10 frames:
      - For each road, the batch "car count" = max number of cars detected for that road
        in any of those 10 frames.
      - If police is detected in at least 1 frame of the batch, the whole batch is
        considered police_detected.
      - After the batch is processed, we compute the mode once and send it to Arduino.
    """
    print(f"\n[INFO] Inference on video source: {video_path}")

    # If it's a string, treat it as a video file path and check existence
    if isinstance(video_path, str):
        if not os.path.isfile(video_path):
            print(f"[WARN] File does not exist: {video_path}")
            return

    # Load road configuration
    try:
        road_cfg = load_road_config(REGION_CONFIG_PATH)
        print(f"[INFO] Loaded road regions from {REGION_CONFIG_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not load road config: {e}")
        return

    # The canonical order of roads (road_1, road_2, road_3, road_4, ...)
    road_names_in_order_cfg = [r["name"] for r in road_cfg["roads"]]

    ensure_dir(out_dir)

    # Honor video_path: int for webcam, str for file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25.0  # sensible default if metadata is missing

    # Use a generic base name for webcam
    if isinstance(video_path, int):
        base = f"webcam_{video_path}"
    else:
        base = Path(video_path).stem

    out_path = out_dir / f"{base}_toycar.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print(f"[INFO] Writing annotated video to: {out_path.resolve()}")
    print("[INFO] Press 'q' in the video window to stop early.")

    # ---------- BATCH STATE ----------
    BATCH_SIZE = 30
    batch_frame_count = 0
    batch_road_counts_list = []   # list of dicts {road_name: count}
    batch_police_flags = []       # list of bools

    frame_idx = 0

    def compute_batch_mode(road_counts_list, police_flags):
        """
        Aggregate a batch of per-frame road_counts and police flags, and
        return (mode, aggregated_road_counts, police_detected_batch).
        """
        # Aggregate counts: for each road, take the maximum over the batch
        agg_counts = {name: 0 for name in road_names_in_order_cfg}
        for rc in road_counts_list:
            for name in road_names_in_order_cfg:
                agg_counts[name] = max(agg_counts[name], rc.get(name, 0))

        police_batch = any(police_flags)

        # ----- Decide mode -----
        if EMERGENCY_SOURCE == "camera" and police_batch:
            mode_to_send = 5
        else:
            counts_in_order = [agg_counts[name] for name in road_names_in_order_cfg]

            # If all counts equal -> 0 (normal)
            if len(set(counts_in_order)) == 1:
                mode_to_send = 0
            else:
                max_count = max(counts_in_order)
                roads_with_max = [i for i, c in enumerate(counts_in_order) if c == max_count]

                if len(roads_with_max) == 1:
                    # Exactly one road has strictly more cars (1..4)
                    mode_to_send = roads_with_max[0] + 1
                else:
                    # Tie for max between multiple roads -> fallback to normal
                    mode_to_send = 0

        return mode_to_send, agg_counts, police_batch
    # ---------- MAIN LOOP ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No more frames (or camera disconnected).")
            break

        frame_idx += 1

        # Start from original frame for saving and showing
        annotated_for_save = frame.copy()

        # Get irregular road regions (polygons) for this frame
        road_regions = get_road_regions(frame, road_cfg)
        road_counts = {}
        police_detected = False

        # Process each road separately
        for region_idx, region in enumerate(road_regions):
            x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
            road_name = region["name"]
            poly = region["poly"]  # list of (x,y) in full-frame coordinates

            # crop must be within frame
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                road_counts[road_name] = 0
                continue

            # Run YOLOE on this crop
            results = model.predict(
                source=crop,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                device=DEVICE,
                save=False,
                verbose=False,
            )

            car_count = 0

            if results:
                r = results[0]

                # Filter detections so we only keep boxes whose centers lie inside the polygon
                if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                    boxes_xyxy = r.boxes.xyxy.cpu().numpy()  # (N,4)
                    keep_indices = []

                    for i_box, (bx1, by1, bx2, by2) in enumerate(boxes_xyxy):
                        # center in crop coordinates
                        cx_crop = (bx1 + bx2) / 2.0
                        cy_crop = (by1 + by2) / 2.0
                        # map to full-frame coordinates
                        cx = cx_crop + x1
                        cy = cy_crop + y1

                        if point_in_polygon(cx, cy, poly):
                            keep_indices.append(i_box)

                    if keep_indices:
                        try:
                            r.boxes = r.boxes[keep_indices]
                        except Exception:
                            r.boxes = None

                        if getattr(r, "masks", None) is not None:
                            try:
                                r.masks = r.masks[keep_indices]
                            except Exception:
                                r.masks = None

                        car_count = len(keep_indices)
                    else:
                        r.boxes = None
                        if getattr(r, "masks", None) is not None:
                            r.masks = None
                        car_count = 0

                # Check for police cars among remaining detections
                if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                    try:
                        classes = r.boxes.cls.cpu().numpy().astype(int)
                        for cls_id in classes:
                            canon_name = CANON_NAME_MAP.get(int(cls_id), "")
                            if canon_name == "police_toy_car":
                                police_detected = True
                                break
                    except Exception:
                        pass

                # Use prompt labels (actual class text) on the boxes in the visualization
                r.names = PROMPT_NAME_MAP

                # Annotate the crop with boxes
                ann_crop = r.plot() if getattr(r, "boxes", None) is not None else crop.copy()
                # Paste the annotated crop back into the full frame (if shapes match)
                if (
                    ann_crop is not None
                    and ann_crop.shape[0] == (y2 - y1)
                    and ann_crop.shape[1] == (x2 - x1)
                ):
                    try:
                        annotated_for_save[y1:y2, x1:x2] = ann_crop
                    except Exception:
                        pass

            road_counts[road_name] = car_count

            # Draw road polygon on the full frame
            pts_arr = np.array(poly, dtype=np.int32)
            cv2.polylines(
                annotated_for_save,
                [pts_arr],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )

            # Put road name and count near first vertex
            label_x, label_y = poly[0]
            cv2.putText(
                annotated_for_save,
                f"{road_name}: {car_count}",
                (label_x + 5, label_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # ---- accumulate into current batch ----
        batch_frame_count += 1
        batch_road_counts_list.append(road_counts)
        batch_police_flags.append(police_detected)

        # Write frame with per-road boxes + counts
        out_writer.write(annotated_for_save)

        # Show debug view
        cv2.imshow("Toy Car Detection - Video (Irregular Roads)", annotated_for_save)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 'q' pressed, stopping early.")
            break

        # ---- when batch is full, compute and send mode ----
        if batch_frame_count >= BATCH_SIZE:
            mode_to_send, agg_counts, police_batch = compute_batch_mode(
                batch_road_counts_list, batch_police_flags
            )

            print(
                f"[BATCH END] frames {frame_idx - BATCH_SIZE + 1}–{frame_idx}, "
                f"agg_counts={agg_counts}, police_batch={police_batch}, mode={mode_to_send}"
            )
            send_mode_to_arduino(ser, mode_to_send)

            # reset batch
            batch_frame_count = 0
            batch_road_counts_list = []
            batch_police_flags = []

        if frame_idx % 50 == 0:
            print(f"[INFO] Processed {frame_idx} frames...")

    # ---- handle last partial batch (if any frames left) ----
    if batch_frame_count > 0 and len(batch_road_counts_list) > 0:
        mode_to_send, agg_counts, police_batch = compute_batch_mode(
            batch_road_counts_list, batch_police_flags
        )
        print(
            f"[FINAL PARTIAL BATCH] last {batch_frame_count} frames, "
            f"agg_counts={agg_counts}, police_batch={police_batch}, mode={mode_to_send}"
        )
        send_mode_to_arduino(ser, mode_to_send)

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Video processing complete.")
# ======================= MAIN =======================
def main():
    # 1) Open serial to Arduino (optional)
    ser = init_serial()

    # 2) Load YOLOE model
    print("[INFO] Loading YOLOE promptable segmentation model...")
    model = YOLOE(WEIGHTS_PATH)

    # 3) Set promptable classes
    print("[INFO] Setting prompt classes:")
    for t in CLASS_PROMPTS:
        print("   -", t)

    text_pe = get_text_embeddings_with_retry(model, CLASS_PROMPTS)
    model.set_classes(CLASS_PROMPTS, text_pe)

    # Best-effort: set canonical names
    try:
        if hasattr(model, "names"):
            model.names = CANON_NAME_MAP
        if hasattr(model, "model") and hasattr(model.model, "names"):
            model.model.names = CANON_NAME_MAP
    except Exception as e:
        print("[WARN] Could not set model names globally:", e)

    # 4) START MICROPHONE THREAD (NON-BLOCKING)
    if EMERGENCY_SOURCE == "audio":
        audio_thread = threading.Thread(
            target=audio_loop,
            args=(ser,),
            daemon=True   # very important
        )
        audio_thread.start()
        print("[INFO] Audio (microphone) thread started.")
    else:
        print("[INFO] EMERGENCY_SOURCE='camera' → microphone will not trigger mode 5.")

    # 5) START CAMERA (BLOCKING MAIN THREAD)
    print("[INFO] Starting camera + vision inference...")
    infer_on_video(
        model=model,
        video_path=2,              # 0 = default webcam (Iriun / USB / Laptop cam)
        out_dir=Path("runs/infer"),
        ser=ser
    )

    # ----------------------------------------
main()