import time
import urllib.request
import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import serial

# =============== SERIAL CONFIG ===============
SERIAL_PORT = "COM8"   # 🔴 CHANGE to your Arduino COM port
BAUD_RATE = 9600
# ============================================

# =============== AUDIO CONFIG ===============
SAMPLE_RATE = 16000
CHUNK_SECONDS = 3
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
# ============================================

# =============== SOUND KEYWORDS ===============
EMERGENCY_KEYWORDS = [
    "siren",
    "police",
    "ambulance",
    "fire engine",
    "emergency",
    "buzzer",
    "alarm",
    "horn",
]
# ============================================

SIREN_THRESHOLD = 0.2


# =============== SERIAL INIT ===============
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("[INFO] Arduino connected.")
except Exception as e:
    print(f"[ERROR] Serial connection failed: {e}")
    ser = None
# ============================================


# =============== LOAD YAMNET ===============
print("[INFO] Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("[INFO] YAMNet loaded.")
# ==========================================


# =============== LOAD CLASS NAMES ===============
labels_path = "codes\\caproad\\yamnet_class_map.csv"

try:
    with open(labels_path, "r") as f:
        lines = f.read().strip().split("\n")
except FileNotFoundError:
    url = "https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv"
    urllib.request.urlretrieve(url, labels_path)
    with open(labels_path, "r") as f:
        lines = f.read().strip().split("\n")

class_names = [l.split(",")[2] for l in lines]
# ===============================================


def is_emergency_sound(mean_scores):
    best_label = None
    best_score = 0.0

    for i, score in enumerate(mean_scores):
        name = class_names[i].lower()
        if any(k in name for k in EMERGENCY_KEYWORDS):
            if score > best_score:
                best_score = score
                best_label = class_names[i]

    if best_label and best_score >= SIREN_THRESHOLD:
        return True, best_label, best_score

    return False, None, 0.0


def detect_chunk(audio_chunk):
    audio_tensor = tf.convert_to_tensor(audio_chunk, dtype=tf.float32)
    scores, _, _ = yamnet_model(audio_tensor)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)
    return mean_scores


def send_to_arduino(value):
    if ser:
        ser.write(f"{value}\n".encode())


def main():
    last_sent = None

    print("[INFO] Microphone active. Press Ctrl+C to stop.\n")

    while True:
        try:
            audio = sd.rec(
                frames=CHUNK_SAMPLES,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio[:, 0]

            mean_scores = detect_chunk(audio)

            # Check emergency sounds
            found, label, score = is_emergency_sound(mean_scores)

            # Top classes
            top_indices = np.argsort(mean_scores)[::-1][:5]
            top_names = [class_names[i].lower() for i in top_indices]

            emergency_in_top = any(
                any(k in name for k in EMERGENCY_KEYWORDS)
                for name in top_names
            )

            # -------- MODE LOGIC --------
            if found or emergency_in_top:
                mode = 2
                print(f"🚨 EMERGENCY SOUND → {label} ({score:.3f})")
            else:
                mode = 0
                print("✅ No emergency sound.")

            # Send only if changed
            if mode != last_sent:
                send_to_arduino(mode)
                last_sent = mode

            # Print top classes
            print("Top classes:")
            for i in top_indices:
                print(f"  {class_names[i]} — {mean_scores[i]:.3f}")
            print()

            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
