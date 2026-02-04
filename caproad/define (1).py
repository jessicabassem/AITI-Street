import cv2
import json
import os

CONFIG_PATH = "road_regions.json"

# ------- Mouse callback state -------
current_points = []     # points of the current polygon
all_roads = []          # list of {"name": str, "points": [(x,y), ...]}
current_road_idx = 0
MAX_ROADS = 4

drawing_window_name = "Define Roads (left-click points, 'n' next road, 's' save)"

def mouse_callback(event, x, y, flags, param):
    global current_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add a vertex to the current polygon
        current_points.append((x, y))

def draw_ui(frame, current_points, all_roads, current_road_idx):
    """Draw polygons and UI text on a copy of the frame."""
    vis = frame.copy()

    # Draw already completed roads (in green)
    for road in all_roads:
        pts = road["points"]
        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0,255,0), thickness=2)
        for (px, py) in pts:
            cv2.circle(vis, (px, py), 3, (0,255,0), -1)

    # Draw current polygon (in blue)
    if len(current_points) >= 1:
        cv2.polylines(vis, [np.array(current_points, dtype=np.int32)], isClosed=False, color=(255,0,0), thickness=2)
    for (px, py) in current_points:
        cv2.circle(vis, (px, py), 3, (255,0,0), -1)

    # Text instructions
    cv2.putText(
        vis,
        f"Road {current_road_idx+1}/{MAX_ROADS}: left-click = add point, 'z'=undo, 'n'=finish road, 's'=save&exit, 'q'=quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis

def define_road_regions(video_source=0, config_path=CONFIG_PATH):
    """
    Open a frame (from webcam or file), let user draw 4 polygons, and save them.
    video_source:
        - 0 for webcam
        - path to video file otherwise
    """
    import numpy as np

    global current_points, all_roads, current_road_idx

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open video/camera for region definition.")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not read first frame.")
        return

    h, w = frame.shape[:2]

    cv2.namedWindow(drawing_window_name)
    cv2.setMouseCallback(drawing_window_name, mouse_callback)

    print("[INFO] Defining road polygons.")
    print("Controls:")
    print("  - Left click: add a vertex")
    print("  - 'z': undo last point")
    print("  - 'n': finish current road polygon (needs >= 3 points)")
    print("  - 's': save config & exit (only if all roads finished)")
    print("  - 'q': quit WITHOUT saving")

    while True:
        vis = draw_ui(frame, current_points, all_roads, current_road_idx)
        cv2.imshow(drawing_window_name, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('z'):
            # Undo last point
            if current_points:
                current_points.pop()

        elif key == ord('n'):
            # Finish current road
            if len(current_points) < 3:
                print("[WARN] Need at least 3 points for a polygon.")
                continue

            road_name = f"Road {current_road_idx+1}"
            all_roads.append({"name": road_name, "points": current_points.copy()})
            current_points = []
            current_road_idx += 1
            print(f"[INFO] Finished {road_name}")

            if current_road_idx >= MAX_ROADS:
                print("[INFO] All roads defined. Press 's' to save or 'q' to discard.")
        
        elif key == ord('s'):
            # Save config
            if current_road_idx < MAX_ROADS:
                print("[WARN] You haven't finished all roads yet.")
                continue

            config = {
                "frame_width": w,
                "frame_height": h,
                "roads": all_roads
            }

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"[INFO] Saved road config to: {os.path.abspath(config_path)}")
            break

        elif key == ord('q'):
            print("[INFO] Quit without saving.")
            break

    cv2.destroyWindow(drawing_window_name)


if __name__ == "__main__":
    import numpy as np
    define_road_regions(0, CONFIG_PATH)