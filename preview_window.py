"""A window that loops the newest preview video in a folder at real NES speed.

train_ppo.py starts this automatically (turn it off with --no-preview-window).
It can also be run by hand on any training run's preview folder:

    python preview_window.py runs/<run name>/previews

Whenever a newer preview appears, it switches to it at the end of the current loop.
Press q in the window to close it (training keeps going).
"""
import glob
import os
import sys
import time

import cv2
import numpy as np

NES_FPS = 60.0988


def newest(folder):
    files = [f for f in glob.glob(os.path.join(folder, "update_*.mp4")) if not f.endswith(".partial.mp4")]
    return max(files, key=os.path.getmtime) if files else None


def load(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    title = "Best run preview (q to close)"
    current, frames = None, []
    waiting = np.zeros((240, 640, 3), dtype=np.uint8)
    cv2.putText(waiting, "waiting for the first preview...", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    while True:
        latest = newest(folder)
        if latest and latest != current:
            loaded = load(latest)
            if loaded:
                current, frames = latest, loaded
        if not frames:
            cv2.imshow(title, waiting)
            if cv2.waitKey(500) & 0xFF == ord("q"):
                break
            continue
        start = time.perf_counter()
        for i, f in enumerate(frames):
            cv2.imshow(title, f)
            delay = start + (i + 1) / NES_FPS - time.perf_counter()
            if cv2.waitKey(max(1, int(delay * 1000))) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                return
        if cv2.waitKey(1000) & 0xFF == ord("q"):  # short pause on the last frame before looping
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
