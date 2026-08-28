import time
import os
import io
import threading
import queue
from datetime import datetime
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput
from flask import Flask, Response

cmd_queue = queue.Queue()


def input_thread():
    """Runs in background, forwards every typed line to the main loop.
    Only this thread ever calls input() -> avoids stdin race conditions."""
    while True:
        line = input()
        cmd_queue.put(line.strip())


# ---------------------------------------------------------------------------
# Mode 2: Live view — Flask MJPEG stream, viewable in a browser. No images saved.
# ---------------------------------------------------------------------------

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


def run_live_view(picam2):
    picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}))
    output = StreamingOutput()
    picam2.start_recording(MJPEGEncoder(), FileOutput(output))

    app = Flask(__name__)

    @app.route("/")
    def index():
        return "<html><body style='margin:0'><img src='/stream.mjpg' /></body></html>"

    @app.route("/stream.mjpg")
    def stream():
        def generate():
            while True:
                with output.condition:
                    output.condition.wait()
                    frame = output.frame
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    print("\nLive view running. On your laptop, open http://<pi-ip-address>:5000")
    print("Press Ctrl+C in this terminal to stop.\n")
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        picam2.stop_recording()


# ---------------------------------------------------------------------------
# Mode 1: Capture mode — timed stills saved to a folder, with stop/resume/new-folder.
# ---------------------------------------------------------------------------

def run_capture_mode(picam2):
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    time.sleep(2)  # sensor warm-up

    folder = input("Folder name to save images to: ").strip()
    interval = float(input("Interval between photos (seconds): ").strip())
    os.makedirs(folder, exist_ok=True)

    print(f"\nCapturing to '{folder}' every {interval}s.")
    print("While running, type 's' + Enter to stop.\n")

    state = "running"  # running -> stopped -> {await_folder -> await_interval -> running}
    count = 0
    new_folder = None

    while True:
        try:
            cmd = cmd_queue.get_nowait().lower()
        except queue.Empty:
            cmd = None

        if state == "running" and cmd == "s":
            state = "stopped"
            print(">> Stopped. 'r' = resume, 'n' = new folder, 'q' = quit.")

        elif state == "stopped":
            if cmd == "r":
                state = "running"
                print(f">> Resumed. Saving to '{folder}' every {interval}s.")
            elif cmd == "n":
                state = "await_folder"
                print("New folder name:")
            elif cmd == "q":
                break

        elif state == "await_folder" and cmd:
            new_folder = cmd
            state = "await_interval"
            print(f"Interval in seconds (blank = keep {interval}s):")

        elif state == "await_interval" and cmd is not None:
            if cmd:
                interval = float(cmd)
            folder = new_folder
            os.makedirs(folder, exist_ok=True)
            count = 0
            state = "running"
            print(f">> Resumed. Saving to '{folder}' every {interval}s.")

        if state == "running":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(folder, f"{timestamp}_{count:04d}.jpg")
            picam2.capture_file(filename)
            print(f"Saved {filename}")
            count += 1
            time.sleep(interval)
        else:
            time.sleep(0.2)

    picam2.stop()
    print("Session ended.")


# ---------------------------------------------------------------------------

def main():
    print("Select mode:")
    print("  1) Capture mode  - take photos on an interval, save to a folder")
    print("  2) Live view mode - stream live video to your browser, nothing saved")
    mode = input("Enter 1 or 2: ").strip()

    picam2 = Picamera2()

    if mode == "1":
        threading.Thread(target=input_thread, daemon=True).start()
        run_capture_mode(picam2)
    elif mode == "2":
        run_live_view(picam2)
    else:
        print("Invalid choice, exiting.")


if __name__ == "__main__":
    main()