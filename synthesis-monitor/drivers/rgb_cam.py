from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np

class RGBCamera:
    def __init__(self):
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"format": "RGB888", "size": (1280, 720)}
        )
        self.camera.configure(config)
        self.camera.start()

    def get_frame(self, mock_up=False):
        if mock_up:
            # generate a mock-up frame for testing purposes
            frame = np.random.rand(720, 1280, 3).astype(np.float32)
        else:
            while True:
                frame = self.camera.capture_array()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def start_stream(self):
        app = Flask(__name__)

        @app.route('/video_feed')
        def video_feed():
            return Response(self.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

        if __name__ == '__main__':
            app.run(host='0.0.0.0', port=5000, use_reloader=False)

app = Flask(__name__)
camera = Picamera2()

config = camera.create_preview_configuration(
    main={"format": "RGB888", "size": (1280, 720)}
)
camera.configure(config)
camera.start()

def generate_frames():
    while True:
        frame = camera.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, use_reloader=False)