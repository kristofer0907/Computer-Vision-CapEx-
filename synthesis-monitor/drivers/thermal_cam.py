from flask import Flask, Response
import numpy as np
import cv2
import board
import busio
import adafruit_mlx90640


class ThermalCamera:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
        self.frame = [0] * 768

    def get_frame(self, mock_up=False):

        if mock_up:
            # generate a mock-up frame for testing purposes
            data = np.random.rand(24, 32).astype(np.float32) * 100
            norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            colored = cv2.resize(colored, (640, 480), interpolation=cv2.INTER_NEAREST)

            ret, buffer = cv2.imencode('.jpg', colored)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            while True:
                try:
                    self.mlx.getFrame(self.frame)
                except ValueError:
                    continue  # skip failed reads, don't crash the stream

                data = np.reshape(self.frame, (24, 32)).astype(np.float32)

                # normalize per-frame to 0-255 for display
                norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

                # upscale — 32x24 native is too small to see anything
                colored = cv2.resize(colored, (640, 480), interpolation=cv2.INTER_NEAREST)

                ret, buffer = cv2.imencode('.jpg', colored)
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
