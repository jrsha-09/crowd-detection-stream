import cv2
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8s.pt")  # put in repo or download on start

VIDEO_PATH = "Shopping_1.mp4"  # change to 0 for webcam

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, imgsz=640, conf=0.5)
        annotated = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return "✅ Crowd Detection Stream is running!"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
