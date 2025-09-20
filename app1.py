import cv2
from flask import Flask, Response
from ultralytics import YOLO
import os  # <-- for Render PORT environment variable

app = Flask(__name__)
model = YOLO("yolo11s.pt")  # make sure this file is in your repo

# Change to 0 for webcam, or a video path
VIDEO_PATH = "Shopping, People, Commerce, Mall, Many, Crowd, Walking Free Stock video footage YouTube - master r.mp4"

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            continue

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
    # Use Render/Railway PORT or fallback to 10000 for local
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
