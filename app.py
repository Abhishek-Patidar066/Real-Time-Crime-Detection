import json
import logging
import random
import sys
from datetime import datetime
from typing import Iterator
import cv2
import numpy as np
from keras.models import load_model
from collections import deque
from flask import Flask, Response, render_template, request, stream_with_context

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

random.seed()

SEQUENCE_LENGTH = 20
LRCN_model = load_model('Models/Model.h5')
CLASSES_LIST = ["Normal", "Abnormal"]
video_reader = cv2.VideoCapture(0)
frames_queue = deque(maxlen=SEQUENCE_LENGTH)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chart-data')
def chart_data() -> Response:
    response = Response(stream_with_context(generate_random_data()), mimetype='text/event-stream')
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response

def read_and_preprocess_frame():
    success, frame = video_reader.read()
    if not success:
        return None
    frame_resized = cv2.resize(frame, (64, 64))
    normalized_frame = frame_resized / 255.0
    return normalized_frame, frame

def generate_frames():
    while True:
        processed_frame = read_and_preprocess_frame()
        if processed_frame is None:
            break
        
        normalized_frame, frame = processed_frame
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            prediction = np.argmax(predicted_labels_probabilities)
            logger.info(f"Prediction: {CLASSES_LIST[prediction]}")
            frames_queue.clear()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_random_data() -> Iterator[str]:
    if request.headers.getlist("X-Forwarded-For"):
        client_ip = request.headers.getlist("X-Forwarded-For")[0]
    else:
        client_ip = request.remote_addr or ""

    logger.info("Client %s connected", client_ip)
    try:
        while True:
            processed_frame = read_and_preprocess_frame()
            if processed_frame is None:
                break

            normalized_frame, _ = processed_frame
            frames_queue.append(normalized_frame)

            if len(frames_queue) == SEQUENCE_LENGTH:
                predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
                predicted_label = np.argmax(predicted_labels_probabilities)
                logger.info(f"Prediction: {CLASSES_LIST[predicted_label]}")
                
                json_data = json.dumps({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "value": int(predicted_label),
                })
                yield f"data:{json_data}\n\n"

    except GeneratorExit:
        logger.info("Client %s disconnected", client_ip)

if __name__ == "__main__":
    try:
        app.run(threaded=True)
    finally:
        video_reader.release()
