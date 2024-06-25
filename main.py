from flask import Flask, render_template, Response
from flask_caching import Cache
from flask_socketio import SocketIO, emit
from PIL import Image
import base64
from io import BytesIO
from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from flask import jsonify


cache = Cache(config={"CACHE_TYPE": "SimpleCache"})

app = Flask(__name__)
socketio = SocketIO(app)
cache.init_app(app)

# Initialize YOLO model
model = YOLO("oldv8.pt")

names = [
    "ال",
    "ع",
    "د",
    "ض",
    "ذ",
    "ا",
    "ب",
    "ف",
    "ج",
    "غ",
    "ه",
    "ح",
    "ئ",
    "ك",
    "خ",
    "لا",
    "ل",
    "م",
    "ن",
    "ق",
    "ر",
    "ص",
    "س",
    "ش",
    "ت",
    "ط",
    "ة",
    "ث",
    "و",
    "ي",
    "ظ",
    "ز",
]


@socketio.on("video_stream")
def handle_video_stream(image):
    img_data = base64.b64decode(image.split(",")[1])
    img = Image.open(BytesIO(img_data))
    img = img.resize((320, 240))
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    result_images = model(img, show=False, conf=0.5, save=False)

    for result_img in result_images:
        for box in result_img.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = names[cls]
            count = cache.get(current_class) or 1
            cache.set(current_class, count + 1)
            cvzone.putTextRect(img_cv2, f"{conf} {current_class}", (x1, y1 + 20))

    _, buffer = cv2.imencode(".jpg", img_cv2)
    img_str = base64.b64encode(buffer).decode()

    emit("detection_results", img_str)


def return_character():
    max_count = 0
    character = None
    for name in names:
        if cache.get(name):
            max_count = max(cache.get(name), max_count)
            if max_count == cache.get(name):
                character = name
    cache.clear()
    return character


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/show_text")
def show_text():
    x = return_character()
    return jsonify({"character": x})


if __name__ == "__main__":
    socketio.run(app)
