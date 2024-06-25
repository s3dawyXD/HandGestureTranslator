from ultralytics import YOLO
import cv2 as cv
import cvzone

# from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math


# Initialize YOLO model
model = YOLO("oldv8.pt")

# Class names for Arabic text
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
# Open the camera
cap = cv.VideoCapture(
    0
)  # 0 corresponds to the default camera, change it if you have multiple cameras
cap.set(3, 1400)
cap.set(4, 900)
while True:

    isTrue, frame = cap.read()
    framee = cv.flip(frame, 1)
    results = model(framee, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(framee, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            current_class = names[cls]

            cvzone.putTextRect(framee, f"{conf} {current_class }", (x1, y1 + 20))

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
    cv.imshow("frame", framee)
    cv.waitKey(1)


# #import time
# import cv2
# from ultralytics import YOLO
# #start_time = time.time()

# model = YOLO("oldv8.pt")
# #img = cv2.imread("car_175.jpg")
# result = model(source=0, show = True , conf = 0.5 , save = True  )
