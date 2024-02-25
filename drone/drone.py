import cv2
import numpy as np
from djitellopy import tello
import time
from backend import *
import math
import logging

# Remove all handlers associated with the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set the logging level (optional)
logging.basicConfig(level=logging.CRITICAL)  # This effectively silences most logging, adjust as needed

me = tello.Tello()
me.connect()
# Getting the drones battery
print(me.get_battery())


me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 20, 0)
time.sleep(2.2)
w, h = 640, 480
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
pError = 0
def findFace(img):
    faceCascade= cv2.CascadeClassifier("C:/Users/Ericc/.vscode/HackTJ2024/H/N/venv/data/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))

        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

def trackFace( info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0
    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    if x == 0:
        speed = 0
        error = 0
    #print(speed, fb)
    me.send_rc_control(0, fb, 0, speed)
    return error

#cap = cv2.VideoCapture(1)

# Classification Init
classifier = Classifier("ml/weights/n5.pt")
classify = False

# me.send_rc_control(0, 0, 0, 0)

while True:

    #_, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img, info = findFace(img)
    pError = trackFace( info, w, pid, pError)
    #print(“Center”, info[0], “Area”, info[1])

    results = classifier.model.predict(img, stream=True)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:
                continue

            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # object details
            inText = objectCategory[cls] if classify else objectName[cls]
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(
                img,
                inText,
                org,
                font,
                fontScale,
                color,
                thickness,
            )

    cv2.imshow("SS-Corp Tetravaal", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break
