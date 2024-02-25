from backend import *
import cv2
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classifier = Classifier("weights/n5.pt")

classify = False

while True:
    success, img = cap.read()
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

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("c"):
        classify = True
    else:
        classify = False
    print(ord("c"))


cap.release()
cv2.destroyAllWindows()
