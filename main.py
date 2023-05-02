import cv2
import torch

# object detection model
model = torch.hub.load("ultralytics/yolov5", model="yolov5s", pretrained=True, _verbose=False, force_reload=True)
class_names = model.names

# capture
cap = cv2.VideoCapture("demo.mp4")
# cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # object detection
    res = model(frame[:, :, ::-1], size=340)
    predictions = res.xyxy[0]

    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred[:4])
        conf = float(pred[4])
        class_id = int(pred[5])
        name = f"{class_names[class_id]} {round(conf, 2)}"
        # print(name)

        if conf >= 0.50:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 17), (x1 + (len(name)*10), y1), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1, y1 - 4), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255))

    cv2.imshow("Frame", frame)
    if cv2.waitKey(30) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
