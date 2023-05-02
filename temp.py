import torch
import cv2

# model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# img
img = cv2.imread("demo2.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 500))

# result
results = model(img)

# data - bounding boxes, confidence scores, and labels(classes)
predictions = results.pred[0]
total_objects = len(predictions)
boxes = predictions[:, :4]  #x1, y1, x2, y2
scores = predictions[:, 4]  # confidence scores
labels = predictions[:, 5]  # class names

print(total_objects)
# print(boxes)
print(scores)
print(str(labels))

results.save(save_dir="results")

cv2.imshow("somesh", results.render()[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
