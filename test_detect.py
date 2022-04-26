import cv2
import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import matplotlib.pyplot as plt
from utils import postprocess
 
WIDTH, HEIGHT = (1024, 1024)
 
image = cv2.imread('test.jpg')
print("SHAPE OF THE IMAGE: ", image.shape[:2])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_image = tf.image.resize(image, (HEIGHT, WIDTH))
images = tf.expand_dims(resized_image, axis=0) / 255
 
model = YOLOv4(
    input_shape=(HEIGHT, WIDTH, 3),
    anchors=YOLOV4_ANCHORS,
    num_classes=80,
    training=False,
    yolo_max_boxes=50,
    yolo_iou_threshold=0.4,
    yolo_score_threshold=0.5,
)
 
model.load_weights('weights/yolov4.h5')
 
boxes, scores, classes, detections = model.predict(images)
 

#print(boxes[0])

boxes = boxes[0] * [1920, 1080, 1920, 1080]
scores = scores[0]
classes = classes[0].astype(int)
detections = detections[0]
print(boxes)
#boxes = postprocess(boxes, image, (HEIGHT, WIDTH))
#print("After: \n", boxes)

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop',  'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
 
#plt.imshow(images[0])
#ax = plt.gca()
 
tracking_obj = ["person"]

for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):
    if score > 0:
        if (len(tracking_obj) !=0 and CLASSES[class_idx] in tracking_obj) or len(tracking_obj) == 0:
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,0,0), 3)
            
cv2.imshow('output', image)
cv2.waitKey(0)
#plt.title('Objects detected: {}'.format(detections))
#plt.axis('off')
#plt.show()