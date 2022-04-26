import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from tracemalloc import start
from turtle import width
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
from feature_creating import create_box_encoder
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from utils import draw_bbox

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weight", type=str, help="a path leading to the weight file for yolov4")
    parser.add_argument("stream", type=str, help="identify method of streaming, video or live")
    parser.add_argument("-v", "--videopath", type=str, help="a path to the video for tracking")
    parser.add_argument("-o", "--output", type=str, help="path to output folder")
    
    args = parser.parse_args()

    model_weights = args.weight
    stream = args.stream
    video_path = args.videopath
    output = args.output

    # Check input arguments
    if stream != "video" and stream != "live":
        print("The streaming method entered is not supported, please check again.")
        sys.exit(1)
    if video_path != None:
        if stream == "live":
            print("The live streamming does not need a video path, please check again.")
            sys.exit(1)
        if not os.path.isfile(video_path):
            print("The path entered does not lead to a video file, please check again.")
            sys.exit(1)

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
    WIDTH, HEIGHT = (320, 320)

    detector = YOLOv4( input_shape=(HEIGHT, WIDTH, 3), anchors=YOLOV4_ANCHORS, num_classes=80,
                    training=False, yolo_max_boxes=50, yolo_iou_threshold=0.5,
                    yolo_score_threshold=0.5)
    detector.load_weights(model_weights)

    model_filename = 'weights/mars-small128.pb'
    encoder = create_box_encoder(model_filename, batch_size=1)
    metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.7)
    tracker = Tracker(metric=metric)

    if stream == "live":
        vid = cv2.VideoCapture(0)
    elif stream == "video":
        vid = cv2.VideoCapture(video_path)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    tracking_obj = ["person"]

    detectionTime, totalTime = [], []

    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break

        image = tf.image.resize(np.copy(original_frame), (HEIGHT, WIDTH))
        image = tf.expand_dims(image, axis=0)/255

        start_time = time.time()
        boxes, scores, classes, detections = detector.predict(image)
        detections = detections[0]
        boxes = boxes[0] * [width, height, width, height]
        scores = scores[0]
        classes = classes[0].astype(int)
        detect_time = time.time()

        trk_boxes, trk_scores, names = [], [], []
        for index, bbox in enumerate(boxes):
            if scores[index] > 0:
                if (len(tracking_obj) !=0 and CLASSES[classes[index]] in tracking_obj) or len(tracking_obj) == 0:
                    trk_boxes.append([int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])])
                    trk_scores.append(scores[index])
                    names.append(CLASSES[classes[index]])

        # Obtain all the detections for the given frame.
        trk_boxes = np.array(trk_boxes) 
        names = np.array(names)
        trk_scores = np.array(trk_scores)
        #print(trk_boxes)
        features = np.array(encoder(original_frame, trk_boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(trk_boxes, trk_scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = CLASSES.index(class_name) # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, classes=CLASSES, tracking=True)

        track_time = time.time()
        detectionTime.append(detect_time-start_time)
        totalTime.append(track_time-start_time)
        
        detectionTime = detectionTime[-20:]
        totalTime = totalTime[-20:]

        ms = sum(detectionTime)/len(detectionTime)*1000
        detect_fps = 1000 / ms
        total_fps = 1000 / (sum(totalTime)/len(totalTime)*1000)

        image = cv2.putText(image, "Time: {:.1f} FPS".format(detect_fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, detect_fps, total_fps))
        if output == None:
            cv2.imshow('output', original_frame)
        else: 
            out.write(image)
            
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
            
    cv2.destroyAllWindows()
