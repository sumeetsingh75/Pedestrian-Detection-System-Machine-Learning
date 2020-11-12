######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a video.
# It draws boxes, scores, and labels around the objects of interest in each
# frame of the video.

# Some of the code is copied from Google's example at
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

# and some is copied from Dat Tran's example at
# https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

# but I changed it to make it more understandable to me.

# Import packages
from imutils.video import FileVideoStream
from imutils.video import FPS
from utils import visualization_utils as vis_util
from utils import label_map_util
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, send
import argparse
import os.path
import eventlet

eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


# ROI variables
x_min = -1
y_min = -1
x_max = -1
y_max = -1
drawing = False


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph-80k'
VIDEO_NAME = 'v1.mp4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH, VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

# This method allows user to draw the Region of Interest on the Video Frame


def gen():

    def draw_region_of_interest():
        global x_min, y_min, x_max, y_max
        x_min = 220
        y_min = 98
        x_max = 640
        y_max = 360
        cv2.rectangle(frame, pt1=(x_min, y_min), pt2=(
            x_max, y_max), color=(0, 255, 255), thickness=1)

    # Flag to draw Region of Interest on first frame
    first_frame = True
    while(video.isOpened()):
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        if first_frame:
            cv2.namedWindow(winname="Draw Region Of Interest")
            cv2.putText(frame, "Define your Region of Interest", (25, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
            # cv2.setMouseCallback("Draw Region Of Interest",
            #                      draw_region_of_interest)

            draw_region_of_interest()

            cv2.imshow("Draw Region Of Interest", frame)

            # while True:
            #     cv2.imshow("Draw Region Of Interest", frame)
            #     if cv2.waitKey(10) == 13 and x_min != -1:
            #         first_frame = False
            #         break
            first_frame = False
            cv2.destroyAllWindows()

        # Stop if end of video
        if not ret:
            break
        roi = frame[y_min:y_max, x_min:x_max]
        frame_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        # Returns frame with bounding boxes and pedestrian count
        ret_roi_frame, pedestrian_count = vis_util.visualize_boxes_and_labels_on_image_array(
            roi,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

        # draw region of interest
        cv2.rectangle(frame, pt1=(x_min, y_min), pt2=(
            x_max, y_max), color=(0, 255, 255), thickness=1)
        # mask output of model over original frame
        frame[y_min:y_max, x_min:x_max] = ret_roi_frame
        # displays pedestrian count
        pedestrian_count_label = 'Pedestrain count: %.2f' % pedestrian_count
        cv2.putText(frame, pedestrian_count_label, (25, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_4)
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Co image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0')
