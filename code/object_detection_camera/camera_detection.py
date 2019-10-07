# -*- coding: utf-8 -*-
"""
@author: sophia's husband
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

cap = cv2.VideoCapture(0)

MODEL_NAME = 'E:/CV/downloads/ssd_inception_v2_coco_2018_01_28'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = 'D:/Program_File/Anaconda3/envs/DLCNN/Lib/site-packages/object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# Loading a Tensorflow model into memory
with gfile.FastGFile(PATH_TO_CKPT, 'rb') as f:
    detection_graph = tf.GraphDef()
    detection_graph.ParseFromString(f.read())
    tf.import_graph_def(detection_graph, name='')


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Help code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# Detection
with tf.Session() as sess:
    while True:
        # Read frame from camera
        ret, image_np = cap.read()
        # Expand dimensions since the model expects images with shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Extract image tensor
        image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
        # Extract detection boxes
        boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
        # Extract detection scores
        scores = sess.graph.get_tensor_by_name('detection_scores:0')
        # Extract detection classes
        classes = sess.graph.get_tensor_by_name('detection_classes:0')
        # Extract number of detections
        num_detections = sess.graph.get_tensor_by_name('num_detections:0')

        # Actual detection
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        # Display output
        cv2.imshow('object detection', cv2.resize(image_np, (400, 300)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
