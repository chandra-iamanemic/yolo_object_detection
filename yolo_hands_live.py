# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:09:11 2019

@author: cks
"""

import cvlib as cv
from cvlib.object_detection import draw_bbox
import sys
import cv2
import time

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
while webcam.isOpened():

    # read frame from webcam 
    status, image = webcam.read()

    time.sleep(1)
    # apply object detection
    (bbox, label, conf) = cv.detect_common_objects(image)
    
    print(bbox, label, conf)
    
    # draw bounding box over detected objects
    out = draw_bbox(image, bbox, label, conf)
    
    # display output
    # press any key to close window           
    cv2.imshow("object_detection", out)
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# release resources
webcam.release()
cv2.destroyAllWindows()

