# -*- coding: utf-8 -*-
"""
Created on Fri May 17 13:17:04 2019

@author: cks
"""

import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

# read input image
image = cv2.imread('C:/Users/cks/Pictures/1.jpg')

# apply object detection
bbox, label, conf = cv.detect_common_objects(image,model = 'yolov3-tiny')

print(bbox, label, conf)

# draw bounding box over detected objects
out = draw_bbox(image, bbox, label, conf)

# display output
# press any key to close window           
cv2.imshow("object_detection", out)
cv2.waitKey()

# save output
#cv2.imwrite("object_detection.jpg", out)

# release resources
cv2.destroyAllWindows()

