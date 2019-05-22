# yolo_object_detection
cvlib is a high level easy-to-use open source Computer Vision library for Python.

This package is a slightly modified version of cvlib which by default loads my weights 
for detecting two classes (Open hand and Closed Hand).

[link to original package](https://github.com/arunponnusamy/cvlib) 

You can run the cvlib detect_common_objects function with your own files which are namely :
class file = (text file containing the names of the distinct classes)
config file = (.cfg file containing the yolo config (used when training the network))
weights = Trained weights for your custom classes (you can train it with darknet)

you can add these files with a downloadable url which can be fed into the detect_common_objects function 
using these parameter inputs (class_file_url,cfg_url_yolov3,cfg_url_yolov2,weights_url_yolov3,weights_url_yolov2)
and for the model parameter you can either give yolov3_tiny or yolov2_tiny based on what you have trained.

## How to setup the package

Download this package and move into cvlib-master

Run the following commands :
cd cvlib-master
python setup.py sdist
pip install .

Note : Use python 3.x (preferrable)

## Webcam live detector

Once the packages are installed run the object_detection_webcam_yolov3_tiny.py

