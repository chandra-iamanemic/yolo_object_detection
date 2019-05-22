import cv2
import os
import numpy as np
import wget

initialize = True
net = None
dest_dir = os.path.expanduser('~') + os.path.sep + '.cvlib' + os.path.sep + 'object_detection' + os.path.sep + 'yolo' + os.path.sep + 'yolov3'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

classes = None
COLORS = np.random.uniform(0, 255, size=(80, 3))

def populate_class_labels(class_file_url):

    class_file_name = 'yolov3_classes.txt'
    class_file_abs_path = dest_dir + os.path.sep + class_file_name
    url = class_file_url
    if not os.path.exists(class_file_abs_path):
        wget.download(url, os.path.join(dest_dir,class_file_name))

    f = open(class_file_abs_path, 'r')
    classes = [line.strip() for line in f.readlines()]

    return classes


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_bbox(img, bbox, labels, confidence, colors=None, write_conf=False):

    global COLORS
    global classes

    if classes is None:
        classes = populate_class_labels()
    
    for i, label in enumerate(labels):

        if colors is None:
            color = COLORS[classes.index(label)]            
        else:
            color = colors[classes.index(label)]

        if write_conf:
            label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'

        cv2.rectangle(img, (bbox[i][0],bbox[i][1]), (bbox[i][2],bbox[i][3]), color, 2)

        cv2.putText(img, label, (bbox[i][0],bbox[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img
    
def detect_common_objects(image, confidence=0.5, nms_thresh=0.1, model='yolov3-tiny',
                          class_file_url = 'https://raw.githubusercontent.com/chandra-iamanemic/yolo_object_detection/master/yolov3.txt',
                          cfg_url_yolov3 = 'https://raw.githubusercontent.com/chandra-iamanemic/yolo_object_detection/master/yolov3-hands.cfg', 
                          cfg_url_yolov2 = "https://raw.githubusercontent.com/chandra-iamanemic/yolo_object_detection/master/yolov2-tiny-hands.cfg", 
                          weights_url_yolov3 = 'https://drive.google.com/uc?export=download&id=1pwXyTHf2UJgiQPoQF6D7IEhWCRQtUUOh',
                          weights_url_yolov2 = 'https://drive.google.com/uc?export=download&id=1BwV7rE0ejqYKJdUvj5KO7hOL-oENpv0l'):
        
    Height, Width = image.shape[:2]
    scale = 0.00392

    global classes
    global dest_dir

    if model == 'yolov2-tiny':
        config_file_name = 'yolov2-tiny.cfg'
        cfg_url = cfg_url_yolov2
        weights_file_name = 'yolov2-tiny.weights'
        weights_url = weights_url_yolov2
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        config_file_abs_path = dest_dir + os.path.sep + config_file_name
        weights_file_abs_path = dest_dir + os.path.sep + weights_file_name    


    else:
        config_file_name = 'yolov3-tiny.cfg'
        cfg_url = cfg_url_yolov3
        weights_file_name = 'yolov3-tiny.weights'
        weights_url = weights_url_yolov3
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)    
    
        config_file_abs_path = dest_dir + os.path.sep + config_file_name
        weights_file_abs_path = dest_dir + os.path.sep + weights_file_name    
    
    if not os.path.exists(config_file_abs_path):
        print("Downloading Yolo Config File from: " + cfg_url )
        wget.download(cfg_url, config_file_abs_path)

    if not os.path.exists(weights_file_abs_path):
        print("Downloading Pre-Trained Weights from: " + weights_url)
        wget.download(weights_url, weights_file_abs_path)  

    global initialize
    global net

    if initialize:
        classes = populate_class_labels(class_file_url)
        net = cv2.dnn.readNet(weights_file_abs_path, config_file_abs_path)
        initialize = False

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            max_conf = scores[class_id]
            if max_conf > confidence:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(max_conf))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_thresh)

    bbox = []
    label = []
    conf = []

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        bbox.append([round(x), round(y), round(x+w), round(y+h)])
        label.append(str(classes[class_ids[i]]))
        conf.append(confidences[i])
        
    return bbox, label, conf
