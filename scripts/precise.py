import os
import sys
import codecs
import json
import xml.etree.ElementTree as ET
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--truth", default="/home/ubuntu/tools/alice/data/ideal/test/ann/", help="path to ground truth annotations folder")
parser.add_argument("--predicted", default="/home/ubuntu/tools/alice/data/ideal/test/out/test/", help="path to predicted boxes folder")
parser.add_argument("--threshold", default=0.5, help="threshold for IOU score")
parser.add_argument("--center", default=True, help="Whether to show Center Score")
parser.add_argument("--iou", default=True, help="Whether to show IOU Score")
parser.add_argument("--labels", default="/home/ubuntu/darkflow/labels.txt", help="path to labels.txt")
args = parser.parse_args()


def isCenterMatch(true, predicted):
    """
    Defines whether true and predicted intersects or not

    Args:
        true - [filename, label, xmin, ymin, xmax, ymax]
        predicted - [filename, label, xmin, ymin, xmax, ymax]
    """
    # predicted coordinates
    xmin_predicted = predicted[2]
    ymin_predicted = predicted[3]
    xmax_predicted = predicted[4]
    ymax_predicted = predicted[5]

    # center of the predicted box
    predictedCenter = (int((xmax_predicted - xmin_predicted)/2.0+xmin_predicted),
                    int((ymax_predicted - ymin_predicted)/2.0)+ymin_predicted)

    # true coordinates
    xmin_true = true[2]
    ymin_true = true[3]
    xmax_true = true[4]
    ymax_true = true[5]

    # center of the true box
    trueCenter = (int((xmax_true - xmin_true)/2.0+xmin_true),
                int((ymax_true - ymin_true)/2.0)+ymin_true)

    if isPointInsideRect(predictedCenter, true) or isPointInsideRect(trueCenter, predicted):
        return 1
    else:
        return 0

def isPointInsideRect(point, rectangle):
    """
        defines whether point inside rectangle or not
    """
    x, y = point[0], point[1]

    xmin = rectangle[2]
    ymin = rectangle[3]
    xmax = rectangle[4]
    ymax = rectangle[5]

    if xmin<=x and ymin<=y and xmax>=x and ymax>=y:
        return True
    else:
        return False

def find_intersection_square(predicted, true):
    """
        returns intersection square between rectangles
    """
    # predicted coordinates
    xmin_predicted = predicted[2]
    ymin_predicted = predicted[3]
    xmax_predicted = predicted[4]
    ymax_predicted = predicted[5]

    # true coordinates
    xmin_true = true[2]
    ymin_true = true[3]
    xmax_true = true[4]
    ymax_true = true[5]

    dx = abs(min(xmax_predicted, xmax_true) - max(xmin_predicted, xmin_true))
    #print min(xmax_predicted, xmax_true)
    dy = abs(min(ymax_predicted, ymax_true) - max(ymin_predicted, ymin_true))
    #print "dx*dy = {}".format(dx*dy)
    if (dx>=0) and (dy>=0):
        #print "dx*dy = ".format(dx*dy)
        return dx*dy

def find_square(coordinates):
    """
        coordinates - [filename, label,xmin,ymin,xmax,ymax]
        returns square for rectangle with given coordinates (coords)
    """
    xmin = coordinates[2]
    ymin = coordinates[3]
    xmax = coordinates[4]
    ymax = coordinates[5]
    width = xmax - xmin
    height = ymax - ymin
    return width * height

def IOU(predicted, true):
    """
    returns intersection over union for rectangles with given coordinates
    """
    predicted_square = find_square(predicted) #square of predicted box
    true_square = find_square(true) # square of ground truth box
    sum_of_squares = predicted_square + true_square

    intersection = find_intersection_square(predicted, true)
    if intersection != sum_of_squares:
        union = sum_of_squares - intersection
        return float(intersection)/union
    else:
        warning = "WARNING!\nIntersection = {}\nSum of squares = {}\npredicted box - {}\ntrue box - {}"
        print warning.format(intersection,
                            sum_of_squares, predicted, true)
        return 0

def get_list_of_classes(filename):
    return open(filename).read().splitlines()

def get_class_measures_dict(classes):
    TP, FP, TN, AP = 0, 0, 0, 0
    measure = [TP, FP, TN, AP]
    class_measures_dict = {}
    for class_ in classes:
        class_measures_dict[class_] = [0,0,0,0]
    return class_measures_dict

# Parse xml annotations
ann_folder = args.truth
dumps_xml = {}

for file in glob.glob(ann_folder + '/' + '*.xml'):
    annotation_file = open(file)
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    # filename without extension
    filename = str(root.find('filename').text).rsplit('.', 1)[0]

    for obj in root.iter('object'):
        current = list()
        label = obj.find('name').text

        xmlbox = obj.find('bndbox')
        xmin = int(float(xmlbox.find('xmin').text))
        xmax = int(float(xmlbox.find('xmax').text))
        ymin = int(float(xmlbox.find('ymin').text))
        ymax = int(float(xmlbox.find('ymax').text))
        current = (filename, label, xmin, ymin, xmax, ymax)
        dumps_xml[current] = 0
    annotation_file.close()

# Parse recognized metadata from json format
json_folder = args.predicted
dumps_json = {}
for file in glob.glob(json_folder + '/' + '*.json'):
    try:
        config = json.loads(codecs.open(file, "r", encoding='utf-8', errors='ignore').read())
    except ValueError:
        print "{} does not have any bounding box".format(file)
        continue

    for dictionary in config:
        label = dictionary["label"]
        xmin = dictionary["topleft"]["x"]
        ymin = dictionary["topleft"]["y"]
        xmax = dictionary["bottomright"]["x"]
        ymax = dictionary["bottomright"]["y"]

        filename = os.path.basename(file).rsplit('.', 1)[0]
        current = (filename, label, xmin, ymin, xmax, ymax)
        dumps_json[current] = 0

classes = get_list_of_classes(args.labels)
class_measures_dict_iou = get_class_measures_dict(classes)

#Intersection Over Union
threshold = args.threshold
class_measures_dict_iou = get_class_measures_dict(classes)
for predicted_coordinate in dumps_json:
    for true_coordinate in dumps_xml:
        if predicted_coordinate[0] == true_coordinate[0] and dumps_xml[true_coordinate] == 0:
            iou_value = IOU(true = true_coordinate, predicted = predicted_coordinate)
            if iou_value >= threshold:

                # filling class measures
                for class_ in classes:
                    true_label = true_coordinate[1]
                    predicted_label = predicted_coordinate[1]

                    if true_label == predicted_label:
                        if true_label == class_:
                            dumps_xml[true_coordinate] = 1
                            dumps_json[predicted_coordinate] = 1
                            class_measures_dict_iou[class_][0] += 1
                            break
                    else:
                        class_measures_dict_iou[predicted_label][1] += 1
                        break

for key in dumps_json:
    if dumps_json[key] == 0: # key - (filename, label, xmin, ymin, xmax, ymax)
        label = key[1]
        if label != 'None':
            class_measures_dict_iou[label][2] += 1

for key in dumps_xml:
    if dumps_xml[key] == 0: # key - (filename, label, xmin, ymin, xmax, ymax)
        label = key[1]
        if label != 'None':
            class_measures_dict_iou[label][2] += 1

AP_list = []
for class_ in classes:
    TP = class_measures_dict_iou[class_][0]
    FP = class_measures_dict_iou[class_][1]
    TN = class_measures_dict_iou[class_][2]
    #print TP, FP, TN
    if TP != 0 or FP != 0 or TN != 0:
        AP = float(TP)/(TP+FP+TN)
        class_measures_dict_iou[class_][3] = AP
        AP_list.append(AP)
IOU_mAP = np.mean(AP_list)

# mAP with centers
# making all values equals 0
for key in dumps_json:
    dumps_json[key] = 0
for key in dumps_xml:
    dumps_xml[key] = 0

class_measures_dict = get_class_measures_dict(classes)
for predicted_coordinate in dumps_json:
    for true_coordinate in dumps_xml:
        if predicted_coordinate[0] == true_coordinate[0] and dumps_xml[true_coordinate] == 0:
            if isCenterMatch(true_coordinate, predicted_coordinate) == 1:

                # filling class measures
                for class_ in classes:
                    true_label = true_coordinate[1]
                    predicted_label = predicted_coordinate[1]

                    if true_label == predicted_label:
                        if true_label == class_:
                            dumps_xml[true_coordinate] = 1
                            dumps_json[predicted_coordinate] = 1
                            class_measures_dict[class_][0] += 1
                            break
                    else:
                        class_measures_dict[predicted_label][1] += 1
                        break

for key in dumps_json:
    if dumps_json[key] == 0: # key - (filename, label, xmin, ymin, xmax, ymax)
        label = key[1]
        if label != 'None':
            class_measures_dict[label][2] += 1

for key in dumps_xml:
    if dumps_xml[key] == 0: # key - (filename, label, xmin, ymin, xmax, ymax)
        label = key[1]
        if label != 'None':
            class_measures_dict[label][2] += 1

AP_list = []
for class_ in classes:
    TP = class_measures_dict[class_][0]
    FP = class_measures_dict[class_][1]
    TN = class_measures_dict[class_][2]
    if TP != 0 or FP != 0 or TN != 0:
        AP = float(TP)/(TP+FP+TN)
        class_measures_dict[class_][3] = AP
        AP_list.append(AP)
centre_mAP = np.mean(AP_list)

bool_center = args.center
bool_iou = args.iou
if bool_center:
    print "CENTRE_score - {}".format(centre_mAP)
    for class_ in classes:
        AP = class_measures_dict[class_][3]
        print "{} - {}".format(class_, AP)
if bool_iou:
    print "IOU_mAP - {}".format(IOU_mAP)
    for class_ in classes:
        AP = class_measures_dict_iou[class_][3]
        print "{} - {}".format(class_, AP)


if not (bool_iou or bool_center):
    print "You didn't set any flag. Run precise.py -h for more information"
