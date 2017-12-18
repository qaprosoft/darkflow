import numpy as np
import math
import cv2
import os
import json
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor
from . import OCR
from math import sqrt
from joblib import Parallel, delayed
import multiprocessing

def recognize_label(dictt, distance_dict, ocr, image):

	x1_label = dictt["topleft"]["x"]
	#print (x1_label)
	y1_label = dictt["topleft"]["y"]
	#print (y1_label)
	x2_label = dictt["bottomright"]["x"]
	#print (x2_label)
	y2_label = dictt["bottomright"]["y"]
	#print (y2_label)
	#print ("14")
	label_coordinate = (x1_label, y1_label, x2_label, y2_label)

	#print ("1")
	#print (distance_dict)
	#if label_coordinate in distance_dict:
	    #print ("3")
	 #   caption_coordinate = distance_dict[label_coordinate]
	  #  recognized = str(ocr.recognize_caption_v2(caption_coordinate, image=image))
	   # dictt["caption"] = recognized#{"caption":recognized, "coord":caption_coordinate}
	#else:

	caption_coordinate = label_coordinate

	recognized = str(ocr.recognize_caption_v2(caption_coordinate, image=image))
	#print ("15")
	dictt["caption"] = recognized#{"caption":recognized, "coord":caption_coordinate}
	#if dictt["label"] != "label":
	return dictt


def _get_center_coordinate(coordinates):
    x1 = coordinates[0]
    y1 = coordinates[1]
    x2 = coordinates[2]
    y2 = coordinates[3]

    x_center = int((x1 + x2)/2)
    y_center = int((y1 + y2)/2)

    return x_center, y_center


def _get_distance(coordinate_1, coordinate_2):
    x_center_1, y_center_1 = _get_center_coordinate(coordinate_1)
    x_center_2, y_center_2 = _get_center_coordinate(coordinate_2)

    distance = sqrt(abs(x_center_1 - x_center_2)**2 + abs(y_center_2 - y_center_1)**2)

    return int(distance)


def _get_min_distance_coordinates(config, coordinate, _dict, _class="label"):
    """
        Count distances between given coordinate and rest coordinates
        Args:
            config - json file
            coordinate - (x1, y1, x2, y2)
            _dict - current dict of nearest coordinates to the caption
            label - label of coordinate

        return coordinate of the nearest label to the current caption
    """
    distances = {} # (coordinate): distance
    for dictt in config:
        if dictt["label"] != _class:
            x1 = dictt["topleft"]["x"]
            y1 = dictt["topleft"]["y"]
            x2 = dictt["bottomright"]["x"]
            y2 = dictt["bottomright"]["y"]
            caption_coordinate = coordinate
            label_coordinates = (x1, y1, x2, y2)

            distance = _get_distance(caption_coordinate, label_coordinates)
            distances[label_coordinates] = distance

    sorted_values = distances.values()
    if not sorted_values:
        return []

    sorted_values = sorted(sorted_values)
    #print ("config {}".format(config))
    for distance in sorted_values:

        min_distance = distance
        min_distance_coordinates = list(distances.keys())[list(distances.values()).index(min_distance)]
        if min_distance_coordinates in _dict:
            continue
        else:
            break

    return min_distance_coordinates


def get_distance_dict(config, _class="label"):
    """
        returns dict of captions with coordinates of the nearest label.
        Will return emty dict if there is no _class elements in the image
        length of the dict == number of _class elements in the config
    """
    distance_dict = {} # key - caption coordinate, value - coordinate of the nearest label
    for dictt in config:
        if dictt["label"] == _class:
            x1 = dictt["topleft"]["x"]
            y1 = dictt["topleft"]["y"]
            x2 = dictt["bottomright"]["x"]
            y2 = dictt["bottomright"]["y"]
            caption_coordinate = (x1, y1, x2, y2)

            min_distance_coordinate = _get_min_distance_coordinates(config, caption_coordinate, distance_dict)
            if not min_distance_coordinate:
                 return {}
            distance_dict[min_distance_coordinate] = caption_coordinate
    return distance_dict


def expit(x):
	return 1. / (1. + np.exp(-x))


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes


def find_labels_for_controls(JSONResult):
    delta_left_x = 50.0
    delta_left_y = 7.0
    delta_top_x = 5
    delta_top_y = 5
    delta_right_x = 50
    delta_right_y = 5
    labels = []
    controls = []
    for item in JSONResult:
        if item['label'] == 'label':
            labels.append(item)
        elif item['label'] != 'button':
            controls.append(item)

	for item in controls:
		if item['label'] == 'text_field':
			if delta_left_y > item['topleft']['y'] - item['bottomright']['y']:
				delta_left_y = (item['topleft']['y'] - item['bottomright']['y']) * 0.5
				delta_right_y = (item['topleft']['y'] - item['bottomright']['y']) * 0.5
				item['delta_y'] = delta_left_y
			if delta_left_x > item['bottomright']['x'] - item['topleft']['x']:
				delta_left_x = (item['bottomright']['x'] - item['topleft']['x']) * 0.3
				delta_right_x = (item['bottomright']['x'] - item['topleft']['x']) * 0.3
				item['delta_x'] = delta_left_y

    for control in controls:
        label = find_left_label(control, labels, delta_left_x, delta_left_y)
        if label is None:
            label = find_top_label(control, labels, delta_top_x, delta_top_y)
        if label is None:
            label = find_right_label(control, labels, delta_right_x, delta_right_y)
        if label is not None:
            control['caption'] = label['caption']



def find_left_label(control, labels, delta_x, delta_y):
	for label in labels:
		if (control['bottomright']['y'] + delta_y > label['bottomright']['y']) and (label['bottomright']['y'] > control['bottomright']['y'] - delta_y):
			if (control['topleft']['x'] + delta_x > label['bottomright']['x']) and (label['bottomright']['x'] > control['topleft']['x'] - delta_x):
				return label
		else:
			continue
	return None


def find_top_label(control, labels, delta_x, delta_y):
    for label in labels:
		if (control['topleft']['x'] + delta_x > label['topleft']['x']) and (label['topleft']['x'] > control['topleft']['x'] - delta_x):
			if (control['topleft']['y'] + delta_y > label['bottomright']['y']) and (label['bottomright']['y'] > control['topleft']['y'] - delta_y):
				return label
		else:
			continue
    return None


def find_right_label(control, labels, delta_x, delta_y):
	for label in labels:
		if (control['topleft']['y'] + delta_y > label['topleft']['y']) and (label['topleft']['y'] > control['topleft']['y'] - delta_y):
			if (control['bottomright']['x'] + delta_x > label['topleft']['x']) and (label['topleft']['x'] > control['bottomright']['x'] - delta_x):
				return label
		else:
			continue
	return None


def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape

	resultsForJSON = []
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = 2 #int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		cv2.putText(imgcv, mess, (left, top - 12),
			0, 1e-3 * h, colors[max_indx],thick - 1//3)

	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))


	# Adding caption tag
	ocr = OCR.OCR()
	distance_dict = get_distance_dict(resultsForJSON)
	resultsForJSON_v2 = []
	if resultsForJSON:
		#print ("12")
		for dictt in resultsForJSON:
            # print ("13")
            # print (dictt)
			resultsForJSON_v2.append(recognize_label(dictt, distance_dict, ocr, imgcv))

		#resultsForJSON_v2 = Parallel(n_jobs=-1, backend="threading")(delayed(recognize_label)(dictt,
         #   distance_dict, ocr, imgcv) for dictt in resultsForJSON)

	if resultsForJSON:
		find_labels_for_controls(resultsForJSON)

	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)
