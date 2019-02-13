import numpy as np
import random
import string
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
from pprint import pprint


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
            if delta_left_y > item['bottomright']['y'] - item['topleft']['y']:
                delta_left_y = (item['bottomright']['y'] - item['topleft']['y']) * 0.5
                delta_right_y = (item['bottomright']['y'] - item['topleft']['y']) * 0.5
            if delta_left_x > item['bottomright']['x'] - item['topleft']['x']:
                delta_left_x = (item['bottomright']['x'] - item['topleft']['x']) * 0.3
                delta_right_x = (item['bottomright']['x'] - item['topleft']['x']) * 0.3

    for control in controls:
        label = find_left_label(control, labels, delta_left_x, delta_left_y)
        if label is None:
            label = find_top_label(control, labels, delta_top_x, delta_top_y)
        if label is None:
            label = find_right_label(control, labels, delta_right_x, delta_right_y)
        if label is not None:
            control['caption'] = label.get('caption')



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


def _get_overlap_rectangle_area(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	return max(0, xB - xA + 1) * max(0, yB - yA + 1)


def _get_area_of_word(box):
	return abs(box[2] - box[0]) * abs(box[3] - box[1])


def append_text_to_result_json(result, words):
	"""
		Appends the captions from OCR into the labels from nnet
		:param result: a single dict record from the nnet result
		:param words: list of captions from OCR
		:return result: modified value with caption text
	"""
	areas = []
	result["caption"] = ""
	for word in words:
		left, top, right, bot = result["topleft"]["x"], result["topleft"]["y"], result["bottomright"]["x"], result["bottomright"]["y"]
		rect = left, top, right, bot
		if _get_overlap_rectangle_area(word[:4], rect) >= _get_area_of_word(word[:4]):
			result["caption"] = result["caption"] + " " + word[-1]
	return result


def crop_image_into_boxes(im, outdir, result_list):
	"""
		Crops an original image into a list of images with found captions
		:param im: original image as np array
		:param outdir: path where the crops is written
		:param result_list: result dict with captions
	"""
	x_begin, x_end = result_list[0]['topleft']['x'], result_list[0]['bottomright']['x']
	y_begin, y_end = result_list[0]['topleft']['y'], result_list[0]['bottomright']['y']
	cropped = im[x_begin:x_end, y_begin:y_end]
	if len(result_list) == 1:
		cv2.imwrite("{}/{}.png".format(outdir, ''.join(random.sample((string.ascii_lowercase + string.digits), 10))), cropped)
		return
	cv2.imwrite("{}/{}.png".format(outdir, ''.join(random.sample((string.ascii_lowercase + string.digits), 10))), cropped)
	return crop_image_into_boxes(im, outdir, result_list[1:])


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

	prepared = None
	captions = list()
	if self.FLAGS.json:
		if self.FLAGS.threshold_prep == True and self.FLAGS.gamma == 1.0:
			prepared = OCR.OCR.prepare_image_for_recognition_using_thresholding(im=imgcv)
			captions = OCR.OCR.get_boxes_from_prepared_image(im=prepared)
		elif self.FLAGS.threshold_prep == False and self.FLAGS.gamma != 1.0:
			prepared = OCR.OCR.prepare_image_for_recognition_using_gammas(im=imgcv, gamma=self.FLAGS.gamma)
			captions = OCR.OCR.get_boxes_from_prepared_image(im=prepared)
		else:
			captions = OCR.OCR.get_boxes_from_unprepared_image(im=im)
		if resultsForJSON:
			for result in resultsForJSON:
				result = append_text_to_result_json(result, captions)
			find_labels_for_controls(resultsForJSON)
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"

		# uncomment these lines when we want to crop image by its bounding boxes
		# crop_path = os.path.join(outfolder, 'crops')
		# os.mkdir(crop_path)
		# crop_image_into_boxes(imgcv, crop_path, resultsForJSON)

		with open(textFile, 'w') as f:
			f.write(textJSON)

		return

	cv2.imwrite(img_name, imgcv)
