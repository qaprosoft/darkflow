import numpy as np
import random
import string
import math
import cv2
import os
import json
import itertools
import shutil
from ...utils.box import BoundBox
from ...cython_utils.cy_yolo2_findboxes import box_constructor
from . import OCR
from math import sqrt
from joblib import Parallel, delayed
import subprocess
from multiprocessing import Pool
from functools import partial


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
	result["caption"] = ''
	for word in words:
		left, top, right, bot = result["topleft"]["x"], result["topleft"]["y"], result["bottomright"]["x"], result["bottomright"]["y"]
		rect = left, top, right, bot
		if _get_overlap_rectangle_area(word[:4], rect) >= _get_area_of_word(word[:4]):
			result["caption"] = result["caption"] + " " + word[-1]
			result["caption"] = result["caption"].lstrip()
	return result


def _create_dir_if_not_exists(path):
	if not os.path.exists(path):
		os.mkdir(path)


def crop_image_into_boxes(im, outdir, labels, result_list):
	"""
	Crops an original image into a list of images with found captions
	:param im: original image as np array
	:param outdir: path where the crops is written
	:param result_list: result dict with captions
	"""
	result_entry = result_list[0]
	x_begin, x_end = result_entry['topleft']['x'], result_entry['bottomright']['x']
	y_begin, y_end = result_entry['topleft']['y'], result_entry['bottomright']['y']
	cropped = im[y_begin:y_end, x_begin:x_end]
	result_path = ""
	if result_entry['label'] in labels:
		result_path = os.path.join(outdir, result_entry['label'])
		_create_dir_if_not_exists(result_path)
	cropped_path = "{}/{}.png".format(result_path, result_entry.get('caption') if result_entry.get('caption') not in ('', ' ') else ''.join(random.sample((string.digits), 5))).replace(" ", "_")
	if len(result_list) == 1:
		cv2.imwrite(cropped_path, cropped)
		return
	cv2.imwrite(cropped_path, cropped)
	return crop_image_into_boxes(im, outdir, labels, result_list[1:])


def get_list_of_label_types(result_list):
	"""
	Retrieves the all labels types from results of OCR (recursive mode)
	:param result_list: dict with results of OCR
	:return: unique set of image label types
	"""
	return {result['label'] for result in result_list}


def get_path_entries(predicate, path):
	"""
	:param predicate: filter criteria for filtering directory entries
	:param path: path to dir for searching
	"""
	return list(map(lambda x: x.path, filter(predicate, os.scandir(path))))


def get_obj_from_json(path):
	with open(path, 'r') as f:
		return json.load(f)


def merge_jsones_from_recursive_call(folders, results):
	"""
	Merge jsones from recursive call into result json and then deletes folders
	:param folders: a list of folders contains jsones
	:param results: result list from first non-recursive call
	:return: merged list of results
	"""
	models_to_labels = {
		"team": "label",
		"logo": "logo"
	}

	path_splitter = '/' if os.name == 'posix' else '\\'
	for folder in folders:
		out_folder = os.path.join(folder, 'out')
		json_paths = get_path_entries(lambda x: x.path.endswith('.json'), out_folder)
		jsones = list()
		for single_path in json_paths:
			obj_from_json = get_obj_from_json(single_path)
			jsones.append(obj_from_json)
		for result in results:
			model = folder.split(path_splitter)[-1]
			for json_list in jsones:
				for single_json in json_list:
					if single_json['label'] == models_to_labels[model] and single_json['caption'].lower() in result['caption'].lower():
						result[model] = None
						result[model] = json_list
						break
		shutil.rmtree(folder)
	return results


def get_captions_from_image(self, im):
	"""
	:param im: image as np array
	:return: list of captured text from an image
	"""
	if self.FLAGS.ocr_gamma:
		prepared = OCR.OCR.prepare_image_for_recognition_using_gammas(im=im, gamma=self.FLAGS.ocr_gamma)
		return OCR.OCR.get_boxes_from_prepared_image(im=prepared)
	prepared = OCR.OCR.prepare_image_for_recognition_using_thresholding(im=im)
	return OCR.OCR.get_boxes_from_prepared_image(im=prepared)


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

	if self.FLAGS.json:
		captions = self.get_captions_from_image(imgcv)
		if resultsForJSON:
			for result in resultsForJSON:
				result = append_text_to_result_json(result, captions)
			find_labels_for_controls(resultsForJSON)

		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"

		if self.FLAGS.recursive_models:
			models_from_cli = set(self.FLAGS.recursive_models.split(","))
			label_types = {result['label'] for result in resultsForJSON}
			labels_to_recognize = label_types.intersection(models_from_cli)
			model_paths = ['/qps-ai/darkflow/cfg/' + model + '.cfg' for model in labels_to_recognize]
			crop_image_into_boxes(imgcv, outfolder, labels_to_recognize, resultsForJSON)
			folders_to_recognize = [os.path.join(outfolder, label) for label in labels_to_recognize]
			label_paths = ['/qps-ai/darkflow/labels-' + label + '.txt' for label in labels_to_recognize]
			backup_paths = ['/qps-ai/darkflow/ckpt/' + backup + '/' for backup in labels_to_recognize]
			call_with_fixed_shell = partial(subprocess.run, shell=True)
			generation_command = "/qps-ai/darkflow/flow --model {} --load -1 --imgdir {} --json --labels {} --backup {}"
			labels_path = "/qps-ai/darkflow/cfg"
			if self.FLAGS.ocr_gamma:
				generation_command += " --ocr_gamma " + str(self.FLAGS.ocr_gamma)
			arg_pairs = list(zip(model_paths, folders_to_recognize, label_paths, backup_paths))
			commands = [generation_command.format(*arg_pair) for arg_pair in arg_pairs]
			for command in commands:
				call_with_fixed_shell(command)
			resultsForJSON = merge_jsones_from_recursive_call(folders_to_recognize, resultsForJSON)
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"

		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)
