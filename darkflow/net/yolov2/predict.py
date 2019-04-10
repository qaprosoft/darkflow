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
from pprint import pprint

DARKFLOW_HOME = os.environ.get('DARKFLOW_HOME')


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


def _get_overlap_rectangle_area(word, res):
	r1l = min(word[0], word[2])
	r1r = max(word[0], word[2])
	r1t = min(word[1], word[3])
	r1b = max(word[1], word[3])

	r2l = min(res[0], res[2])
	r2r = max(res[0], res[2])
	r2t = min(res[1], res[3])
	r2b = max(res[1], res[3])

	left = max(r1l, r2l)
	right = min(r1r, r2r)
	top = max(r1t, r2t)
	bottom = min(r1b, r2b)

	return max(right - left, 0) * max(bottom - top, 0)


def _get_area_of_word(box):
	return abs(box[2] - box[0]) * abs(box[3] - box[1])


def append_text_to_result_json(result, words):
	"""
	Appends the captions from OCR into the labels from nnet
	:param result: a single dict record from the nnet result
	:param words: list of captions from OCR
	:return result, unmerged_captions: typle with  modified value with caption text and captions that doesnt appear inside label
	"""
	result["caption"] = ''
	unmerged_captions = ''
	for word in words:
		left, top, right, bot = result["topleft"]["x"], result["topleft"]["y"], result["bottomright"]["x"], result["bottomright"]["y"]
		rect = left, top, right, bot
		if _get_overlap_rectangle_area(word[:4], rect) / _get_area_of_word(word[:4]) >= 0.6:
			result["caption"] = result["caption"] + " " + word[-1]
			result["caption"] = result["caption"].lstrip()
		else:
			unmerged_captions = unmerged_captions + ' ' + word[-1]
			unmerged_captions = unmerged_captions.lstrip()
	return result, unmerged_captions


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
	for result_entry in result_list:
		if 'tesseract' in result_entry['label']:
			continue
		x_begin, x_end = result_entry['topleft']['x'], result_entry['bottomright']['x']
		y_begin, y_end = result_entry['topleft']['y'], result_entry['bottomright']['y']
		cropped = im[y_begin:y_end, x_begin:x_end]
		result_path = ""
		if result_entry['label'] in labels:
			result_path = os.path.join(outdir, result_entry['label'])
			_create_dir_if_not_exists(result_path)
		cropped_path = "{}/{}-{}-{}-{}.png".format(result_path, x_begin, y_begin, x_end, y_end)
		cv2.imwrite(cropped_path, cropped)


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
	path_splitter = '/' if os.name == 'posix' else '\\'
	for folder in folders:
		out_folder = os.path.join(folder, 'out')
		json_paths = get_path_entries(lambda x: x.path.endswith('.json'), out_folder)
		json_names = [json_path.split(path_splitter)[-1].split('.')[0] for json_path in json_paths]
		jsones = dict()
		for json_name in json_names:
			for json_path in json_paths:
				if json_name in json_path:
					jsones[json_name] = get_obj_from_json(json_path)

		for result in results:
			model = folder.split(path_splitter)[-1]
			for k in jsones.keys():
				topleft_x, topleft_y = int(k.split('-')[0]), int(k.split('-')[1])
				botright_x, botright_y = int(k.split('-')[2]), int(k.split('-')[3])
				if topleft_x == result["topleft"]["x"] and topleft_y == result["topleft"]["y"] and botright_x == result['bottomright']['x'] and botright_y == result['bottomright']['y']:
					result[model] = None
					result[model] = jsones[k]
					break

		shutil.rmtree(folder)
	return results


def get_captions_from_image(self, im, resize_coef):
	"""
	:param im: image as np array
	:param resize_coef: for resizing an original image
	:return: list of captured text from an image
	"""
	if self.FLAGS.model == DARKFLOW_HOME + "/cfg/nhl.cfg" and not self.FLAGS.ocr_gamma:
		resize_coef = 1  # no need to resize that image
		prepared = OCR.OCR.prepare_nhl_image_for_recognition(im=im, resize_coef=resize_coef)
		return OCR.OCR.get_boxes_from_prepared_image(im=prepared, resize_coef=resize_coef)
	if self.FLAGS.ocr_gamma:
		prepared = OCR.OCR.prepare_image_for_recognition_using_gammas(im=im, gamma=self.FLAGS.ocr_gamma, resize_coef=resize_coef)
		return OCR.OCR.get_boxes_from_prepared_image(im=prepared, resize_coef=resize_coef)
	prepared = OCR.OCR.prepare_image_for_recognition_using_thresholding(im=im, resize_coef=resize_coef)
	return OCR.OCR.get_boxes_from_prepared_image(im=prepared, resize_coef=resize_coef)


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
			if confidence > 0.25:
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
		resize_coefficient = 2
		captions = self.get_captions_from_image(imgcv, resize_coefficient)
		unmerged_captions = ''
		if resultsForJSON:
			for result in resultsForJSON:
				appended_result = append_text_to_result_json(result, captions)
				result = appended_result[0]
				unmerged_captions = appended_result[1]

		resultsForJSON.append({"label": "tesseract", "caption": unmerged_captions})

		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"

		if self.FLAGS.recursive_models:
			models_from_cli = set(self.FLAGS.recursive_models.split(","))
			model_paths = [DARKFLOW_HOME + '/cfg/' + model + '.cfg' for model in models_from_cli]
			crop_image_into_boxes(imgcv, outfolder, models_from_cli, resultsForJSON)
			folders_to_recognize = [os.path.join(outfolder, label) for label in models_from_cli]
			label_paths = [DARKFLOW_HOME + '/labels-' + label + '.txt' for label in models_from_cli]
			backup_paths = [DARKFLOW_HOME + '/ckpt/' + backup + '/' for backup in models_from_cli]
			call_with_fixed_shell = partial(subprocess.run, shell=True)
			generation_command = DARKFLOW_HOME + "/flow --model {} --load -1 --imgdir {} --json --labels {} --backup {}"
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
