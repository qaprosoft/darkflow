import os
import json
import cv2
from multiprocessing import Process
from .yolov2.OCR import OCR


class OCRProcessor(object):
	"""
	Provides methods for recognizing text from image and save result jsones
	"""
	RESIZE_COEFFICIENT = 1  # passes to preprocess and ocr-ing image

	def __init__(self, imgdir):
		self.imgdir = imgdir
		self.images = [im.path for im in os.scandir(self.imgdir) 
						if im.path.endswith('.png')]
		assert self.images, 'Nothing to recognize'
		self.outdir = os.path.join(self.imgdir, 'out')

	def process(self):
		for image in self.images:
			p = Process(target=_apply_ocr_to_image, args=(image, self.outdir, ))
			p.start()
			p.join

def _apply_ocr_to_image(image_path, outdir):
	imgcv = cv2.imread(image_path)
	prepared = OCR.prepare_image_for_recognition_using_thresholding(
			imgcv, 
			OCRProcessor.RESIZE_COEFFICIENT
	)
	boxes = OCR.get_boxes_from_prepared_image(
			prepared, 
			OCRProcessor.RESIZE_COEFFICIENT
	)
	output_dict = {'label': 'tesseract', 'caption': ''}
	for box in boxes:
		output_dict['caption'] = output_dict['caption'] + ' ' + box[-1]
	output_dict['caption'].lstrip()
	json_name = os.path.splitext(image_path)[0].split(os.sep)[-1] + '.json'
	print(json_name)
	json_path = os.path.join(outdir, json_name)
	print(json_path)
	with open(json_path, 'w') as f:
		json.dump(output_dict, f)
