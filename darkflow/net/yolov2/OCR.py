"""
Script takes each bounding box in each annotation, makes different crops of
each bounding box, recognizes text on the crop and then delete cropped image
"""

import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import fromstring, Element
import pytesseract as pts
from PIL import Image
import glob
from skimage.filters import threshold_sauvola

class OCR:

    def _get_image_by_basename(self, basename, image_folder):
        for file in os.listdir(image_folder):
            #print ("1-6")
            file_base = os.path.basename(file).rsplit(".", 1)[0]
            if file_base == basename:
                return file
            else:
                continue
        raise ValueError

    @staticmethod
    def prepare_image_for_recognition_using_gammas(im, gamma):
        """
            Prepares an whole image for recognition using gamma-correction method. 
            Increases a contrast of a source image
            :param im: path to an image
            :param gamma: gamma-coefficient (values from 0.04 to 25)
            :return: prepared image as PIL image
        """
        if gamma < 0.04 or gamma > 25.0:  # do nothing with the image if gamma is out of diapason
            gamma = 1.0

        image = np.array(Image.open(im))
        resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        image = np.array(resized)

        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(image, lookUpTable)
        return Image.fromarray(res.astype(np.uint8))

    @staticmethod
    def prepare_image_for_recognition_using_thresholding(im):
        """
            Prepares an whole image for recognition using thresholds.
            Converts source to grayscale image and filters
            :param im: path to an image
            :return: prepared image as PIL image
        """
        image = np.array(Image.open(im))
        resized = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        image = np.array(resized)
        image = threshold_sauvola(image, window_size=3, k=0.05)
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(image, 0, 255, cv2.THRESH_TOZERO | cv2.THRESH_OTSU)[1]
        return Image.fromarray(image.astype(np.uint8))

    @staticmethod
    def get_boxes_from_prepared_image(im):
        """
            Provides coordinates and recognition confidence from text labels of the recognized image in original size
            :param im: string object which contains info about recognized text, its coordinates,
            confidence of recognition, etc.
            :return: a list of boxes, like ["top (int), left (int), right (int), bottom (int), confidence (str), text (str)",]
        """
        pts_data = pts.image_to_data(image=im)
        entries = pts_data.split("\n")
        raw_boxes = [entry.split("\t")[6:] for entry in entries
                     if entry.split("\t")[-1] not in ('', ' ') and len(entry.split("\t")[6:]) == 6][1:]

        for raw_box in raw_boxes:
            left, top, w, h = [int(coord) for coord in raw_box[:4]]
            left //= 2
            top //= 2
            w //= 2
            h //= 2

            right = left + w
            bottom = top + h
            raw_box[:4] = left, top, right, bottom

        return raw_boxes

    def recognize_caption_v2(self, coordinates, basename=None, image_folder=None, image=None):
        """
            Args:
                coordinates - coordinates of the image for croppping
                basename - name of the image without extension
                image_folder - folder where look for images
        """

        x1 = coordinates[0]
        y1 = coordinates[1]
        x2 = coordinates[2]
        y2 = coordinates[3]
        #print ("1-1")
        if image is None:
            try:
                image_name = self._get_image_by_basename(basename, image_folder)
                #print ("1-5")
                #print "image_name {}".format(image_name)
            except ValueError:
                print ("No image {} in image folder".format(basename))

            image = cv2.imread(image_name)
            #print ("1-2")
            if type(image) is type(None):
                return "None"

        image = image[y1:y2,x1:x2]
        #print ("x1 {}, x2 {}, y1 {}, y2 {}".format(x1, x2, y1, y2))
        new_size = (image.shape[1]*5, image.shape[0]*5)
        pil_img = Image.fromarray(image)
        resized = pil_img.resize(new_size, resample=Image.LANCZOS)
        image = np.array(resized)
        ###
        image = threshold_sauvola(image, window_size=3, k=0.01)
        #print ("1-4")
        ###
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        ###
        image = cv2.threshold(image, 0, 255,cv2.THRESH_TOZERO | cv2.THRESH_OTSU)[1]

        pil_img = Image.fromarray(image.astype(np.uint8))
        #print ("1-4")
        #cv2.imwrite(filename=save_folder+i, img=image)
        return pts.image_to_string(image=pil_img)

