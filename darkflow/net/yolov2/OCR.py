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
        resized = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
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
        resized = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        image = np.array(resized)
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(cv2.GaussianBlur(image, (9, 9), 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
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

