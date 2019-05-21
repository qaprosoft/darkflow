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


nhl_colors = {
    "regular": [
        np.array([80, 95, 105]),
        np.array([246, 246, 246])
    ],
    "alert": [
        np.array([150, 1, 10]),
        np.array([245, 25, 60])
    ],
    "white": [
        np.array([180, 180, 180]),
        np.array([255, 255, 255])
    ]
}


espn_colors = {
    "regular": [
        np.array([116, 116, 116]),
        np.array([234, 234, 234])
    ],
    "white": [
        np.array([236, 236, 236]),
        np.array([255, 255, 255])
    ],
    "alert": [
        np.array([150, 1, 10]),
        np.array([225, 25, 60])
    ]
}


def _get_mask(im, colors):
    return cv2.inRange(im, colors[0], colors[1])


def _invert_image(im):
    return cv2.bitwise_not(im)


def _concat_images(src1, src2, mask=None):
    return cv2.bitwise_and(src1, src2, mask=mask)


class OCR:

    @staticmethod
    def prepare_image_for_recognition_using_masks(im, model, resize_coef):
        """
        Applies masks by fonts above to images from nhl or espn and merges masked images for OCR.
        :param im: image as np array
        :param resize_coef: for resizing original image
        :return: prepared image as nparray
        """
        resized = cv2.resize(im, None, fx=resize_coef, fy=resize_coef, interpolation=cv2.INTER_LANCZOS4)
        image = np.array(resized)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        colors = nhl_colors if 'nhl' in model else espn_colors
        masks = [_get_mask(rgb, colors[type]) for type in colors.keys()]
        masked_images = [_concat_images(image, image, mask=mask) for mask in masks]
        inverted_images = [_invert_image(masked_image) for masked_image in masked_images]

        result_im = inverted_images[0]
        for i in range(1, len(inverted_images)):
            result_im = _concat_images(result_im, inverted_images[i])
        return Image.fromarray(result_im.astype(np.uint8))

    @staticmethod
    def prepare_image_for_recognition_using_gammas(im, gamma, resize_coef):
        """
        Prepares an whole image for recognition using gamma-correction method.
        Increases a contrast of a source image
        :param im: image as np array
        :param gamma: gamma-coefficient (values from 0.04 to 25)
        :param resize_coef: for resizing original image
        :return: prepared image as PIL image
        """
        if gamma < 0.04 or gamma > 25.0:  # do nothing with the image if gamma is out of diapason
            gamma = 1.0
        resized = cv2.resize(im, None, fx=resize_coef, fy=resize_coef, interpolation=cv2.INTER_LANCZOS4)
        image = np.array(resized)

        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        res = cv2.LUT(image, lookUpTable)
        return Image.fromarray(res.astype(np.uint8))

    @staticmethod
    def prepare_image_for_recognition_using_thresholding(im, resize_coef):
        """
        Prepares an whole image for recognition using thresholds.
        Converts source to grayscale image and filters
        :param im: image as np array
        :param resize_coef: for resizing original image
        :return: prepared image as PIL image
        """
        resized = cv2.resize(im, None, fx=resize_coef, fy=resize_coef, interpolation=cv2.INTER_LINEAR)
        image = np.array(resized)
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(cv2.GaussianBlur(image, (9, 9), 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return Image.fromarray(image.astype(np.uint8))

    @staticmethod
    def get_boxes_from_prepared_image(im, resize_coef):
        """
        Provides coordinates and recognition confidence from text labels of the recognized image in original size
        :param im: string object which contains info about recognized text, its coordinates,
        confidence of recognition, etc.
        :param resize_coef: for resizing into original
        :return: a list of boxes, like ["top (int), left (int), right (int), bottom (int), confidence (str), text (str)",]
        """
        pts_data = pts.image_to_data(image=im, config="-l eng")
        entries = pts_data.split("\n")
        raw_boxes = [entry.split("\t")[6:] for entry in entries
                     if entry.split("\t")[-1] not in ('', ' ') and len(entry.split("\t")[6:]) == 6][1:]

        for raw_box in raw_boxes:
            left, top, w, h = [int(coord) for coord in raw_box[:4]]
            left //= resize_coef
            top //= resize_coef
            w //= resize_coef
            h //= resize_coef

            right = left + w
            bottom = top + h
            raw_box[:4] = left, top, right, bottom
        return raw_boxes

