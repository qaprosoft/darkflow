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

    def __init__(self):
        """
            OCR constructor.
        """
        pass

    def _get_image_by_basename(self, basename, image_folder):
        for file in os.listdir(image_folder):
            #print ("1-6")
            file_base = os.path.basename(file).rsplit(".", 1)[0]
            if file_base == basename:
                return file
            else:
                continue
        raise ValueError

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
