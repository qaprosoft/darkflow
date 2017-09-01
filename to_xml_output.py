import json
from lxml import etree
import os
import codecs
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json", help="path to predicted json")
parser.add_argument("--save", help="where to save generated xml")

args = parser.parse_args()

json_folder = args.json

for file in os.listdir(json_folder):
    #defence from the macos system files
    if file.startswith("."):
        continue

    if file[-4:] == "json":
        annotation = etree.Element("annotation")

        #filename tag creating
        filename = etree.Element("filename")
        name = file[:-5]
        filename.text = name
        annotation.append(filename)

        try:
            config = json.loads(codecs.open(json_folder+file, "r", encoding='utf-8', errors='ignore').read())

            #isEmpty
            empty = etree.Element("empty")
            empty.text = "False"
            annotation.append(empty)

            for dictt in config:
                x1 = dictt["topleft"]["x"]
                y1 = dictt["topleft"]["y"]
                x2 = dictt["bottomright"]["x"]
                y2 = dictt["bottomright"]["y"]

                #object tag creating
                objectt = etree.Element("object")
                annotation.append(objectt)

                #name
                name = etree.Element("name")
                name.text = dictt["label"]
                objectt.append(name)

                #confidence
                confidence = etree.Element("confidence")
                confidence.text = str(dictt["confidence"]).decode("utf-8")
                objectt.append(confidence)

                #BNDBOX tag in OBJECT tag
                bndbox = etree.Element("bndbox")
                objectt.append(bndbox)

                #xmin
                xmin = etree.Element("xmin")
                xmin.text = str(x1).decode("utf-8")
                bndbox.append(xmin)

                #ymin
                ymin = etree.Element("ymin")
                ymin.text = str(y1).decode("utf-8")
                bndbox.append(ymin)

                #xmax
                xmax = etree.Element("xmax")
                xmax.text = str(x2).decode("utf-8")
                bndbox.append(xmax)

                #ymax
                ymax = etree.Element("ymax")
                ymax.text = str(y2).decode("utf-8")
                bndbox.append(ymax)
        except ValueError:
            x1 = 0
            x2 = 0
            y1 = 0
            y2 = 0

            #isEmpty
            empty = etree.Element("empty")
            empty.text = "True"
            annotation.append(empty)

            #object tag creating
            objectt = etree.Element("object")
            annotation.append(objectt)

            #name
            name = etree.Element("name")
            name.text = "None"
            objectt.append(name)

            #confidence
            confidence = etree.Element("confidence")
            confidence.text = "0"
            objectt.append(confidence)

            #BNDBOX tag in OBJECT tag
            bndbox = etree.Element("bndbox")
            objectt.append(bndbox)

            #xmin
            xmin = etree.Element("xmin")
            xmin.text = str(x1).decode("utf-8")
            bndbox.append(xmin)

            #ymin
            ymin = etree.Element("ymin")
            ymin.text = str(y1).decode("utf-8")
            bndbox.append(ymin)

            #xmax
            xmax = etree.Element("xmax")
            xmax.text = str(x2).decode("utf-8")
            bndbox.append(xmax)

            #ymax
            ymax = etree.Element("ymax")
            ymax.text = str(y2).decode("utf-8")
            bndbox.append(ymax)
        tree = etree.ElementTree(annotation)
        xml_name = file[0:-4] + "xml"
        save = args.save
        print save+xml_name
        tree.write(save+xml_name, pretty_print=True, xml_declaration=False, encoding="utf-8")
