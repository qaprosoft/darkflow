import urllib
import subprocess
import sys
import os
import argparse

#print "\nOur deep learning model is processing your image ..."

os.chdir(sys.path[0])

help_string = "recognize.py"

parser = argparse.ArgumentParser(help_string)
parser.add_argument("--darkflow_home", default="/home/ubuntu/tools/darkflow", help="specify DARKFLOW_HOME location")
parser.add_argument("--model", default="general", help="specify model name")
parser.add_argument("--gpu", default="0.1", help="how much gpu (from 0.0 to 1.0)")
parser.add_argument("--load", default="-1", help="how to initialize the net? Either from .weights or a checkpoint, or even from scratch")
parser.add_argument("--folder", help="enter path to folder for recognition")
parser.add_argument("--output", help="enter type of output file", default="json", choices=["json", "xml", "img"])
parser.add_argument("--labels", default="/home/ubuntu/tools/darkflow/labels.txt", help="path to labels.txt")

args = parser.parse_args()

DARKFLOW_HOME = args.darkflow_home
MODEL = args.model
TEST_FOLDER = args.folder
OUTPUT = args.output
GPU = args.gpu
LOAD = args.load


if TEST_FOLDER == None:
    print "There is not folder for recognition! Please, use -h flag for more imformation"
    sys.exit(1)
#TODO: verify that provided folder exists preliminary!

# darflow evaluation parameters
BACKUP = DARKFLOW_HOME + "/ckpt/" + MODEL + "/"
MODEL_CONFIG = DARKFLOW_HOME + "/cfg/" + MODEL + ".cfg"
LABELS = DARKFLOW_HOME + "/labels-" + MODEL + ".txt"
print "TEST_FOLDER: " + TEST_FOLDER
print "OUTPUT: " + OUTPUT
print "LABELS: " + LABELS

FLOW_PATH = DARKFLOW_HOME + "/flow"

# One script screates json, the other  - images with bounding boxes
generate_json = FLOW_PATH + " --imgdir " + TEST_FOLDER + " --backup " + BACKUP + " --load " + LOAD + " --model " + MODEL_CONFIG + " --json --labels " + LABELS
print generate_json
generate_img = FLOW_PATH + " --imgdir " + TEST_FOLDER + " --backup " + BACKUP + " --load " + LOAD + " --model " + MODEL_CONFIG + " --labels " + LABELS
print generate_img
generate_xml = "python to_xml_output.py --json " + TEST_FOLDER + "/out/ --save " + TEST_FOLDER + "/out/"
print generate_xml

# json and img generations can be launched in background
if OUTPUT=="json":
    subprocess.call(generate_json, shell=True)
if OUTPUT=="img":
    subprocess.call(generate_img, shell=True)
if OUTPUT=="xml":
    subprocess.call(generate_json, shell=True)
    subprocess.call(generate_xml, shell=True)
