import urllib
import subprocess
import sys
import os
import argparse

#print "\nOur deep learning model is processing your image ..."

help_string = "recognizeControls.py"

# values of the dict -> [load, gpu, cfg]
models = {
    "mlb":[7600, 0.9, "tiny-ai-weights"],
    "default":[97000, 0.9, "tiny-ai-weights-lc-mlb"]
}

root = "/home/ubuntu/darkflow/"

parser = argparse.ArgumentParser(help_string)
parser.add_argument("--model", choices=models.keys(), default="default", help="choose model among {}".format(models.keys()))
parser.add_argument("--url", default="http://54.245.105.12:8888/downloaded_img/-1027146364.jpg", help="enter url for testing")
parser.add_argument("--input", default = "downloaded_img/", help="path to folder for saving downloaded image or for looking for images to test")
parser.add_argument("--verbose", default=False, help="print model output in the console", choices=["True", "False"])
parser.add_argument("--outputType", default="xml", choices=["xml", "json"], help="type of the output")
parser.add_argument("--save", default="downloaded_img/out/", help="folder for saving recognized images and json/xml")

args = parser.parse_args()
model = args.model
URL = args.url
outputType = args.outputType
save = args.save
verbose = args.verbose

# darkflow evaluation parameters
gpu = models[model][1]
load = models[model][0]
model_name = models[model][2]
download_img = args.input #"/home/ubuntu/darkflow/downloaded_img/" # where to download image from url
model_path = "/home/ubuntu/darkflow/flow"


if URL != None:
    IMAGE = URL.rsplit('/',1)[1]
    #print download_img + IMAGE
    img = urllib.urlretrieve(URL, root+download_img+IMAGE)

# One script screates json, the other  - images with bounding boxes
flow_json = "{} --test {} --load {} --model /home/ubuntu/darkflow/cfg/{}.cfg --json".format(model_path, download_img, load, model_name)
flow_images = "{} --test {} --load {} --model /home/ubuntu/darkflow/cfg/{}.cfg".format(model_path, download_img, load, model_name)
xml_output = "python to_xml_output.py --json /home/ubuntu/darkflow/{} --save /home/ubuntu/darkflow/{}".format(download_img+"out/", save)

#web-server parameters
port = 8888
web_address_raw = "http://54.245.105.12:{}{}".format(port, download_img) # where test images is saving
if save != download_img+"/out":
    web_address = "http://54.245.105.12:{}{}".format(port, save) # where output images is saving
else:
    web_address = "http://54.245.105.12:{}{}".format(port, download_img) # where output images is saving

#os.chdir(sys.path[0])
#os.chdir("/home/")

start_server = "python3 -m http.server {}".format(port)

if verbose=="True":
    subprocess.Popen(flow_json, shell=True)
    subprocess.Popen(flow_images, shell=True)
    #subprocess.Popen(start_server, shell=True, cwd="/")
    if outputType == "xml":
        subprocess.Popen(xml_output, shell=True)
else:
    #calling evaluation
    subprocess.Popen(flow_json, shell=True, stdout=open(os.devnull, "wb"), stderr=subprocess.STDOUT)
    subprocess.Popen(flow_images, shell=True, stdout=open(os.devnull, "wb"), stderr=subprocess.STDOUT)
    #starting server
    #subprocess.Popen(start_server, shell=True, stdout=open(os.devnull, "wb"), stderr=subprocess.STDOUT, cwd="/")
    if outputType == "xml":
        subprocess.Popen(xml_output, shell=True, stdout=open(os.devnull, "wb"), stderr=subprocess.STDOUT)

filename, file_extension = os.path.splitext(download_img+IMAGE)

f_ext_len = len(file_extension)
shift = f_ext_len - 1

if save != download_img+"out/":
    print ("1 " + root+download_img+"out/"+IMAGE)
    print ("1 " + root+save+IMAGE)
    print ("2 " + root+download_img+"out/"+IMAGE[:-shift]+"xml")
    print ("2 " + root+save+IMAGE[:-shift]+"xml")
    print ("3 " + root+download_img+"out/"+IMAGE[:-shift]+"json")
    print ("3 " + root+save+IMAGE[:-shift]+"json")
    os.rename(root+download_img+"out/"+IMAGE, root+save+IMAGE)
    os.rename(root+download_img+"out/"+IMAGE[:-shift]+"xml", root+save+IMAGE[:-shift]+"xml")
    os.rename(root+download_img+"out/"+IMAGE[:-shift]+"json", root+save+IMAGE[:-shift]+"json")
print "{"
print "input_image:\"{}\",".format(web_address_raw+IMAGE)
print "output_image:\"{}\",".format(web_address+IMAGE)
if outputType == "xml":
    print "output_metadata:\"{}\"".format(web_address+IMAGE[:-shift]+"xml")
else:
    print "output_metadata:\"{}\"".format(web_address+IMAGE[:-shift]+"json")
print "}"
