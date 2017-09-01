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
    "default":[129650, 0.9, "tiny-ai-weights-lc-mlb"]
}

parser = argparse.ArgumentParser(help_string)
parser.add_argument("--model", choices=models.keys(), default="default", help="choose model among {}".format(models.keys()))
parser.add_argument("--url", help="enter url for testing")
parser.add_argument("--verbose", help="print model output in the console", choices=["True", "False"])

args = parser.parse_args()
model = args.model
URL = args.url

if URL == None:
    print "Please, use -h flag for more imformation"
    sys.exit(1)

# darflow evaluation parameters
gpu = models[model][1]
load = models[model][0]
model_name = models[model][2]
download_img = "/home/ubuntu/darkflow/downloaded_img/" # where to download image from url
model_path = "/home/ubuntu/darkflow/flow"


IMAGE = URL.rsplit('/',1)[1]
#print IMAGE
#print download_img+IMAGE
img = urllib.urlretrieve(URL, download_img+IMAGE)
img = URL

# One script screates json, the other  - images with bounding boxes
sript_to_call = "{} --test {} --load {} --model /home/ubuntu/darkflow/cfg/{}.cfg --json".format(model_path, download_img, load, model_name)
print sript_to_call
sript_to_call_2 = "{} --test {} --load {} --model /home/ubuntu/darkflow/cfg/{}.cfg".format(model_path, download_img, load, model_name)
print sript_to_call_2
script_to_call_3 = "python to_xml_output.py --json /home/ubuntu/darkflow/downloaded_img/out/ --save /home/ubuntu/darkflow/downloaded_img/out/"
print script_to_call_3

#web-server parameters
port = 8888
web_address_raw = "http://54.245.105.12:{}/downloaded_img/".format(port)
web_address = "http://54.245.105.12:{}/downloaded_img/out/".format(port)

#os.chdir(sys.path[0])

start_server = "python3 -m http.server {}".format(port)

verbose = args.verbose
if verbose=="True":
    subprocess.Popen(sript_to_call, shell=True)
    subprocess.Popen(sript_to_call_2, shell=True)
    subprocess.Popen(start_server, shell=True)
    subprocess.Popen(script_to_call_3, shell=True)
else:
    #calling evaluation
    subprocess.Popen(sript_to_call, shell=True, stdout=open(os.devnull, "wb"), stderr=subprocess.STDOUT)
    subprocess.Popen(sript_to_call_2, shell=True, stdout=open(os.devnull, "wb"), stderr=subprocess.STDOUT)
    #starting server
    subprocess.Popen(start_server, shell=True, stdout=open(os.devnull, "wb"), stderr=subprocess.STDOUT)
    subprocess.Popen(script_to_call_3, shell=True)

filename, file_extension = os.path.splitext(download_img+IMAGE)
#filename, file_extension = os.path.splitext(img)

print "{"
print "input_image:\"{}\",".format(web_address_raw+IMAGE)
print "output_image:\"{}\",".format(web_address+IMAGE)
f_ext_len = len(file_extension)
shift = f_ext_len - 1
print "output_metadata:\"{}\"".format(web_address+IMAGE[:-shift]+"xml")
print "}"
