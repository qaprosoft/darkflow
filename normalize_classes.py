import cv2

#just for example
eA = {
"name" : "fundingCode",
"style" : "text-align:center",
"id" : "fundingCode",
"type" : "text",
"class" : "form-control"
}

xml_template ="""
<object>
    <name></name>
    <pose>Unspecified</pose>
    <difficult>0</difficult>
    <bndbox>
        <xmin>{}</xmin>
        <ymin>{}</ymin>
        <xmax>{}</xmax>
        <ymax>{}</ymax>
    </bndbox>
</object>
"""


def clean_json(dictt):
    """
    Fuction takes elementsAttributes dict from the json and try to find out what
    is class does it represent

    dictt - elementsAttributes from default json
    """
    #             0          1            2             3?           4        5?         6
    classes = ["button", "checkbox", "text_field", "date_picker", "radio", "select", "undef"]
    if is_empy(dictt):
        return classes[6]

    key_words = {
        "btn": classes[0],
        "text": classes[2],
        "radio": classes[4],
        "checkbox": classes[1],
        "sign in": classes[0],
        "form-control": classes[2],
        "submit": classes[0],
        "button": classes[0]
    }

    keys = dictt.keys()
    patterns = key_words.keys()

    i = 0
    for name in keys:
        for pattern in patterns:
            if str(dictt[name]).strip().lower().find(pattern) == -1:
                i = 0
            else:
                i = 1
                return key_words[pattern]
    if i == 0:
        return classes[6]

def logging(img_name, coord, rect_id):
    """
        img_name - short image name
        xml_string - xml pattern for inserting in the logging file
        coord - tuple with undef rect coordinates
        rect_id - id of current rect
    """

    x1 = coord[0]
    y1 = coord[1]
    x2 = coord[2]
    y2 = coord[3]

    log = "Image: {}\nrect_id: {}\nXML template:{}".format(img_name, rect_id, xml_template.format(x1, y1, x2, y2))
    line = "___________________________________________________________________________________________________\n"
    with open(img_name+".txt", "w+") as f:
        f.write(log)
        f.write(line)
        f.close()

def is_empy(dictt):
    return not bool(dictt)

#print clean_json(eA)
