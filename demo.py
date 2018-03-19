# -*- coding:utf-8 -*-
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import json


class ANNO:
    def __init__(self):
        self.height = ""
        self.width = ""
        self.imgid = ""
        self.segmentation = ""
        self.area = ""
        self.iscrowd = ""
        self.bbox = ""
        self.catid = ""
        self.annid = ""
        self.dictionary = {}
        self.image = {}
        self.annotation = {}
        self.category = {}

    def init_index(self):
        self.dictionary = {}
        self.dictionary["images"] = []
        self.dictionary["annotations"] = []
        self.dictionary["categories"] = []

    def input_data(self, dict):
        self.filename = dict["imgid"] + ".jpg"
        self.height = dict["height"]
        self.width = dict["width"]
        self.imgid = dict["imgid"]
        self.segmentation = dict["segmentation"]
        self.area = dict["area"]
        self.iscrowd = dict["iscrowd"]
        self.bbox = dict["bbox"]
        self.catid = dict["catid"]
        self.annid = dict["annid"]
        self.image = {"file_name": self.filename, "height": self.height, "width": self.width, "id": self.imgid}
        self.annotation = {"segmentation": self.segmentation, "area": self.area, "iscrowd": self.iscrowd, "image_id":
            self.imgid, "bbox": self.bbox, "category_id": self.catid, "id": self.annid}
        self.make_dict()

    def make_dict(self):
        self.dictionary["images"].append(self.image)
        self.dictionary["annotations"].append(self.annotation)

    def add_category(self):
        self.category = {"supercategory": "vehicle", "id": 1, "name": "vehicle"}
        self.dictionary["categories"].append(self.category)


def convert_annotation(image_id):
    in_file = open('xml/%s.xml' % (image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    var = 1
    i = 1
    segmentation = []

    for obj in root.iter('object'):
        polygon = obj.find('polygon')
        while var == 1:
            polygon_index = "point" + str(i)
            i = i + 1
            if polygon.find(polygon_index) is None:
                var = 0
            else:
                coordinates = polygon.find(polygon_index).text
                co_list = coordinates.split(',')
                for index in co_list:
                    x = int(index)
                    segmentation.append(x)
    return width, height, segmentation


def get_bbox(segmentation):
    x = segmentation[::2]
    y = segmentation[1::2]
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    width = xmax - xmin
    height = ymax - ymin
    return [xmin, ymin, width, height]


def get_area(segmentation):
    xarray = segmentation[::2]
    yarray = segmentation[1::2]
    length = len(xarray)
    listall = []
    for i in range(0, length):
        ll = [[xarray[i], yarray[i]]]
        listall.append(ll)
    array_value = np.array(listall)
    contours = []
    contours.append(array_value)
    area = cv2.contourArea(contours[0])
    return area


def save_json(height, width, imgid, segmentation, area, iscrowd, bbox, catid, annid, image_id):
    annotation = {}
    annotation["height"] = height
    annotation["width"] = width
    annotation["imgid"] = imgid
    annotation["segmentation"] = [segmentation]
    annotation["area"] = area
    annotation["iscrowd"] = iscrowd
    annotation["bbox"] = bbox
    annotation["catid"] = catid
    annotation["annid"] = annid

    json_file = open('json/%s.json' % (image_id), 'w')
    json.dump(annotation, json_file)
    json_file.close()


def read_json(json_name):
    read_file = open(json_name, 'r')
    dictionary = json.load(read_file)
    return dictionary


image_ids = open('ls.txt').read().strip().split()
catid = 1
for image_id in image_ids:
    w, h, segment = convert_annotation(image_id)
    bbox = get_bbox(segment)
    area = get_area(segment)
    save_json(h, w, image_id, segment, area, 1, bbox, catid, image_id, image_id)


voc = ANNO()

voc.init_index()
voc.add_category()

for image_id in image_ids:
    diction = read_json('json/%s.json' % (image_id))
    voc.input_data(diction)

json_file = open('result.json', 'w')
json.dump(voc.dictionary, json_file)
json_file.close()
