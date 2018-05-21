# -*- coding:utf-8 -*-
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import json
import copy

classes = ["vehicle"]

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
        self.dictionary = {"images": [], "type": "instances", "annotations": [], "categories": []}

    def input_data(self, dict, annid):
        self.filename = str(dict["images"]["imgid"]) + ".jpg"
        self.height = dict["images"]["height"]
        self.width = dict["images"]["width"]
        self.imgid = dict["images"]["imgid"]
        self.image = {"file_name": self.filename, "height": self.height, "width": self.width, "id": self.imgid}

        for dic in dict["annotations"]:
            annid += 1
            self.segmentation = dic["segmentation"]
            self.area = dic["area"]
            self.iscrowd = dic["iscrowd"]
            self.bbox = dic["bbox"]
            self.catid = dic["category_id"]
            self.annid = annid
            self.annotation = {"segmentation": self.segmentation, "area": self.area, "iscrowd": self.iscrowd, "image_id":
            self.imgid, "bbox": self.bbox, "category_id": self.catid, "id": self.annid}
        self.make_dict()
        return annid

    def make_dict(self):
        self.dictionary["images"].append(self.image)
        self.dictionary["annotations"].append(self.annotation)

    def add_category(self):
        for cls in classes:
            cls_id = classes.index(cls) + 1
            self.category = {"supercategory": cls, "id": cls_id, "name": cls}
            self.dictionary["categories"].append(self.category)

def convert_annotation(image_id):
    segmentations = []
    classall = []
    in_file = open('xml/%s.xml' % (image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    for obj in root.iter('object'):
        i = 0
        var = 1
        segmentation = []
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls) + 1
        classall.append(cls_id)
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
        segmentations.append(segmentation)
    return width, height, segmentations, classall


def get_bbox(segmentations):
    boxes = []
    for i in range(len(segmentations)):
        segmentation = copy.deepcopy(segmentations[i])
        x = segmentation[::2]
        y = segmentation[1::2]
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        width = xmax - xmin
        height = ymax - ymin
        box = [xmin, ymin, width, height]
        boxes.append(box)
    return boxes


def get_area(segmentations):
    areas = []
    for i in range(len(segmentations)):
        segmentation = copy.deepcopy(segmentations[i])
        xarray = segmentation[::2]
        yarray = segmentation[1::2]
        listall = []
        for i in range(len(xarray)):
            ll = [[xarray[i], yarray[i]]]
            listall.append(ll)
        array_value = np.array(listall)
        contours = []
        contours.append(array_value)
        area = cv2.contourArea(contours[0])
        areas.append(area)
    return areas


def save_json(height, width, imgid, segmentations, areas, iscrowd, bboxes, catids, annid, image_id):
    dictionary = {"annotations": []}
    image = {"height": height, "width": width, "imgid": imgid}
    dictionary["images"] = image
    for i in range(len(segmentations)):
        annotation = {"segmentation": [segmentations[i]], "area": areas[i], "iscrowd": iscrowd, "image_id": imgid,
                      "bbox": bboxes[i], "category_id": catids[i], "id": annid}
        dictionary["annotations"].append(annotation)

    json_file = open('json/%s.json' % (image_id), 'w')
    json.dump(dictionary, json_file)
    json_file.close()


def read_json(json_name):
    read_file = open(json_name, 'r')
    dictionary = json.load(read_file)
    return dictionary


image_ids = open('ls.txt').read().strip().split()
for image_id in image_ids:
    width, height, segmentations, classall = convert_annotation(image_id)
    bboxes = get_bbox(segmentations)
    areas = get_area(segmentations)
    save_json(height, width, image_id, segmentations, areas, 0, bboxes, classall, image_id, image_id)


voc = ANNO()
voc.init_index()
voc.add_category()
annid = 1
for image_id in image_ids:
    diction = read_json('json/%s.json' % (image_id))
    annid = voc.input_data(diction, annid)
    print annid
json_file = open('results.json', 'w')
json.dump(voc.dictionary, json_file)
json_file.close()
