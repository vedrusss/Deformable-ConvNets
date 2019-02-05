"""
ARLO_LIKE_PASCAL_VOC.PY
Created on 11/13/2018 by A.Antonenka

Copyright (c) 2018, Arlo Technologies, Inc.
350 East Plumeria, San Jose California, 95134, U.S.A.
All rights reserved.

This software is the confidential and proprietary information of
Arlo Technologies, Inc. ("Confidential Information"). You shall not
disclose such Confidential Information and shall use it only in
accordance with the terms of the license agreement you entered into
with Arlo Technologies.
"""
"""
Arlo modification of original pascal_voc.py provided by Deformable-ConvNets.
Arlo like Pascal VOC database
This class loads ground truth notations from standard Pascal VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.

The difference is based on other classnames and their amount as well as some offset issues.
"""

"""
NOTICE: Arlo dataset must be organized the same way as PascalVOC dataset:
Arlo_Like_VOC
|    |-Annotations
|    |-Images
|    |-ImageSets
|          |-Main
|              |-train_list_A.txt  (this filename (with extension) you must provide as 'image_set' parameter)
|              |-train_list_B.txt
|              |- ...
|              |-test_list_A.txt
|              |-test_list_B.txt
|              |- ...
"""

import cPickle
import cv2
import os
import numpy as np
import PIL
import xml.etree.ElementTree as ET

from imdb import IMDB
from pascal_voc_eval import voc_eval, voc_eval_sds
from ds_utils import unique_boxes, filter_small_boxes

Arlo_Classes = ['animal', 'bicycle', 'bird', 'bus',      'car',
                'cat',    'dog',     'horse','motorbike','person']

#Arlo_Classes = ['person', 'bicycle', 'bird', 'bus',      'car',
#                'cat',    'dog',     'horse','motorbike','animal']

class Arlo_Like_PascalVOC(IMDB):
    def __init__(self, image_set, root_path, data_path, result_path=None, mask_size=-1, binary_thresh=None):
        """
        fill basic information to initialize imdb. Params:
         image_set: filename (with extension) which lists imagenames (with extensions) used for training
         root_path: root folder where dataset (Arlo_Like_VOC) is kept. Here 'cache' dir may be created as well
         data_path: this is Arlo_Like_VOC folder path. Where Annotations, Images and ImageSets are stored
        returns: imdb object
        """
        global Arlo_Classes
        self.root_path = root_path # root folder where 'data_path' subfolder is kept
        self.data_path = os.path.join(self.root_path, 'Arlo_Like_PascalVOC') # folder where 'Annotations', 'JPEGImages' and 'ImageSets' are stored
        
        super(Arlo_Like_PascalVOC, self).\
          __init__('arlo_'+self._remExt(image_set), image_set, self.root_path, self.data_path, result_path)

        
        self.classes = ['__background__'] + Arlo_Classes  # always index 0
        self.num_classes = len(self.classes)
        
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print('num_images provided for training:', self.num_images)
        
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh
        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        loads the list of image filenames (without path) provided within image_set file
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set)
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index (image filename), find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'Images', index)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: [[roi_rec for image 1],[roi_rec for image 2],...,[roi_rec for image N]]
        if exists loads cache, otherwise creates one
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid: roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self.load_pascal_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid: cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def load_pascal_annotation(self, index):
        """
        for a given index (image filename), loads image and bounding boxes info from XML file
        :param index: index of a specific image (image filename)
        :return: record['image (path)', 'height', 'width', 'boxes', 
                        'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'flipped']
        """
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        xmlfilename = os.path.join(self.data_path, 'Annotations', self._remExt(index) + '.xml')
        tree = ET.parse(xmlfilename)
        size = tree.find('size')
        roi_rec['height'] = float(size.find('height').text)
        roi_rec['width']  = float(size.find('width').text)
        objs = tree.findall('object')
        notDifficult = lambda obj: (obj.find('difficult') is not None and int(obj.find('difficult').text) == 0)
        if not self.config['use_diff']: objs = [obj for obj in objs if notDifficult(obj)] # use only not difficult objects
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            cls = class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec

    @staticmethod
    def _remExt(filename):
        return '.'.join(filename.split('.')[:-1])
