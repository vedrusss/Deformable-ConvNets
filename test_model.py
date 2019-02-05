"""
TEST_MODEL.PY
Created on 1/31/2019 by A.Antonenka

Copyright (c) 2019, Arlo Technologies, Inc.
350 East Plumeria, San Jose California, 95134, U.S.A.
All rights reserved.

This software is the confidential and proprietary information of
Arlo Technologies, Inc. ("Confidential Information"). You shall not
disclose such Confidential Information and shall use it only in
accordance with the terms of the license agreement you entered into
with Arlo Technologies.
"""
"""
This is a single script to test Deformable-ConvNets model against separate images
"""

import _init_paths
import os, argparse
import pprint
import config.config as config_uitil
import numpy as np
import mxnet as mx
from   dataset.arlo_like_pascal_voc import Arlo_Classes
import cv2
from utils.image import resize, transform
from core.tester import im_detect, Predictor
from symbols.resnet_v1_101_rfcn_dcn import resnet_v1_101_rfcn_dcn
from utils.load_model import load_param
from nms.nms import gpu_nms_wrapper # py_nms_wrapper, cpu_nms_wrapper
import json

def create_predictor(ctx, config, pretrained_weights_fn, frames):
  # load symbol and weights
  sym_instance = resnet_v1_101_rfcn_dcn()  # instead must create one
  sym = sym_instance.get_symbol(config, is_train=False)

  pretrained_weights_fn_prefix, epoch = parse_weights_fn_2_paths(pretrained_weights_fn)
  arg_params, aux_params = load_param(pretrained_weights_fn_prefix, epoch, process=True)    

  data = None
  for filename in frames:
    img = cv2.imread(filename)
    if img is None: continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = prepare_data(img, config)
    break
  if data is None: return None,None,None,None # no valid data for test provided - cannot prepare the predictor
  # create predictor
  data_names  = ['data', 'im_info']
  label_names = []
  data = [mx.nd.array(data[name]) for name in data_names]
  provide_data   = [[(k, v.shape) for k, v in zip(data_names, data)]]
  provide_label  = [None]
  max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
  predictor      = Predictor(sym, data_names, label_names, context=ctx, max_data_shapes=max_data_shape,
                             provide_data=provide_data,    provide_label=provide_label,
                             arg_params=arg_params,        aux_params=aux_params)
  # warm up
  for j in xrange(2): _,_,_ = predict(data, config, predictor, data_names)

  nms = gpu_nms_wrapper(config.TEST.NMS, 0)
  # set up class names
  classes     = Arlo_Classes
  return predictor, nms, data_names, classes

def detect(predictor, config, nms, data_names, classes, img, threshold):
  data = prepare_data(img, config)
  data = [mx.nd.array(data[name]) for name in data_names]
  scores, boxes, data_dict = predict(data, config, predictor, data_names)
  boxes = boxes[0].astype('f')
  scores = scores[0].astype('f')
  # nms
  dets = []
  for j in range(1, scores.shape[1]): # j is classname index, '0' is background
    cls_scores = scores[:, j, np.newaxis]
    cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
    cls_dets = np.hstack((cls_boxes, cls_scores))
    keep = nms(cls_dets)
    cls_dets = cls_dets[keep, :]
    cls_dets = cls_dets[cls_dets[:, -1] > threshold, :]  # TODO: perclass thresholding!!
    # convert to representative form
    dets += [(classes[j-1], cls_dets[i]) for i in range(cls_dets.shape[0])]
  dets = [list(region_confidence) + [cls] for cls, region_confidence in dets] # l,t,r,b,conf,label
  return dets

def main(args):
  thresholdSet = parseThresholds(args.thresholds, ('day','night'))
  frames = [os.path.join(args.frame_folder, name) for name in os.listdir(args.frame_folder) if os.path.isfile(os.path.join(args.frame_folder, name))]
  config = config_uitil.config
  config = config_uitil.update_config(args.cfg, config)
  pprint.pprint(config)

  os.environ['PYTHONUNBUFFERED']             = '1'
  os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
  os.environ['MXNET_ENABLE_GPU_P2P']         = '0'

  ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]

  predictor, nms, data_names, classes = create_predictor(ctx, config, args.pretrained, frames)
  if predictor is None:
    print("Cannot create predictor. No valid images for test provided")
    return

  for i, frame_fn in enumerate(frames):
    img = cv2.imread(frame_fn)
    if img is None:
      print("Cannot read " + frame_fn)
      continue
    condition = 'night' if np.logical_and((img[:,:,0]==img[:,:,1]).all(),
                                          (img[:,:,1]==img[:,:,2]).all()) else 'day'
    dets = detect(predictor, config, nms, data_names, classes, img, 0.01) # list of (l,t,r,b,conf,label)
    for j, det in enumerate(dets):
      box = det[:4]
      conf, label = det[4:]
      threshold = thresholdSet[condition].get(label, 1.1) # remove detections not listed in thresholdSet
      conf = int(round(100. * conf, 0))
      if conf < threshold: continue
      print("{} ({} from {})\t {} - {} ({}); {} {} {} {}".format(frame_fn, i, len(frames),
            label, conf, threshold, box[0], box[1], box[2], box[3]))
      put_onto_image(img, box, conf, label, j)      
    if args.display:
      cv2.imshow('win', img)
      key = cv2.waitKey()
      if key == 27: break
    if args.output is not None:
      output_fn = prepare_output(args.output, frame_fn.split('/')[-1])
      cv2.imwrite(output_fn, img)
      print("Saved to " + output_fn)
  if args.display: cv2.destroyAllWindows()

def parseThresholds(thresholdsString, conditions):
  thresholdSet = {condition:{} for condition in conditions}
  thresholds = json.loads(thresholdsString)
  for condition, pairs in thresholds.items():
    for classname, value in pairs.items():
      thresholdSet[condition][classname] = value
  return thresholdSet

def parse_weights_fn_2_paths(pretrained_weights_fn):
  pretrained_weights_fn = '.'.join(pretrained_weights_fn.split('.')[:-1])
  path_els = pretrained_weights_fn.split('/')
  pretrained_weights_folder = '/'.join(path_els[:-1])
  name_els = path_els[-1].split('-')
  name = '-'.join(name_els[:-1])
  epoch = int(name_els[-1])
  pretrained_weights_fn_prefix = os.path.join(pretrained_weights_folder, name)
  return pretrained_weights_fn_prefix, epoch

def predict(data_part, config, predictor, data_names):
  #data_names = data_part.keys()
  data_batch = mx.io.DataBatch(data=[data_part], label=[], pad=0, index=0,
                               provide_data =[[(k, v.shape) for k, v in zip(data_names, data_part)]],
                               provide_label=[None])
  scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
  scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
  return scores, boxes, data_dict

def prepare_data(img, config):
  target_size   = config.SCALES[0][0]
  max_size      = config.SCALES[0][1]
  img, im_scale = resize(img, target_size, max_size, stride=config.network.IMAGE_STRIDE)
  im_tensor = transform(img, config.network.PIXEL_MEANS)
  im_info   = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
  return {'data': im_tensor, 'im_info': im_info}

def prepare_output(folder, frame_fn):
  if not os.path.exists(folder): os.makedirs(folder)
  return os.path.join(folder, frame_fn)

def put_onto_image(img, box, conf, label, j):
  def get_position(b, j):
    if   j == 0: p = (b[0], b[1]+20)
    elif j == 1: p = (b[0], b[3]-20)
    elif j == 2: p = ((b[0]+b[2])/2, b[1]+20)
    elif j == 3: p = ((b[0]+b[2])/2, b[3]-20)
    return p

  while j >= 4: j -= 4
  colors = ((255,0,0), (0,255,0), (0,0,255), (0,255,255))
  
  h, w = img.shape[:2]
  box = [int(el) for el in box]
  if box[0] < 1:  box[0] = 1
  if box[1] < 1:  box[1] = 1
  if box[2] >= w: box[2] = w - 1
  if box[3] >= h: box[3] = h - 1
  cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), colors[j], 2)
  cv2.putText(img, label+' - '+str(conf), get_position(box, j), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[j], 2)

def parse_args():
  parser = argparse.ArgumentParser(description='Evaluate RFCN_DCN model and save results to savant_test.json')
  parser.add_argument('--cfg',         type=str, required=True, help='experiment configure file name')
  parser.add_argument('--pretrained',  type=str, required=True, help='path to trained weights')
  parser.add_argument('--frame_folder',type=str, required=True, help='folder where eval frames are stored')
  parser.add_argument('--thresholds',  type=str, default='',    help='thresholds')
  parser.add_argument('--output',      default=None,            help='where to store results')
  parser.add_argument('--display',     action='store_true',     help='Display detections frame by frame')
  args, extra = parser.parse_known_args()
  return args

if __name__ == '__main__':
  main(parse_args())
