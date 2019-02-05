"""
EVALUATE_RFCN_DCN_MODEL_TO_SAVANT_JSON.PY
Created on 11/12/2018 by A.Antonenka

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
This is a single script to provide pre-trained RFCN-DCN NN finetuning using VOC formatted arlo data
"""

import _init_paths
import os, argparse
import pprint
import config.config as config_uitil
import numpy as np
import mxnet as mx
from   dataset.arlo_like_pascal_voc import Arlo_Classes
#from symbols.resnet_v1_101_rfcn_dcn import resnet_v1_101_rfcn_dcn
import cv2
#import skvideo.io
from utils.image import resize, transform
import json
from core.tester import im_detect, Predictor
#from symbols import *
from symbols.resnet_v1_101_rfcn_dcn import resnet_v1_101_rfcn_dcn
from utils.load_model import load_param
from nms.nms import gpu_nms_wrapper # py_nms_wrapper, cpu_nms_wrapper

def evaluate(ctx, config, pretrained_weights_fn, frames, threshold, output_fn, display=False):  # symbol_fn
  # load symbol and weights
  #  sym = mx.symbol.load(symbol_fn)  #  Cannot load symbol from file because of custom operators
  sym_instance = resnet_v1_101_rfcn_dcn()  # instead must create one
  sym = sym_instance.get_symbol(config, is_train=False)

  pretrained_weights_fn_prefix, epoch = parse_weights_fn_2_paths(pretrained_weights_fn)
  arg_params, aux_params = load_param(pretrained_weights_fn_prefix, epoch, process=True)    

  # set up class names
  classes     = Arlo_Classes
#  classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#             'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#             'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#             'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
#             'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
#             'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
#             'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
#             'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
  num_classes = len(classes) + 1 # include _background_

  def prepare_data(img, config):
    target_size   = config.SCALES[0][0]
    max_size      = config.SCALES[0][1]
    img, im_scale = resize(img, target_size, max_size, stride=config.network.IMAGE_STRIDE)
    im_tensor = transform(img, config.network.PIXEL_MEANS)
    im_info   = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
    return {'data': im_tensor, 'im_info': im_info}

  # load test data
#  data = []
#  for url in frames[:1]:
#    img = cv2.imread(url)
#    if img is None:
#      print("Cannot load frame " + url)
#      continue
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    target_size   = config.SCALES[0][0]
#    max_size      = config.SCALES[0][1]
#    img, im_scale = resize(img, target_size, max_size, stride=config.network.IMAGE_STRIDE)
#    im_tensor = transform(img, config.network.PIXEL_MEANS)
#    im_info   = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
#    data.append({'data': im_tensor, 'im_info': im_info})
#  if len(data) == 0:
#    print("Couldn't load any data for evaluation")
#    return

  for url in frames:
    img = cv2.imread(url)
    if img is None: continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = prepare_data(img, config)
    break

  # create predictor
  data_names  = ['data', 'im_info']
  label_names = []
  #data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
  data = [mx.nd.array(data[name]) for name in data_names]
  provide_data   = [[(k, v.shape) for k, v in zip(data_names, data)]]
  #provide_data   = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
  provide_label  = [None]
  #provide_label  = [None for i in xrange(len(data))]
  max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
  predictor      = Predictor(sym, data_names, label_names, context=ctx, max_data_shapes=max_data_shape,
                             provide_data=provide_data,    provide_label=provide_label,
                             arg_params=arg_params,        aux_params=aux_params)
  nms = gpu_nms_wrapper(config.TEST.NMS, 0)

  def predict(data_part, config, predictor, data_names):
    #data_names = data_part.keys()
    data_batch = mx.io.DataBatch(data=[data_part], label=[], pad=0, index=0,
                                 provide_data =[[(k, v.shape) for k, v in zip(data_names, data_part)]],
                                 provide_label=[None])
    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
    scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
    return scores, boxes, data_dict

  # warm up
  for j in xrange(2): _,_,_ = predict(data, config, predictor, data_names)

  # test 
  savant_json = {"1": []}
  for idx, url in enumerate(frames):
    img = cv2.imread(url)
    if img is None:
      print("Cannot load frame " + url)
      continue
    print("Processing {} ({} from {})".format(url, idx+1, len(frames)))
    data = prepare_data(img, config)
    data = [mx.nd.array(data[name]) for name in data_names]
    scores, boxes, data_dict = predict(data, config, predictor, data_names) # [idx]
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
      cls_dets = cls_dets[cls_dets[:, -1] > threshold, :]
      # convert to representative form
      dets += [(classes[j-1], cls_dets[i]) for i in range(cls_dets.shape[0])]

    # make up json result
    frame_num = int(url.split('.')[-2])        
    day = not np.logical_and((img[:,:,0]==img[:,:,1]).all(), (img[:,:,1]==img[:,:,2]).all())
    colorDepth = 1702195828 if day else 2036429415
    width, height = img.shape[:2]
    savant_json["1"].append(
      {"url": url, "responses": [
                    {"frames": [
                     {"fnum": frame_num, "flags": 2, "identifiedObjects": []}],
                      "width": width,    "height": height, "colorDepth": colorDepth}]})
    for cls, region_confidence in dets:
      xmi, ymi, xma, yma = region_confidence[:-1]
      region= ",".join(str(i) for i in [xmi/width, ymi/height, xma/width, yma/height])
      conf = float(region_confidence[-1])
      conf = int(round(100. * conf, 0))
      savant_json["1"][-1]["responses"][-1]["frames"][-1]["identifiedObjects"].append(
                          {"region": region, "classification": {"type": cls, "confidence": conf}})
      if display:
        cv2.rectangle(img, (int(xmi),int(ymi)), (int(xma),int(yma)), (0,0,255),2)
        print("{}\t {} - {}; {} {} {} {}".format(url, cls, conf, xmi, ymi, xma, yma))
    if display:
      cv2.imshow('win', img)
      key = cv2.waitKey()
      if key == 27: break
  if display: cv2.destroyAllWindows()
    
  # save savant_test.json
  output_folder = '/'.join(output_fn.split('/')[:-1])
  if output_folder != '' and not os.path.exists(output_folder): os.makedirs(output_folder)
  print("Saving savant_test results to " + output_fn)
  with open(output_fn, 'w') as f:  json.dump(post_process_json(savant_json), f, indent=4)


def parse_weights_fn_2_paths(pretrained_weights_fn):
  pretrained_weights_fn = '.'.join(pretrained_weights_fn.split('.')[:-1])
  path_els = pretrained_weights_fn.split('/')
  pretrained_weights_folder = '/'.join(path_els[:-1])
  name_els = path_els[-1].split('-')
  name = '-'.join(name_els[:-1])
  epoch = int(name_els[-1])
  pretrained_weights_fn_prefix = os.path.join(pretrained_weights_folder, name)
  return pretrained_weights_fn_prefix, epoch

def post_process_json(raw_json):
  # This is a very hacky way to get savant_test.json file in the correct format, please dont judge
  first = False
  count = 0
  for i, v in enumerate(raw_json['1']):   
    if first:
      curr_url = v['url'].split('.')[0] + '.mp4'
      if url == curr_url:
        raw_json['1'][count]['responses'][0]['frames'] += v['responses'][0]['frames']
      else:
        raw_json['1'][count]['url'] = url
        count = i
      # Last element
      if i == (len(raw_json['1'])-1) and url != curr_url:
        raw_json['1'][i]['url'] = curr_url
    url = v['url'].split('.')[0] + '.mp4'
    first=True
    
  # Collect all mp4 keys
  clean_json = {'1': []}
  for i in raw_json['1']:
    if i['url'].endswith('4'): clean_json['1'].append(i)
  return clean_json

def main(args):
  frames = [os.path.join(args.frame_folder, name) for name in open(args.input_list).read().splitlines()]
  config = config_uitil.config
  config = config_uitil.update_config(args.cfg, config)
  pprint.pprint(config)

  os.environ['PYTHONUNBUFFERED']             = '1'
  os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
  os.environ['MXNET_ENABLE_GPU_P2P']         = '0'

  ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]

  evaluate(ctx, config, args.pretrained, frames, args.threshold, args.output_fn, args.display)

  return args.output_fn

def parse_args():
  parser = argparse.ArgumentParser(description='Evaluate RFCN_DCN model and save results to savant_test.json')
  parser.add_argument('--cfg',         type=str, required=True,     help='experiment configure file name')
#  parser.add_argument('--model',       type=str, required=True,     help='path to model file (symbol.json)')
  parser.add_argument('--pretrained',  type=str, required=True,     help='path to trained weights')
  parser.add_argument('--frame_folder',type=str, required=True,     help='folder where eval frames are stored')
  parser.add_argument('--input_list',  type=str, required=True,     help='path to file which lists frame names')
  parser.add_argument('--threshold',   type=float, default=0.1,     help='common detection threshold')
  parser.add_argument('--output_fn',   default='./savant_test.json',help='where to store savant_test.json')
  parser.add_argument('--display',     action='store_true',         help='Display detections frame by frame')
  args, extra = parser.parse_known_args()
  return args

if __name__ == '__main__':
  main(parse_args())
