"""
FINETUNE_RFCN_DCN_END2END.PY
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
import time
import config.config as config_uitil
import logging
import numpy as np
import mxnet as mx

from symbols.resnet_v1_101_rfcn_dcn import resnet_v1_101_rfcn_dcn
from core                  import callback, metric
from core.loader           import AnchorLoader
from core.module           import MutableModule
from utils.load_data       import load_gt_roidb, merge_roidb, filter_roidb
from utils.load_model      import load_param
from utils.PrefetchingIter import PrefetchingIter
from utils.lr_scheduler    import WarmupMultiFactorScheduler

def create_logger(output_path, logging_level):
  if not os.path.exists(output_path): os.makedirs(output_path)
  log_file = 'finetuning_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
  logging.basicConfig(filename=os.path.join(output_path, log_file), format='%(asctime)-15s %(message)s')
  logger = logging.getLogger()
  logger.setLevel(logging_level)
  return logger

def parse_weights_fn_2_paths(pretrained_weights_fn, output_path):
  pretrained_weights_fn = '.'.join(pretrained_weights_fn.split('.')[:-1])
  path_els = pretrained_weights_fn.split('/')
  pretrained_weights_folder = '/'.join(path_els[:-1])
  name_els = path_els[-1].split('-')
  name = '-'.join(name_els[:-1])
  epoch = int(name_els[-1])
  result_weights_fn_prefix = os.path.join(output_path, name)
  pretrained_weights_fn_prefix = os.path.join(pretrained_weights_folder, name)
  return pretrained_weights_fn_prefix, epoch, result_weights_fn_prefix

def train_net(config, ctx, pretrained_weights_fn, epochs, lr, logger):
  pretrained_weights_fn_prefix, begin_epoch, result_weights_fn_prefix = \
    parse_weights_fn_2_paths(pretrained_weights_fn, config.output_path)
  end_epoch = begin_epoch + epochs
  
  # load symbol
  sym_instance = resnet_v1_101_rfcn_dcn()
  #  shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)
  #  sym_instance = eval(config.symbol + '.' + config.symbol)()
  sym = sym_instance.get_symbol(config, is_train=True)
  feat_sym = sym.get_internals()['rpn_cls_score_output']
#  sym_test = sym_instance.get_symbol(config, is_train=False)
  sym.save(os.path.join(config.output_path, 'symbol.json'))
  # setup multi-gpu
  batch_size = len(ctx)
  input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

  # load datasets and prepare imdb for training
  datasets   = config.dataset.dataset.split('+')
  image_sets = config.dataset.image_set.split('+')
  roidbs = [load_gt_roidb(datasets[i], image_set, config.dataset.root_path, 
                          config.dataset.dataset_path,
                          flip=config.TRAIN.FLIP, rotate=config.TRAIN.ROTATE, 
                          max_rotation_deg=config.TRAIN.MAX_ROTATION_DEG, 
                          rotation_step=config.TRAIN.ROTATION_STEP) 
            for i, image_set in enumerate(image_sets)]
  roidb = merge_roidb(roidbs)
  roidb = filter_roidb(roidb, config)

  # load training data
  train_data = AnchorLoader(feat_sym, roidb, config, 
        batch_size=input_batch_size, shuffle=config.TRAIN.SHUFFLE, ctx=ctx,
        feat_stride=config.network.RPN_FEAT_STRIDE, anchor_scales=config.network.ANCHOR_SCALES,
        anchor_ratios=config.network.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING)

  # infer max shape
  max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, 
                              max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
  max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
  max_data_shape.append(('gt_boxes', (config.TRAIN.BATCH_IMAGES, 100, 5)))
  print('providing maximum shape', max_data_shape, max_label_shape)

  data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
  pprint.pprint(data_shape_dict)
  sym_instance.infer_shape(data_shape_dict)

  # load and initialize params
  if config.TRAIN.RESUME:
    print('continue training from ', begin_epoch)
    arg_params, aux_params = load_param(pretrained_weights_fn_prefix, begin_epoch, convert=True)
  else:
    arg_params, aux_params = load_param(pretrained_weights_fn_prefix, begin_epoch, convert=True)
    sym_instance.init_weight(config, arg_params, aux_params)

  # check parameter shapes
  sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

  # create solver
  fixed_param_prefix = config.network.FIXED_PARAMS
  data_names = [k[0] for k in train_data.provide_data_single]
  label_names = [k[0] for k in train_data.provide_label_single]

  mod = MutableModule(sym, data_names=data_names, label_names=label_names,
            logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in range(batch_size)],
            max_label_shapes=[max_label_shape for _ in range(batch_size)], 
            fixed_param_prefix=fixed_param_prefix)

#  if config.TRAIN.RESUME:
#    mod._preload_opt_states = '%s-%04d.states'%(prefix, begin_epoch)

  # decide training params
  # metric
  rpn_eval_metric = metric.RPNAccMetric()
  rpn_cls_metric = metric.RPNLogLossMetric()
  rpn_bbox_metric = metric.RPNL1LossMetric()
  eval_metric = metric.RCNNAccMetric(config)
  cls_metric = metric.RCNNLogLossMetric(config)
  bbox_metric = metric.RCNNL1LossMetric(config)
  eval_metrics = mx.metric.CompositeEvalMetric()
  # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
  for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, 
                       eval_metric, cls_metric, bbox_metric]:   eval_metrics.add(child_metric)
  # callback
  batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=config.default.frequent)
  means = np.tile(np.array(config.TRAIN.BBOX_MEANS), 2 
                           if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
  stds = np.tile(np.array(config.TRAIN.BBOX_STDS), 2 
                           if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
  epoch_end_callback = [mx.callback.module_checkpoint(mod, result_weights_fn_prefix, 
                                                      period=1, save_optimizer_states=False),
                        callback.do_checkpoint(result_weights_fn_prefix, means, stds)]
  # decide learning rate
  base_lr = lr
  lr_factor = config.TRAIN.lr_factor
  ##lr_epoch_diff = [float(config.TRAIN.lr_step.split(',')[0])]
  ##lr = base_lr # * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
  ##lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]

  # TODO: lr_epoch values must be always greater than begin_epoch
  lr_epoch = [float(epoch) for epoch in config.TRAIN.lr_step.split(',')]
  lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
  lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
  lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]


  print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
  lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, 
                                            config.TRAIN.warmup_lr, config.TRAIN.warmup_step)
  # optimizer
  optimizer_params = {'momentum': config.TRAIN.momentum,
                      'wd': config.TRAIN.wd,
                      'learning_rate': lr,
                      'lr_scheduler': lr_scheduler,
                      'rescale_grad': 1.0,
                      'clip_gradient': None}

  if not isinstance(train_data, PrefetchingIter):  train_data = PrefetchingIter(train_data)

  # train
  mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
          batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
          optimizer='sgd', optimizer_params=optimizer_params,
          arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)

  return end_epoch

def parse_args():
  parser = argparse.ArgumentParser(description='Finetune R-FCN-DCN network')
  parser.add_argument('--cfg', type=str, required=True, help='experiment configure file name')
  parser.add_argument('--gpus',type=str, default='0', help='Specify gpu devices for training, split by comma')
  parser.add_argument('--pretrained', type=str, required=True, help='path to pretrained weights')
  parser.add_argument('--epochs', type=int, required=True, help='Amount epochs to finetune')
  parser.add_argument('--dataset_root', type=str, required=True, help='root_path to datasets for training')
  parser.add_argument('--dataset', type=str, required=True, help='datasets used for training')
  parser.add_argument('--image_set', type=str, default='fp_20_train.txt', help='images sets splitted by + sign. Amount must be equal to specified datasets')
  parser.add_argument('--output_path', type=str, default=None, help='folder to store resulting model to')
  parser.add_argument('--frequent', type=int, default=None, help='frequency of logging')
  args, extra = parser.parse_known_args()
  return args

def main(args):
  config = config_uitil.config
  config = config_uitil.update_config(args.cfg, config)
  config.gpus                   = args.gpus
  config.dataset.dataset        = args.dataset
  config.dataset.root_path      = args.dataset_root
  config.dataset.image_set      = args.image_set
  config.dataset.test_image_set = args.image_set
  if args.frequent    is not None: config.default.frequent = args.frequent
  if args.output_path is not None: config.output_path      = args.output_path
  pprint.pprint(config)
  logger = create_logger(output_path=config.output_path, logging_level=logging.INFO)
  logger.info('training config:{}\n'.format(pprint.pformat(config)))

  os.environ['PYTHONUNBUFFERED']             = '1'
  os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
  os.environ['MXNET_ENABLE_GPU_P2P']         = '0'

  ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]

  end_epoch = train_net(config, ctx, args.pretrained, args.epochs, config.TRAIN.lr, logger)

  trained_weights = [os.path.join(config.output_path, name) 
                    for name in os.listdir(config.output_path) if name.endswith('.params')]
  
  trained_params = None
  for fn in trained_weights:
    if str(end_epoch) in fn.split('/')[-1]: trained_params = fn
  assert(trained_params is not None),\
    "Cannot find trained params for end epoch {} among {}".format(end_epoch, trained_weights)

  return trained_params

if __name__ == '__main__':
  args = parse_args()
  main(args)
