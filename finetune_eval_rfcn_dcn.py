import os, argparse
from finetune_rfcn_dcn_end2end import main as train
#from eval_model_to_savant_json import main as test

def parse_args():
  parser = argparse.ArgumentParser(description='Run R-FCN-DCN network finetuning and evaluation')
  parser.add_argument('--train', type=int, default=0, help="Specify to run finetuning before evaluation. Or don't to skip finetuning")
  parser.add_argument('--cfg', type=str, required=True, help='resnet_v1_101_voc071217_rfcn_dcn_end2end_ohem.yaml path')
  parser.add_argument('--gpus',type=str, default='0', help='Specify gpu devices for training, split by comma (for testing only 1 gpu will be used)')
  parser.add_argument('--pretrained', type=str, required=True, help='path to pretrained weights')
  parser.add_argument('--epochs', type=int, required=True, help='Amount epochs to finetune')
  parser.add_argument('--dataset_root', type=str, required=True, help='root_path to datasets for training')
  parser.add_argument('--dataset', type=str, required=True, help='datasets used for training')
  parser.add_argument('--image_set', type=str, default='fp_20_train.txt', help='images sets splitted by + sign. Amount must be equal to specified datasets')
  parser.add_argument('--output_path', type=str, default=None, help='folder to store resulting model to')
  parser.add_argument('--frequent', type=int, default=None, help='frequency of logging')
  # next params for evaluation only
  parser.add_argument('--frame_folder',type=str, required=True,     help='folder where eval frames are stored')
  parser.add_argument('--input_list',  type=str, required=True,     help='path to file which lists frame names')
  parser.add_argument('--output_fn',   default='./savant_test.json',help='where to store savant_test.json')
  parser.add_argument('--threshold',   type=float, default=0.1,     help='common detection threshold')
  # params for EvaluationToolBox script
  parser.add_argument('--eval_folder', type=str, required=True,     help="Path to EvaluationToolBox")
  parser.add_argument('--vatic_gt',    type=str, required=True,     help="Path to folder with vatic annotations from arlo")
  parser.add_argument('--eval_video_list', type=str, required=True, help="Path to list of videos")
  return parser.parse_args()

def main(args):
  # finetune the network if required
  if args.train: args.pretrained = train(args)
  # obtain detections against evaluation set using provided params
  args.output_fn = os.path.join(args.output_path, 'savant_test.json')
#  args.display = False
#  result_fn = test(args)
  # evaluate detection results
#  os.chdir(args.eval_folder)
#  os.system("python scripts/EvalFrameWiseObjRecognition.py -e -g {} -d {} -p {} -o {}".
#             format(args.vatic_gt, result_fn, args.eval_video_list, args.output_path))

if __name__ == '__main__':
  main(parse_args())