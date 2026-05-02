'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import argparse, glob, os, torch, warnings, time
from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=300,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/home/zhangxl24/SpeakerRecongnition/somedata/train_list_1.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="/home/database/sre/voxceleb/voxceleb1/dev/wav",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/home/database/sre/voxceleb/voxceleb1/voxceleb1_test_v2.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="/home/database/sre/voxceleb/voxceleb1/test/wav",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')

#parser.add_argument('--train_list', type=str,   default="/home/zhangxl24/SpeakerRecongnition/somedata/cn_train_list_1.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
#parser.add_argument('--train_path', type=str,   default="/home/database/sre/CN-Celeb-2022/task1/cn_1/data",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
#parser.add_argument('--eval_list',  type=str,   default="/home/database/sre/CN-Celeb-2022/task1/cn_1/eval/cnceleb_test_v1.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
#parser.add_argument('--eval_path',  type=str,   default="/home/database/sre/CN-Celeb-2022/task1/cn_1/eval/test",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--save_path',  type=str,   default="vox1",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')
parser.add_argument('--musan_path', type=str,   default="/home/database/noise/musan",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--noisecat',  type=str,   default="",                                          help='Path of the initial_model')

## Attribution settings
parser.add_argument('--attribution_interval', type=int, default=10, help='Interval of epochs to run attribution analysis')
parser.add_argument('--attribution_samples', type=str, default="", help='Comma separated list of audio paths for attribution analysis')

## Model and Loss settings
parser.add_argument('--device',  type=int,   default=3,   help='device')
parser.add_argument('--C',       type=int,   default=512,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=1211,   help='Number of speakers')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

## Define the data loader
trainloader = train_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	EER1, minDCF1,EER2, minDCF2 = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	print("EER1 %2.2f%%, minDCF1 %.4f%%,EER2 %2.2f%%, minDCF2 %.4f%%"%(EER1, minDCF1,EER2, minDCF2))
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
if len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = ECAPAModel(**vars(args))

## Setup attribution samples
if args.attribution_samples:
	samples = args.attribution_samples.split(',')
	s.set_attribution_samples(samples)
else:
	# 如果未指定，尝试从训练列表随机选择3个样本
	try:
		with open(args.train_list, 'r') as f:
			lines = f.readlines()
			if len(lines) >= 3:
				samples = []
				# 简单解析训练列表，假设格式: speaker_id audio_path
				# 或者像dataLoader里那样根据实际情况解析
				# 这里做一个鲁棒的尝试
				import random
				selected_lines = random.sample(lines, 3)
				for line in selected_lines:
					parts = line.strip().split()
					# 尝试找到看起来像路径的部分
					for part in parts:
						if '/' in part or '.wav' in part or '.flac' in part:
							# 如果是相对路径，可能需要拼接train_path
							full_path = os.path.join(args.train_path, part)
							if os.path.exists(full_path):
								samples.append(full_path)
							elif os.path.exists(part):
								samples.append(part)
							break
				if len(samples) == 3:
					print(f"[Attribution] Automatically selected samples: {samples}")
					s.set_attribution_samples(samples)
	except Exception as e:
		print(f"[Attribution] Failed to auto-select samples: {e}")

EERs = []
MinDCFs = []
score_file = open(args.score_save_path, "a+")
best = 100
while(1):
	## Training for one epoch
	loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)

	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		eer1, mindcf1,eer2, mindcf2 =s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
		mindcf = min(mindcf1, mindcf2)
		eer = min(eer1, eer2)
		if (eer<best and epoch >=10) or epoch % 10 ==1:
			s.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)
		best = min(eer, best)
		MinDCFs.append(mindcf)
		EERs.append(eer)
		print("EER1 %2.2f%%, minDCF1 %.4f%%,EER2 %2.2f%%, minDCF2 %.4f%%" % (eer1, mindcf1,eer2, mindcf2))
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%,MinDCF %2.2f%%, MinDCF %2.2f%%"%(epoch, acc, EERs[-1], min(EERs), MinDCFs[-1], min(MinDCFs)))
		score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%,MinDCF %2.2f%%, MinDCF %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs),MinDCFs[-1], min(MinDCFs)))
		score_file.flush()

	## Attribution analysis
	if epoch % args.attribution_interval == 0:
		attribution_save_dir = os.path.join(args.save_path, "attribution")
		s.run_attribution_analysis(epoch, attribution_save_dir)

	if epoch >= args.max_epoch:
		s.cleanup_attribution() # Cleanup before exit
		quit()

	epoch += 1
