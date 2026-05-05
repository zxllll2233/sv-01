'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''
import tqdm,soundfile,random
import argparse, glob, os, torch, warnings, time

from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel,add_noise,computer_s,add_cn1


parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=150,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=200,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list', type=str,   default="/home/zhangxl24/SpeakerRecongnition/somedata/train_list_1.txt",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--train_path', type=str,   default="/home/database/sre/voxceleb/voxceleb1/dev/wav",                    help='The path of the training data, eg:"/data08/VoxCeleb2/train/wav" in my case')
parser.add_argument('--eval_list',  type=str,   default="/home/database/sre/voxceleb/voxceleb1/voxceleb1_test_v2.txt",              help='The path of the evaluation list, veri_test2.txt comes from https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt')
parser.add_argument('--eval_path',  type=str,   default="/home/database/sre/voxceleb/voxceleb1/test/wav",                    help='The path of the evaluation data, eg:"/data08/VoxCeleb1/test/wav" in my case')
parser.add_argument('--musan_path', type=str,   default="/home/database/noise/musan",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--cn1_path', type=str,   default="/home/database/sre/CN-Celeb-2022/task1/cn_1/eval/test",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')

#parser.add_argument('--rir_path',   type=str,   default="/home/database/noise/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--save_path',  type=str,   default="/home/zhangxl24/SpeakerRecongnition/wode/Baseline/vox1/clean",                                     help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="",                                          help='Path of the initial_model')
## Model and Loss settings"exps/adv/my2/model/model_0060.model"
parser.add_argument('--C',       type=int,   default=512,   help='Channel size for the speaker encoder')
parser.add_argument('--device',  type=int,   default=0,   help='device')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=1211,   help='Number of speakers')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')
parser.add_argument('--eval_cn1',    dest='--eval_cn1', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)



## Only do evaluation, the initial_model is necessary

mod_list = []



'''if args.eval == True:
	files = []
	embeddings = {}
	noiselist = {}
	score_file = open(args.score_save_path, "a+")

	noisetypes = ['singing', 'interview']


	augment_files = glob.glob(os.path.join(args.cn1_path, '*.flac'))
	for file in augment_files:
		#print(file)#/home/database/sre/CN-Celeb-2022/task1/cn_1/eval/test/id00917-interview-02-012.flac
		if file.split('-')[3] in noisetypes:
			if file.split('-')[3] not in noiselist:
				noiselist[file.split('-')[3]] = []
			noiselist[file.split('-')[3]].append(file)
	for key in noiselist:
		print(key)



	lines = open(args.eval_list).read().splitlines()
	for line in lines:
		files.append(line.split()[1])  # 1 id00800-singing-01-001.flac id00800-singing-01-002.flac
		files.append(line.split()[2])
	setfiles = list(set(files))  # [id00800-singing-01-001.flac....]
	setfiles.sort()

	s_clean = ECAPAModel(**vars(args))# s = ECAPAModel(some_key=new_value, **vars(args))
	s_clean.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Baseline/vox1/clean/model/model_0022.model')
	mod_list.append(s_clean)

	s_base_noise = ECAPAModel(**vars(args))
	s_base_noise.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Baseline/vox1/noise/model/model_0077.model')
	mod_list.append(s_base_noise)

	s_emb = ECAPAModel(**vars(args))
	s_emb.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Baseline/vox1/emb/model/model_0054.model')
	mod_list.append(s_emb)

	s_frame_x3 = ECAPAModel(**vars(args))
	s_frame_x3.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Noise_adv_vox1/exps/5.10/model/model_0056.model')
	mod_list.append(s_frame_x3)

	s_emb_x3 = ECAPAModel(**vars(args))
	s_emb_x3.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Noise_adv_vox1/exps/3.05/model/model_0077.model')
	mod_list.append(s_emb_x3)
	print("Models loaded from previous state!")
	#['singing', 'interview', 'movie','play']
	audioset = {
		"singing": {},
		"interview": {}
	}
	for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):

		audio, _ = soundfile.read(os.path.join(args.eval_path, file))
		audioset["singing"][file] = torch.Tensor(
			add_cn1(numpy.stack([audio], axis=0), noiselist=noiselist,noisecat='singing')).cuda(device=args.device)
		audioset["interview"][file] = torch.Tensor(
			add_cn1(numpy.stack([audio], axis=0), noiselist=noiselist, noisecat='interview')).cuda(device=args.device)


	embset = {
		"singing": {},
		"interview": {}
	}
	Datatype = ['singing', 'interview']
	Model_list = ['clean', 'noise', 'emb','x3','x3+emb']

	score_file = open(args.score_save_path, "a+")

	for datatype in Datatype:
		m_num = 0
		for s in mod_list:
			s_name = Model_list[m_num]
			for idx, file in enumerate(setfiles):
				embset[datatype][file] = s.get_embeddings(audioset[datatype][file])
			eer, mindcf = computer_s(lines, embset[datatype])
			print(f"数据：{datatype}\t模型：{s_name}\tEER:{eer}\tMinDcf:{mindcf}")
			score_file.write(f"数据：{datatype}\t模型：{s_name}\tEER:{eer}\tMinDcf:{mindcf}\n")
			score_file.flush()
			m_num += 1

	quit()'''

if args.eval == True:
	files = []
	embeddings = {}
	score_file = open(args.score_save_path, "a+")
	noisetypes = ['noise', 'speech', 'music']
	noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
	numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
	noiselist = {}
	augment_files = glob.glob(os.path.join(args.musan_path, '*/*/*.wav'))

	for file in augment_files:
		if file.split('/')[-3] not in noiselist:
			noiselist[file.split('/')[-3]] = []
		noiselist[file.split('/')[-3]].append(file)

	for key in noiselist:
		lenk = len(noiselist[key])
		noiselist[key] = noiselist[key][int(lenk / 2):]
	#print(noiselist)
	lines = open(args.eval_list).read().splitlines()
	for line in lines:
		files.append(line.split()[1])  # 1 id00800-singing-01-001.flac id00800-singing-01-002.flac
		files.append(line.split()[2])
	setfiles = list(set(files))  # [id00800-singing-01-001.flac....]
	setfiles.sort()

	s_clean = ECAPAModel(**vars(args))# s = ECAPAModel(some_key=new_value, **vars(args))
	s_clean.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Baseline/vox1/frame_x3_test2/model/model_0076.model')
	mod_list.append(s_clean)

	s_base_noise = ECAPAModel(**vars(args))
	s_base_noise.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Baseline/vox1/frame_x2/model/model_0066.model')
	mod_list.append(s_base_noise)

	s_emb = ECAPAModel(**vars(args))
	s_emb.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Baseline/vox1/convx/model/model_0066.model')
	mod_list.append(s_emb)

	s_frame_x0 = ECAPAModel(**vars(args))
	s_frame_x0.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Baseline/vox1/convx/model/model_0066.model')
	mod_list.append(s_frame_x0)

	s_frame_x1 = ECAPAModel(**vars(args))
	s_frame_x1.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Noise_adv_vox1/exps/5.14-x1/model/model_0082.model')
	mod_list.append(s_frame_x1)

	s_frame_x2 = ECAPAModel(**vars(args))
	s_frame_x2.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Noise_adv_vox1/exps/5.05/model/model_0095.model')
	mod_list.append(s_frame_x2)

	s_frame_x3 = ECAPAModel(**vars(args))
	s_frame_x3.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Noise_adv_vox1/exps/5.10/model/model_0056.model')
	mod_list.append(s_frame_x3)

	print("ddd")
	s_emb_x3 = ECAPAModel(**vars(args))
	s_emb_x3.load_parameters('/home/zhangxl24/SpeakerRecongnition/wode/Noise_adv_vox1/exps/3.05/model/model_0077.model')
	mod_list.append(s_emb_x3)
	print("Models loaded from previous state!")

	#print(add_data_list)
	#print(mod_list)
	audioset = {
		"clean": {},
		"noise": {},
		"speech": {},
		"music": {},
		"all": {}
	}
	for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
		audio, _ = soundfile.read(os.path.join(args.eval_path, file))
		audioset["clean"][file] = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda(device=args.device)

		audioset["noise"][file] = torch.Tensor(
			add_noise(numpy.stack([audio], axis=0), noisecat='noise', numnoise=numnoise, noiselist=noiselist,
					  noisesnr=noisesnr)).cuda(device=args.device)

		audioset["speech"][file] = torch.Tensor(
			add_noise(numpy.stack([audio], axis=0), noisecat='speech', numnoise=numnoise, noiselist=noiselist,
					  noisesnr=noisesnr)).cuda(device=args.device)

		audioset["music"][file] = torch.Tensor(
			add_noise(numpy.stack([audio], axis=0), noisecat='music', numnoise=numnoise, noiselist=noiselist,
					  noisesnr=noisesnr)).cuda(device=args.device)

		audioset["all"][file] = torch.Tensor(
			add_noise(numpy.stack([audio], axis=0), noisecat=noisetypes[random.randint(0, 2)], numnoise=numnoise,
					  noiselist=noiselist,
					  noisesnr=noisesnr)).cuda(device=args.device)

	embset = {
		"clean": {},
		"noise": {},
		"speech": {},
		"music": {},
		"all": {}
	}
	Datatype = ['clean', 'noise', 'speech', 'music','all']
	Model_list = ['clean', 'noise', 'emb','x0', 'x1', 'x2', 'x3','x3+emb']

	score_file = open(args.score_save_path, "a+")

	for datatype in Datatype:
		m_num = 0
		for s in mod_list:
			s_name = Model_list[m_num]
			for idx, file in enumerate(setfiles):
				embset[datatype][file] = s.get_embeddings(audioset[datatype][file])
			eer, mindcf = computer_s(lines, embset[datatype])
			print(f"数据：{datatype}\t模型：{s_name}\tEER:{eer}\tMinDcf:{mindcf}")
			score_file.write(f"数据：{datatype}\t模型：{s_name}\tEER:{eer}\tMinDcf:{mindcf}\n")
			score_file.flush()
			m_num += 1

	quit()
## Define the data loader
trainloader = train_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()
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

EERs = []
MinDCFs = []

score_file = open(args.score_save_path, "a+")
best = 100
while(1):
	## Training for one epoch
	loss, lr, acc1,acc2 = s.train_network(epoch = epoch, loader = trainLoader)

	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		eer1, mindcf1,eer2, mindcf2 = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
		mindcf = min(mindcf1, mindcf2)
		eer = min(eer1, eer2)
		if eer < best or epoch % 10 == 1:
			s.save_parameters(args.model_save_path + "/model_%04d.model" % epoch)
			best = min(eer, best)
		MinDCFs.append(mindcf)
		EERs.append(eer)
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, cleanACC %2.2f%%, noiseACC %2.2f%%,  EER1 %2.2f%%,EER2 %2.2f%%, bestEER %2.2f%%,MinDCF1 %.4f%%,MinDCF2 %.4f%%, bestMinDCF %.4f%%"%(epoch, acc1,acc2, eer1,eer2, min(EERs), mindcf1,mindcf2, min(MinDCFs)))
		score_file.write("%d epoch, LR %f, LOSS %f, cleanACC %2.2f%%,noiseACC %2.2f%%, EER1 %2.2f%%,EER2 %2.2f%% bestEER %2.2f%%,MinDCF1 %.4f%%,MinDCF2 %.4f%%, bestMinDCF %.4f%%\n"%(epoch, lr, loss, acc1,acc2, eer1,eer2, min(EERs),mindcf1,mindcf2, min(MinDCFs)))
		score_file.flush()

	if epoch >= args.max_epoch:
		quit()

	epoch += 1
