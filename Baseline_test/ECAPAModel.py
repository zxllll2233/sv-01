'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle,glob,random
import torch.nn as nn
from torch.cuda import device

from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
from attribution import ECAPAAttributionAnalyzer

def add_noise(audio, noisecat,numnoise,noiselist,noisesnr):
	clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
	numnoise = numnoise[noisecat]
	noiselist = random.sample(noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
	noises = []
	for noise in noiselist:
		noiseaudio, sr = soundfile.read(noise)
		length = audio.shape[1]
		if noiseaudio.shape[0] <= length:
			shortage = length - noiseaudio.shape[0]
			noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
		start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
		noiseaudio = noiseaudio[start_frame:start_frame + length]
		noiseaudio = numpy.stack([noiseaudio], axis=0)
		noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
		noisesnr = random.uniform(noisesnr[noisecat][0], noisesnr[noisecat][1])
		noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
	noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
	return noise + audio

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step,device ,**kwargs):
		super(ECAPAModel, self).__init__()
		self.devicce = device
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda(device = self.devicce)
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda(device = self.devicce)

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

		# Attribution analysis
		self.attribution_samples = []
		self.attribution_analyzer = None
		self.C = C  # 保存C参数用于初始化分析器

	def set_attribution_samples(self, sample_paths):
		self.attribution_samples = sample_paths

	def run_attribution_analysis(self, epoch, save_dir):
		if not self.attribution_samples:
			return

		print(f"[Attribution] Epoch {epoch}: 开始归因分析...")
		t_start = time.time()

		try:
			# Lazy initialization
			if self.attribution_analyzer is None:
				# 初始化 Attribution Analyzer (Integrated Gradients)
				self.attribution_analyzer = ECAPAAttributionAnalyzer(
					self.speaker_encoder,
					C=self.C,
					device=self.devicce
				)

			# 修改为直接传入save_dir，由analyzer内部处理子目录结构
			self.attribution_analyzer.analyze_and_save(
				self.attribution_samples,
				save_dir,
				epoch
			)
			# 清理hooks，防止影响后续训练
			self.attribution_analyzer.cleanup()
			
			t_end = time.time()
			print(f"[Attribution] 分析完成，耗时 {t_end - t_start:.1f}s")
			
		except Exception as e:
			print(f"[Attribution] Error during analysis: {str(e)}")
			import traceback
			traceback.print_exc()
			# 确保出错也能清理hooks
			if self.attribution_analyzer:
				self.attribution_analyzer.cleanup()

	def cleanup_attribution(self):
		if self.attribution_analyzer:
			self.attribution_analyzer.cleanup()
			self.attribution_analyzer = None

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		for num, (data, labels) in enumerate(loader, start = 1):
			self.zero_grad()
			labels            = torch.LongTensor(labels).cuda(self.devicce)
			speaker_embedding = self.speaker_encoder.forward(data.cuda(self.devicce), aug = False)
			nloss, prec       = self.speaker_loss.forward(speaker_embedding, labels)			
			nloss.backward()
			self.optim.step()
			index += len(labels)
			top1 += prec
			loss += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list, eval_path):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))

			# Full utterance
			data_1 = numpy.stack([audio], axis=0)
			data_1 = torch.FloatTensor(data_1).cuda(device = self.devicce)
			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			data_2 = numpy.stack(feats, axis = 0).astype(numpy.float)
			data_2 = torch.FloatTensor(data_2).cuda(device=self.devicce)
			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores1, scores2, labels = [], [], []

		for line in lines:
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))  # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))

			score_1 = score_1.detach().cpu().numpy()
			score_2 = score_2.detach().cpu().numpy()
			scores1.append(score_1)
			scores2.append(score_2)
			labels.append(int(line.split()[0]))

		# Coumpute EER and minDCF
		EER1 = tuneThresholdfromScore(scores1, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores1, labels)
		minDCF1, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		EER2 = tuneThresholdfromScore(scores2, labels, [1, 0.1])[1]
		fnrs2, fprs2, thresholds2 = ComputeErrorRates(scores2, labels)
		minDCF2, _ = ComputeMinDcf(fnrs2, fprs2, thresholds2, 0.05, 1, 1)

		return EER1, minDCF1, EER2, minDCF2

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		print(f"{path}")
		loaded_state = torch.load(path, map_location='cpu')
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)