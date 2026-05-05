'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle,glob,random
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
from adv import DG_frame, DG_embedding


def add_cn1(audio,noiselist,noisecat):
	clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
	numnoise = [3, 7]
	noisesnr = [13, 20]
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
		snr = random.uniform(noisesnr[0], noisesnr[1])
		noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noiseaudio)
	noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
	return noise + audio
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
		snr = random.uniform(noisesnr[noisecat][0], noisesnr[noisecat][1])
		noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noiseaudio)
	noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
	return noise + audio

def computer_s(lines,embeddings):
	scores, labels = [], []
	for line in lines:
		embedding_1 = embeddings[line.split()[1]]
		embedding_2 = embeddings[line.split()[2]]
		score = torch.mean(torch.matmul(embedding_1, embedding_2.T))  # higher is positive
		score = score.detach().cpu().numpy()
		scores.append(score)
		labels.append(int(line.split()[0]))

	# Coumpute EER and minDCF
	EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
	fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
	minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

	return EER, minDCF

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step,device,max_epoch,**kwargs):
		super(ECAPAModel, self).__init__()
		self.device =device
		self.max_epoch = max_epoch
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda(device=self.device)
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda(device=self.device)

		self.DG_1 = DG_frame(out_channel=2).cuda(device=self.device)
		self.DG_2 = DG_frame(out_channel=3).cuda(device=self.device)
		self.DG_3 = DG_embedding().cuda(device=self.device)

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		self.mse = nn.MSELoss(reduction='mean').cuda(device=self.device)
		'''print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%((sum(param.numel() for param in self.speaker_encoder.parameters())
																			  +sum(param.numel() for param in self.DG_1.parameters())
																			  + sum(param.numel() for param in self.DG_2.parameters())
																			 )/ 1024 / 1024))'''

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		lr = self.optim.param_groups[0]['lr']

		## Initial loss
		index = 0

		spk_loss,top_clean,top_noise = 0,0,0

		f_noise_loss,e_noise_loss,f_noise_top,e_noise_top = 0,0,0,0

		aug_loss,which_aug_top = 0,0

		mse_loss,rec_loss_1,rec_loss_2 = 0,0,0


		for num, (data, noise_data, labels,augtype) in enumerate(loader, start = 1):
			self.zero_grad()

			constant = 0.1
			alpha = 0.05*epoch

			s_labels = torch.ones_like(labels)
			s_labels = torch.LongTensor(s_labels).cuda(device=self.device)
			t_labels = torch.zeros_like(labels)
			t_labels = torch.LongTensor(t_labels).cuda(device=self.device)

			labels = torch.LongTensor(labels).cuda(device=self.device)
			augtype_label = torch.LongTensor(augtype).cuda(device=self.device)

			clean_frame, clean_embedding = self.speaker_encoder.forward(data.cuda(device=self.device), aug = False)
			noise_frame, noise_embedding = self.speaker_encoder.forward(noise_data.cuda(device=self.device), aug = False)

			Mse_loss = self.mse(clean_embedding.detach(), noise_embedding)

			#说话人损失
			clean_nloss, clean_prec       = self.speaker_loss.forward(clean_embedding, labels)
			noise_nloss, noise_prec 		= self.speaker_loss.forward(noise_embedding, labels)
			nloss = clean_nloss + noise_nloss


			#emb
			'''clean_emb_dis_loss, clean_emb_pre = self.DG_3.forward(clean_embedding , s_labels, constant=constant)
			noise_emb_dis_loss, noise_emb_pre = self.DG_3.forward(noise_embedding, t_labels, constant=constant)
			emb_dis_loss = clean_emb_dis_loss + noise_emb_dis_loss
			emb_pre = (clean_emb_pre + noise_emb_pre) / 2

			#帧级别 噪声存在判别
			clean_frame_dis_loss,clean_frame_rec_loss, clean_frame_pre = self.DG_1.forward(clean_frame, s_labels,constant=constant)
			noise_frame_dis_loss, noise_frame_rec_loss,noise_frame_pre = self.DG_1.forward(noise_frame, t_labels,constant=constant)

			frame_dis_loss = clean_frame_dis_loss + noise_frame_dis_loss
			#frame_rec_loss = clean_frame_rec_loss + noise_frame_rec_loss

			frame_pre   = (clean_frame_pre + noise_frame_pre)/2

			# 帧级别 噪声类型判别
			aug_frame_loss,aug_frame_rec_loss, aug_frame_pre = self.DG_2.forward(noise_frame, augtype_label, constant=constant)

			#对抗 损失
			Dis_loss = frame_dis_loss + aug_frame_loss#总的 域loss'''
			#Rec_loss = (frame_rec_loss + aug_frame_rec_loss)*0.0001'''

			#loss反传
			total_loss = nloss# +(Dis_loss )*alpha + Mse_loss*10*alpha
			total_loss.backward()
			self.optim.step()

			index += len(labels)

			#prec
			top_clean += clean_prec
			top_noise += noise_prec
			#f_noise_top += frame_pre
			#e_noise_top += emb_pre
			#which_aug_top += aug_frame_pre

			
			#loss
			spk_loss += nloss.detach().cpu().numpy()		#nloss = clean_nloss + noise_nloss
			#f_noise_loss += frame_dis_loss.detach().cpu().numpy()
			#e_noise_loss += emb_dis_loss.detach().cpu().numpy()
			#aug_loss += aug_frame_loss.detach().cpu().numpy()
			mse_loss += Mse_loss.detach().cpu().numpy()
			#rec_loss_1 += frame_rec_loss.detach().cpu().numpy()
			#rec_loss_2 += aug_frame_rec_loss.detach().cpu().numpy()

			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
							 " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
							 " spkLoss: %.3f, c_ACC: %2.2f%%, n_ACC: %2.2f%% || f_loss: %.5f,f_ACC: %2.2f%%,a_loss: %.5f,a_ACC: %2.2f%% ,e_loss: %.5f,e_ACC: %2.2f%%,mseloss : %.5f\r"
							 % (spk_loss / (num), top_clean / index * len(labels), top_noise / index * len(labels),
								f_noise_loss / (num), f_noise_top / index * len(labels),
								aug_loss / (num), which_aug_top / index * len(labels),
								e_noise_loss / (num),  e_noise_top / index * len(labels),mse_loss/(num)
								))
			'''sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
							 " [%2d] Lr: %5f, Training: %.2f%%, " % (epoch, lr, 100 * (num / loader.__len__())) + \
							 " spkLoss: %.3f, c_ACC: %2.2f%%, n_ACC: %2.2f%% || f_loss: %.5f,f_ACC: %2.2f%%,a_loss: %.5f,a_ACC: %2.2f%% ,mseloss : %.5f || rec1:%.5f, rec2:%.5f\r"
							 % (spk_loss / (num), top_clean / index * len(labels), top_noise / index * len(labels),
								f_noise_loss / (num), f_noise_top / index * len(labels),
								aug_loss / (num), which_aug_top / index * len(labels),
								mse_loss / (num), rec_loss_1, rec_loss_2
								))'''
			sys.stderr.flush()
		sys.stdout.write("\n")

		return spk_loss/num, lr, top_clean/index*len(labels),top_noise/index*len(labels)

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

		for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
			audio, _ = soundfile.read(os.path.join(eval_path, file))
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda(device=self.device)

			# Spliited utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf) + max_audio])
			feats = numpy.stack(feats, axis=0).astype(numpy.float)
			data_2 = torch.FloatTensor(feats).cuda(device=self.device)
				# Speaker embeddings
			with torch.no_grad():
				_,embedding_1 = self.speaker_encoder.forward(data_1, aug=False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				_,embedding_2 = self.speaker_encoder.forward(data_2, aug=False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores1, scores2,labels = [], [], []

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

	def get_embeddings(self,audio):
		self.eval()
		with torch.no_grad():
			_, embedding_1 = self.speaker_encoder.forward(audio, aug=False)
			embedding_1 = F.normalize(embedding_1, p=2, dim=1)

		return embedding_1
	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self_state = self.state_dict()
		loaded_state = torch.load(path, map_location=device)
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


'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle,glob,random
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
from adv import DG_frame,DG_embedding

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
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step,device,**kwargs):
		super(ECAPAModel, self).__init__()
		self.device =device
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda(device=self.device)
		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda(device=self.device)

		self.DG_frame = DG_frame(C=C).cuda(device=self.device)
		self.DG_embedding = DG_embedding().cuda(device=self.device)

		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		self.mse = nn.MSELoss(reduction='mean').cuda(device=self.device)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%((sum(param.numel() for param in self.speaker_encoder.parameters())
																			  +sum(param.numel() for param in self.DG_frame.parameters())
																			  +sum(param.numel() for param in self.DG_embedding.parameters()))/ 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, m_loss,s1_top_speaker, s2_top_speaker, top_f_dis ,top_e_dis,f_loss,f_rec_loss,e_loss,e_rec_loss,s_loss = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		for num, (data, noise_data, labels,augtype) in enumerate(loader, start = 1):
			self.zero_grad()

			s_labels = torch.ones_like(labels)
			s_labels = torch.LongTensor(s_labels).cuda(device=self.device)
			t_labels = torch.zeros_like(labels)
			t_labels = torch.LongTensor(t_labels).cuda(device=self.device)

			labels = torch.LongTensor(labels).cuda(device=self.device)
			augtype_label = torch.LongTensor(augtype).cuda(device=self.device)

			s1_x_frame, s1_speaker_embedding = self.speaker_encoder.forward(data.cuda(device=self.device), aug = True)
			s2_x_frame, s2_speaker_embedding = self.speaker_encoder.forward(noise_data.cuda(device=self.device), aug=True)

			Mse_loss = self.mse(s2_speaker_embedding, s1_speaker_embedding.detach())

			#说话人损失
			s1_nloss, s1_prec       = self.speaker_loss.forward(s1_speaker_embedding, labels)
			s2_nloss, s2_prec 		= self.speaker_loss.forward(s2_speaker_embedding, labels)
			nloss = s1_nloss + s2_nloss
			prec = (s1_prec + s2_prec)/2

			#帧级别判别损失
			s1_frame_dis_loss, s1_frame_rec_loss, s1_frame_pre = self.DG_frame.forward(s1_x_frame, s_labels,constant=0.1)
			s2_frame_dis_loss, s2_frame_rec_loss, s2_frame_pre = self.DG_frame.forward(s2_x_frame, t_labels,constant=0.1)

			frame_dis_loss = (s1_frame_dis_loss + s2_frame_dis_loss)*0.1
			frame_rec_loss = (s1_frame_rec_loss + s2_frame_rec_loss)*0.1

			frame_pre   = (s1_frame_pre + s2_frame_pre)/2
			frame_loss = frame_dis_loss + frame_rec_loss

			#embedding级别判别损失
			s1_emb_dis_loss, s1_emb_rec_loss, s1_emb_pre = self.DG_embedding.forward(s1_speaker_embedding, s_labels,constant=0.1)
			s2_emb_dis_loss, s2_emb_rec_loss, s2_emb_pre = self.DG_embedding.forward(s2_speaker_embedding, t_labels,constant=0.1)

			emb_dis_loss = (s1_emb_dis_loss + s2_emb_dis_loss)*0.1 #emb判别loss
			emb_rec_loss = (s1_emb_rec_loss + s2_emb_rec_loss)*0.1 #emb重构loss

			emb_pre = (s1_emb_pre + s2_emb_pre)/2
			emb_loss = emb_dis_loss + emb_rec_loss  #emb  loss

			#判别损失
			dis_loss = frame_loss + emb_loss #总的 域loss
			dis_pre = (frame_pre + emb_pre)/2

			#loss反传
			total_loss = nloss + Mse_loss + dis_loss*0.05*epoch
			total_loss.backward()
			self.optim.step()

			#prec
			index += len(labels)
			s1_top_speaker += s1_prec
			s2_top_speaker += s2_prec
			top_f_dis += frame_pre
			top_e_dis += emb_pre


			#loss

			s_loss += nloss.detach().cpu().numpy()
			f_loss += frame_dis_loss.detach().cpu().numpy()
			f_rec_loss += frame_rec_loss.detach().cpu().numpy()
			e_loss += emb_dis_loss.detach().cpu().numpy()
			e_rec_loss += emb_rec_loss.detach().cpu().numpy()
			m_loss += Mse_loss.detach().cpu().numpy()
			#显示
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" sLoss: %.3f, clean_ACC: %2.2f%%, noise_ACC: %2.2f%% ,mse_loss: %.3f|| f_loss: %.3f,f_rec_loss: %.3f,e_loss: %.3f,e_rec_loss: %.3f,f_ACC: %2.2f%% ,e_ACC: %2.2f%%\r"
							 %(s_loss/(num), s1_top_speaker/index*len(labels),s2_top_speaker/index*len(labels),m_loss/(num),
							   f_loss / (num),f_rec_loss / (num),e_loss/(num),e_rec_loss / (num),top_f_dis/index*len(labels),top_e_dis/index*len(labels)
			))
			sys.stderr.flush()
		sys.stdout.write("\n")

		return s_loss/num, lr, s1_top_speaker/index*len(labels),s2_top_speaker/index*len(labels)

	def eval_network(self, eval_list, eval_path, musan_path, noise=True, noisecat = None, snr = None):
		self.eval()
		files = []
		embeddings = {}
		lines = open(eval_list).read().splitlines()
		for line in lines:
			files.append(line.split()[1])
			files.append(line.split()[2])
		setfiles = list(set(files))
		setfiles.sort()
		if noise:
			noisetypes = ['noise', 'speech', 'music']
			noisesnr = {'noise': [snr,snr], 'speech': [snr,snr], 'music': [snr,snr]}
			numnoise = {'noise': [1, 1], 'speech': [1,1], 'music': [1, 1]}
			noiselist = {}
			augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
			for file in augment_files:
				if file.split('/')[-3] not in noiselist:
					noiselist[file.split('/')[-3]] = []
				noiselist[file.split('/')[-3]].append(file)
			for key in noiselist:
				lenk = len(noiselist[key])
				noiselist[key] = noiselist[key][int(lenk / 2):]


		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))

			# Full utterance
			data_1 = numpy.stack([audio], axis=0)
			if noise:
				data_1 = add_noise(data_1,noisecat=noisecat,numnoise=numnoise,noiselist=noiselist,noisesnr=noisesnr)
			data_1 = torch.FloatTensor(data_1).cuda(device=self.device)
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
			if noise:
				data_2 = add_noise(data_2, noisecat=noisecat, numnoise=numnoise, noiselist=noiselist, noisesnr=noisesnr)
			data_2 = torch.FloatTensor(data_2).cuda(device=self.device)
			# Speaker embeddings
			with torch.no_grad():
				_,embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				_,embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		for line in lines:			
			embedding_11, embedding_12 = embeddings[line.split()[1]]
			embedding_21, embedding_22 = embeddings[line.split()[2]]
			# Compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			#score = (score_1 + score_2) / 2
			score = score_1
			score = score.detach().cpu().numpy()
			scores.append(score)
			labels.append(int(line.split()[0]))
			
		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		return EER, minDCF

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
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
			self_state[name].copy_(param)'''