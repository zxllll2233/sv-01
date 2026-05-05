'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch
from scipy import signal
from torch.onnx.symbolic_opset9 import tensor


class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, num_frames, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		# Load and configure augmentation files
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,7], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-3] not in self.noiselist:
				self.noiselist[file.split('/')[-3]] = []
			self.noiselist[file.split('/')[-3]].append(file)
		for key in self.noiselist:
			lenk = len(self.noiselist[key])
			self.noiselist[key] = self.noiselist[key][0:int(lenk/2)]
		#self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
		# Load data & labels
		self.data_list  = []
		self.data_label = []
		lines = open(train_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name     = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		audio, sr = soundfile.read(self.data_list[index])		
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')
		start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
		audio = audio[start_frame:start_frame + length]
		audio = numpy.stack([audio],axis=0)
		# Data Augmentation
		augtype = random.randint(0,2)
		if augtype == 0: # Babble
			noise_audio = self.add_noise(audio, 'speech')
		elif augtype == 1: # Music
			noise_audio = self.add_noise(audio, 'music')
		elif augtype == 2: # Noise
			noise_audio = self.add_noise(audio, 'noise')

		return torch.FloatTensor(audio[0]), torch.FloatTensor(noise_audio[0]),self.data_label[index], augtype

	def __len__(self):
		return len(self.data_list)

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio