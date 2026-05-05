'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle,glob,random
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
from adv import DG_frame,DG_embedding,DG_embedding_no_encoder
import numpy as np
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
        #print('hello')
        #print(noisecat)
        #print(f"noisesnr{noisesnr}")
        snr = random.uniform(noisesnr[noisecat][0], noisesnr[noisecat][1])
        noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * noiseaudio)
    noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
    return noise + audio

class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C , n_class, m, s, test_step,device, max_epoch,**kwargs):
        super(ECAPAModel, self).__init__()
        self.device =device
        ## ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C = C).cuda(device=self.device)
        ## Classifier
        self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda(device=self.device)

        self.DG_frame = DG_frame(C=C).cuda(device=self.device)
        self.DG_embedding = DG_embedding_no_encoder().cuda(device=self.device)

        self.max_epoch = max_epoch

        self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
        self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
        self.mse = nn.MSELoss(reduction='mean').cuda(device=self.device)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%((sum(param.numel() for param in self.speaker_encoder.parameters())
                                                                              +sum(param.numel() for param in self.DG_frame.parameters())
                                                                              +sum(param.numel() for param in self.DG_embedding.parameters()))/ 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        start_steps = epoch * len(loader)
        total_steps = self.max_epoch * len(loader)
        ## Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, m_loss,s1_top_speaker, s2_top_speaker, top_f_dis ,top_e_dis,f_loss,f_rec_loss,e_loss,e_rec_loss,s_loss = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, noise_data, labels) in enumerate(loader, start = 1):
            self.zero_grad()
            p = float((num + start_steps) / total_steps)
            constant = 2. / (1. + np.exp(-10 * p)) - 1
            alpha = 1./(1.+ p*10)**0.75
            s_labels = torch.ones_like(labels)
            s_labels = torch.LongTensor(s_labels).cuda(device=self.device)
            t_labels = torch.zeros_like(labels)
            t_labels = torch.LongTensor(t_labels).cuda(device=self.device)

            labels = torch.LongTensor(labels).cuda(device=self.device)

            s1_x_frame, s1_speaker_embedding = self.speaker_encoder.forward(data.cuda(device=self.device), aug = True)
            s2_x_frame, s2_speaker_embedding = self.speaker_encoder.forward(noise_data.cuda(device=self.device), aug=True)

            Mse_loss = self.mse(s2_speaker_embedding, s1_speaker_embedding.detach())

            #说话人损失
            s1_nloss, s1_prec       = self.speaker_loss.forward(s1_speaker_embedding, labels)
            s2_nloss, s2_prec 		= self.speaker_loss.forward(s2_speaker_embedding, labels)
            nloss = s1_nloss + s2_nloss
            prec = (s1_prec + s2_prec)/2

            #帧级别判别损失
            s1_frame_dis_loss, s1_frame_rec_loss, s1_frame_pre = self.DG_frame.forward(s1_x_frame, s_labels,constant=constant)
            s2_frame_dis_loss, s2_frame_rec_loss, s2_frame_pre = self.DG_frame.forward(s2_x_frame, t_labels,constant=constant)

            frame_dis_loss = (s1_frame_dis_loss + s2_frame_dis_loss)*0.1
            frame_rec_loss = (s1_frame_rec_loss + s2_frame_rec_loss)*0.1

            frame_pre   = (s1_frame_pre + s2_frame_pre)/2
            frame_loss = frame_dis_loss + frame_rec_loss

            #embedding级别判别损失
            s1_emb_dis_loss,s1_emb_pre = self.DG_embedding.forward(s1_speaker_embedding, s_labels,constant=constant)
            s2_emb_dis_loss, s2_emb_pre = self.DG_embedding.forward(s2_speaker_embedding, t_labels,constant=constant)

            emb_dis_loss = (s1_emb_dis_loss + s2_emb_dis_loss)*0.1
            #emb_rec_loss = (s1_emb_rec_loss + s2_emb_rec_loss)*0.1

            emb_pre = (s1_emb_pre + s2_emb_pre)/2
            emb_loss = emb_dis_loss# + emb_rec_loss

            #判别损失
            dis_loss = frame_loss + emb_loss
            dis_pre = (frame_pre + emb_pre)/2

            #loss反传
            total_loss = nloss + Mse_loss + dis_loss*alpha
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
            #e_rec_loss += emb_rec_loss.detach().cpu().numpy()
            m_loss += Mse_loss.detach().cpu().numpy()
            #显示
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " sLoss: %.3f, clean_ACC: %2.2f%%, noise_ACC: %2.2f%% ,mse_loss: %.3f|| f_loss: %.3f,f_rec_loss: %.3f,e_loss: %.3f,f_ACC: %2.2f%% ,e_ACC: %2.2f%%\r"
                             %(s_loss/(num), s1_top_speaker/index*len(labels),s2_top_speaker/index*len(labels),m_loss/(num),
                               f_loss / (num),f_rec_loss / (num),e_loss/(num),top_f_dis/index*len(labels),top_e_dis/index*len(labels)
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
            numnoise = {'noise': [1, 1], 'speech': [1, 1], 'music': [1, 1]}
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
            self_state[name].copy_(param)