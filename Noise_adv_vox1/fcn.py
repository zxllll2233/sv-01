import torch
from tools import *
from torch import nn
from torch.autograd import Function

class GRLayer(Function):

    @staticmethod
    def forward(ctx, input,constant):
        ctx.alpha=constant
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output,None

def grad_reverse(x,constant):
    return GRLayer.apply(x,constant)


class Frame_Encoder_1(nn.Module):
    def __init__(self):
        super(Frame_Encoder_1, self).__init__()

        self.encoder1 = nn.Sequential(
            # Encoder
            # input (b, 1536, 202)
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False,dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1, bias=False,dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 192, kernel_size=3, stride=1, padding=1, bias=False,dilation=1),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            # output (b, 128, 51)
        )

    def forward(self, *input):
        out = self.encoder1(*input)
        return out

class Frame_Decoder_1(nn.Module):
    def __init__(self):
        super(Frame_Decoder_1, self).__init__()

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(192, 256, kernel_size=3, stride=1, padding=1,  bias=False, dilation=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 512, kernel_size=3, stride=1, padding=1, bias=False, dilation=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False, dilation=1),
            nn.ReLU()
            # output (b, 512, t)
        )

    def forward(self, input):
        out = self.decoder1(input)
        return out

class Frame_Encoder_2(nn.Module):
    def __init__(self):
        super(Frame_Encoder_2, self).__init__()

        self.encoder2 = nn.Sequential(
            # Encoder
            # input (b, 1536, 202)
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, bias=False,dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=2, bias=False,dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 192, kernel_size=3, stride=1, padding=2, bias=False,dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            # output (b, 128, t)
        )

    def forward(self, input):
        out = self.encoder2(input)
        return out

class Frame_Decoder_2(nn.Module):
    def __init__(self):
        super(Frame_Decoder_2, self).__init__()

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(192, 256, kernel_size=3, stride=1, padding=2,  bias=False, dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 512, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.ConvTranspose1d(512, out_channels=512, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
            nn.ReLU()
            # output (b, 512, t)
        )

    def forward(self, input):
        out = self.decoder2(input)
        return out

class Frame_Encoder_3(nn.Module):
    def __init__(self):
        super(Frame_Encoder_3, self).__init__()

        self.encoder3 = nn.Sequential(
            # Encoder
            # input (b, 1536, 202)
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=3, bias=False,dilation=3),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=3, bias=False,dilation=3),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 192, kernel_size=3, stride=1, padding=3, bias=False,dilation=3),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            # output (b, 128, 51)
        )

    def forward(self, input):
        out = self.encoder3(input)
        return out

class Frame_Decoder_3(nn.Module):
    def __init__(self):
        super(Frame_Decoder_3, self).__init__()

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(192, 256, kernel_size=3, stride=1, padding=3,  bias=False, dilation=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 512, kernel_size=3, stride=1, padding=3, bias=False, dilation=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=3, bias=False, dilation=3),
            nn.ReLU()
            # output (b, 512, 302)
        )

    def forward(self, input):
        out = self.decoder3(input)
        return out

class frame_discriminator(nn.Module):
    def __init__(self,out_channel):
        super(frame_discriminator,self).__init__()
        self.out_channel =out_channel
        self.discriminator = nn.Sequential(
            nn.Conv1d(192, 512, kernel_size=1, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(512, self.out_channel, kernel_size=1, stride=1, padding=1, bias=False),
        )

    def forward(self, x,constant):
        x = grad_reverse(x,constant)
        out = self.discriminator(x)
        return out #(b,2,50)

class DG_frame(nn.Module):
    def __init__(self,out_channel):
        super(DG_frame, self).__init__()
        self.out_channel = out_channel
        self.frame_encoder1 = Frame_Encoder_1()
        self.frame_decoder1 = Frame_Decoder_1()
        self.frame_discriminator1 = frame_discriminator(out_channel=self.out_channel)

        self.frame_encoder2 = Frame_Encoder_2()
        self.frame_decoder2 = Frame_Decoder_2()
        self.frame_discriminator2 = frame_discriminator(out_channel=self.out_channel)

        self.frame_encoder3 = Frame_Encoder_3()
        self.frame_decoder3 = Frame_Decoder_3()
        self.frame_discriminator3 = frame_discriminator(out_channel=self.out_channel)


        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        #self.ln_frame = nn.LayerNorm(normalized_shape=[76])

    def forward(self, x, labels=None, constant=None):

        rec_x1 = self.frame_encoder1(x)
        rec_x2 = self.frame_encoder2(x)
        rec_x3 = self.frame_encoder3(x)

        dec_x1 = self.frame_decoder1(rec_x1)
        dec_x2 = self.frame_decoder2(rec_x2)
        dec_x3 = self.frame_decoder3(rec_x3)

        rec_l1 = self.mse(x.detach(), dec_x1)
        rec_l2 = self.mse(x.detach(), dec_x2)
        rec_l3 = self.mse(x.detach(), dec_x3)


        output1 = self.frame_discriminator1(rec_x1,constant)
        output2 = self.frame_discriminator2(rec_x2,constant)
        output3 = self.frame_discriminator3(rec_x3,constant) # [batch,num_domain,t]



        loss1,loss2,loss3 = 0,0,0
        prec1,prec2,prec3 = 0,0,0

        for i in range(output1.shape[2]):
            loss1+=self.ce(output1[:, :, i], labels)
            prec1 += accuracy(output1[:, :, i].detach(), labels.detach(), topk=(1,))[0]
        loss1 = loss1/output1.shape[2]
        prec1 = prec1/output1.shape[2]

        for i in range(output2.shape[2]):
            loss2+=self.ce(output2[:, :, i], labels)
            prec2 += accuracy(output2[:, :, i].detach(), labels.detach(), topk=(1,))[0]
        loss2 = loss2/output2.shape[2]
        prec2 = prec2/output2.shape[2]

        for i in range(output3.shape[2]):
            loss3+=self.ce(output3[:, :, i], labels)
            prec3 += accuracy(output3[:, :, i].detach(), labels.detach(), topk=(1,))[0]
        loss3 = loss3/output3.shape[2]
        prec3 = prec3/output3.shape[2]

        return (loss1+loss2+loss3)/3,(rec_l1+rec_l2+rec_l3)/3,(prec1+prec2+prec3)/3

##########################################################

class Emb_Encoder(nn.Module):
    def __init__(self):
        super(Emb_Encoder, self).__init__()

        self.encoder = nn.Sequential(
            # Encoder
            nn.Linear(192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 192)
        )

    def forward(self, *input):
        out = self.encoder(*input)
        return out

class Emb_Decoder(nn.Module):
    def __init__(self):
        super(Emb_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            # DEcoder
            nn.Linear(192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 192)
        )

    def forward(self, input):
        out = self.decoder(input)
        return out

class embedding_discriminator(nn.Module):
    def __init__(self,out_channel):
        super(embedding_discriminator,self).__init__()

        self.out_channel = out_channel
        self.discriminator = nn.Sequential(
            nn.Linear(192, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.out_channel)
        )

    def forward(self,x,constant):
        x=grad_reverse(x,constant)
        x=self.discriminator(x)

        return x

class DG_embedding(nn.Module):
    def __init__(self,out_channel):
        super(DG_embedding, self).__init__()
        self.out_channel = out_channel
        self.encoder = Emb_Encoder()
        self.decoder = Emb_Decoder()
        self.embedding_discriminator = embedding_discriminator(out_channel=self.out_channel)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()
        #self.ln_emb = nn.LayerNorm(normalized_shape=[192])
    def forward(self, x, labels = None,constant = None):

        encoder_x = self.encoder(x)
        rec_feature = self.decoder(encoder_x)
        rec_loss = self.mse(rec_feature, x.detach())

        output = self.embedding_discriminator(encoder_x,constant) #[batch,num_domain]
        cls_loss = self.ce(output, labels)
        prec1 = accuracy(output.detach(), labels.detach(), topk=(1,))[0]
        return cls_loss, rec_loss, prec1


##########################################################
class DG_embedding_no_encoder(nn.Module):
    def __init__(self):
        super(DG_embedding_no_encoder, self).__init__()
        self.embedding_discriminator = embedding_discriminator()
        self.ce = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels = None,constant = None):

        output = self.embedding_discriminator(x,constant) #[batch,num_domain]
        cls_loss = self.ce(output, labels)
        prec1 = accuracy(output.detach(), labels.detach(), topk=(1,))[0]
        return cls_loss, prec1

class Emb_REC(nn.Module):
    def __init__(self):
        super(Emb_REC, self).__init__()

        self.encoder = nn.Sequential(
            # Encoder
            nn.Linear(384, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 192)
        )

    def forward(self, *input):
        out = self.encoder(*input)
        return out