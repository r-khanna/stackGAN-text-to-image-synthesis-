#!/usr/bin/env python
# coding: utf-8


import torch 
import torch.nn as nn


############## Configurations


dim_text_embedding = 1000
dim_conditioning_var = 128
dim_noise = 100
channels_gen = 128
channels_discr = 64
upscale_factor = 2


# upsacles image by factor of 2 and also changes number of channels in upscaled image

def upscale(in_channels,out_channels):
    return nn.Sequential(
            nn.Upsample(scale_factor=upscale_factor, mode='nearest'),
            nn.Conv2d(in_channels,out_channels,3,1,1,bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))



# convolutional residual block, keeps number of channels constant

class ResBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels = channels
        self.block = nn.Sequential(
                        nn.Conv2d(channels,channels,3,1,1,bias = False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(True),
                        nn.Conv2d(channels,channels,3,1,1,bias = False),
                        nn.BatchNorm2d(channels)
                        )
        self.ReLU = nn.ReLU(True)
        
    def forward(self,x):
        residue = x
        x = self.block(x)
        x = x + residue
        x = self.ReLU(x)
        return x



class Conditional_augmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_fc_inp = dim_text_embedding
        self.dim_fc_out = dim_conditioning_var
        self.fc = nn.Linear(self.dim_fc_inp, self.dim_fc_out*2, bias= True)
        self.relu = nn.ReLU()
            
    def get_mu_logvar(self,textEmbedding):
        x = self.relu(self.fc(textEmbedding))
        
        mu = x[:,:dim_conditioning_var]
        logvar = x[:,dim_conditioning_var:]
        return mu,logvar
        
    
    def get_conditioning_variable(self,mu,logvar):
        epsilon = torch.randn(mu.size())
        std = torch.exp(0.5*logvar)
        
        return mu + epsilon*std
    
    def forward(self,textEmbedding):
        mu, logvar = self.get_mu_logvar(textEmbedding)
        return self.get_conditioning_variable(mu, logvar)


class Discriminator_logit(nn.Module):
    def __init__(self,dim_discr,dim_condVar,concat=False):
        super().__init__()
        self.dim_discr = dim_discr
        self.dim_condVar = dim_condVar
        self.concat = concat
        if concat == True:
            self.logits = nn.Sequential(
                            nn.Conv2d(dim_discr*8 + dim_condVar,dim_discr*8,3,1,1, bias = False),
                            nn.BatchNorm2d(dim_discr*8),
                            nn.LeakyReLU(.2, True),
                            nn.Conv2d(dim_discr*8, 1, kernel_size=4, stride=4),
                            nn.Sigmoid()
                        )
        
        else :
            self.logits = nn.Sequential(
                            nn.Conv2d(dim_discr*8, 1, kernel_size=4, stride=4),
                            nn.Sigmoid()
                        )
        
    def forward(self, hidden_vec, cond_aug=None):
        if self.concat is True and cond_aug is not None:
            cond_aug = cond_aug.view(-1, self.dim_condVar, 1, 1)
            cond_aug = cond_aug.repeat(1, 1, 4, 4)
            hidden_vec = torch.cat((hidden_vec,cond_aug),1)
        
        return self.logits(hidden_vec).view(-1)


class Stage1_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_noise = dim_noise
        self.dim_cond_aug = dim_conditioning_var
        self.channels_fc = channels_gen * 8
        self.cond_aug_net = Conditional_augmentation()
        
        self.fc = nn.Sequential(
                    nn.Linear(self.dim_noise + self.dim_cond_aug, self.channels_fc * 4 * 4, bias = False),
                    nn.BatchNorm1d(self.channels_fc * 4 * 4),
                    nn.ReLU(True)
                    )
        
        self.upsample = nn.Sequential(
                            upscale(self.channels_fc,self.channels_fc//2),
                            upscale(self.channels_fc//2,self.channels_fc//4),
                            upscale(self.channels_fc//4,self.channels_fc//8),
                            upscale(self.channels_fc//8,self.channels_fc//16)
                            )
        
        self.generated_image = nn.Sequential(
                                nn.Conv2d(self.channels_fc//16,3,3,1,1,bias = False),
                                nn.Tanh())
        
        
    def forward(self,noise,text_embedding):
        cond_aug = self.cond_aug_net(text_embedding)
        x = torch.cat((noise,cond_aug),1)
        
        x = self.fc(x)
        x = x.view(-1,self.channels_fc, 4, 4)
        x = self.upsample(x)
        
        image = self.generated_image(x)
        
        return image
        


class Stage1_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels_initial = channels_discr
        
        self.downsample = nn.Sequential(
                            nn.Conv2d(3, self.channels_initial, kernel_size=4, stride=2, padding=1),
                            nn.LeakyReLU(0.2,inplace=True),
            
                            nn.Conv2d(self.channels_initial , self.channels_initial*2, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.channels_initial*2),
                            nn.LeakyReLU(0.2,inplace=True),
            
                            nn.Conv2d(self.channels_initial*2, self.channels_initial*4, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.channels_initial*4),
                            nn.LeakyReLU(0.2,inplace=True),
            
                            nn.Conv2d(self.channels_initial*4, self.channels_initial*8, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.channels_initial*8),
                            nn.LeakyReLU(0.2,inplace=True),
        )
        
        self.cond_logit = Discriminator_logit(self.channels_initial,dim_conditioning_var,True)
        self.uncond_logit = Discriminator_logit(self.channels_initial,dim_conditioning_var,False)
        
    def forward(self,img):
        return self.downsample(img)


class Stage2_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample_channels = channels_gen
        self.dim_embedding = dim_conditioning_var
        self.cond_aug_net = Conditional_augmentation()
        self.Stage1_G = Stage1_Generator()
        self.downsample = nn.Sequential(
                            nn.Conv2d(3, self.downsample_channels, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
            
                            nn.Conv2d(self.downsample_channels, self.downsample_channels*2, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.downsample_channels*2),
                            nn.ReLU(inplace=True),
            
                            nn.Conv2d(self.downsample_channels*2, self.downsample_channels*4, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.downsample_channels*4),
                            nn.ReLU(inplace=True),
                        )
        self.hidden = nn.Sequential(
                        nn.Conv2d(self.downsample_channels*4 + self.dim_embedding, self.downsample_channels*4, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(self.downsample_channels*4),
                        nn.ReLU(True)
                        )
        self.residual = nn.Sequential(
                            ResBlock(self.downsample_channels*4),
                            ResBlock(self.downsample_channels*4),
                            ResBlock(self.downsample_channels*4),
                            ResBlock(self.downsample_channels*4)            
                        )
        self.upsample = nn.Sequential(
                            upscale(self.downsample_channels*4,self.downsample_channels*2),
                            upscale(self.downsample_channels*2,self.downsample_channels),
                            upscale(self.downsample_channels,self.downsample_channels//2),
                            upscale(self.downsample_channels//2,self.downsample_channels//4)
                        )
        self.image = nn.Sequential(
                        nn.Conv2d(self.downsample_channels//4, 3, 3, 1, 1, bias = False),
                        nn.Tanh()
                        )
        
    def forward(self,noise, text_embedding):
        image = self.Stage1_G(noise, text_embedding)
        image = image.detach()
        enc_img = self.downsample(image)
        
        cond_aug = self.cond_aug_net(text_embedding)
        cond_aug = cond_aug.view(-1, self.dim_embedding, 1, 1)
        cond_aug = cond_aug.repeat(1, 1, 16, 16)
        
        x = torch.cat((enc_img, cond_aug),1)
        x = self.hidden(x)
        x = self.residual(x)
        x = self.upsample(x) 
        enlarged_img = self.image(x)
        
        return enlarged_img


class Stage2_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels_initial = channels_discr
        self.downsample = nn.Sequential(
                            nn.Conv2d(3, self.channels_initial, 4, 2, 1, bias = False),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial, self.channels_initial*2, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*2),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*2, self.channels_initial*4, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*4),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*4, self.channels_initial*8, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*8),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*8, self.channels_initial*16, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*16),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*16, self.channels_initial*32, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*32),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*32, self.channels_initial*16, 3, 1, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*16),
                            nn.LeakyReLU(0.2, inplace = True),
            
                            nn.Conv2d(self.channels_initial*16, self.channels_initial*8, 3, 1, 1, bias = False),
                            nn.BatchNorm2d(self.channels_initial*8),
                            nn.LeakyReLU(0.2, inplace = True)
                            )
        
        self.cond_logit = Discriminator_logit(self.channels_initial,dim_conditioning_var,True)
        self.uncond_logit = Discriminator_logit(self.channels_initial,dim_conditioning_var,False)
        
    def forward(self,image):
        return self.downsample(image)

