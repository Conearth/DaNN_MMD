#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 21:46:27 2020

@author: fj
"""
import torchvision.models as model
import os
import argparse
import sys
import math
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import numpy as np
import torch.nn.functional as F
from MMD_RBF import mmd_rbf

max_iter = 10000.0
num_class = 31
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=max_iter):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def Entropy(input_):
    input_softmax = nn.functional.softmax(input_,dim=1)
    epsilon = 1e-5
    entropy = -input_softmax *torch.log(input_softmax + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def lr_scheduler_(p:int, alpha=0.01, beta=-0.75):
    return np.power(1 + alpha * p, beta)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)
        
class DaNNmodel(nn.Module):
    def __init__(self):
        super(DaNNmodel,self).__init__()
        resnet_50 = model.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet_50.children())[:-1])
        del resnet_50
        self.bottleneck = nn.Linear(2048, 1024) # 1024 or 256
        self.class_classifier1 = nn.Linear(1024, num_class)
        self.bottleneck.apply(init_weights)
        self.class_classifier1.apply(init_weights)
    
    def forward(self,x):
        feature = self.feature_extractor(x)
        feature = feature.view(-1,2048)
        if args.target != 'dslr':
            feature = F.dropout(feature,p=0.2)
        return feature, self.class_classifier1(self.bottleneck(feature))

class AdversialDiscreminator(nn.Module):
    def __init__(self,num_class=num_class,hidden_node=1024):
        super().__init__()
#        in_feature = 2048
        self.discreminator = nn.Sequential(nn.Linear(num_class,hidden_node),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(),
                                           nn.Linear(hidden_node,hidden_node),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(),
                                           nn.Linear(hidden_node,1))
#                                           nn.ReLU(inplace=True),
#                                           nn.Dropout())
#        self.linear = nn.Linear(num_class, 1)
        
#        self.linear.apply(init_weights)          
        self.discreminator.apply(init_weights)
        self.sigmoid = nn.Sigmoid()
        self.iter_num = 0
    
    def forward(self,x):
        coeff = calc_coeff(self.iter_num)
        if self.training:
            self.iter_num += 1
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        ad_output = self.discreminator(x)
#        ad_output = self.linear(ad_output)
        return self.sigmoid(ad_output) # ad_output#,

#%% candidate discreminator
class MAdversialDiscreminator(nn.Module):
    def __init__(self,in_feature=31,hidden_node=1024):
        super().__init__()
#        in_feature = 2048
        self.discreminator = nn.Sequential(nn.Linear(in_feature,hidden_node),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(),
                                           nn.Linear(hidden_node,hidden_node),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(),
                                           nn.Linear(hidden_node, 31))
        self.discreminator.apply(init_weights)
#        self.sigmoid = nn.Sigmoid()
        self.iter_num = 0
    
    def forward(self,x):
        coeff = calc_coeff(self.iter_num)
        if self.training:
            self.iter_num += 1
        x = x * 1.0
        if x.requires_grad:
            x.register_hook(grl_hook(coeff))
        return self.discreminator(x)

#%% label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.iter_num = 0
        self.smoothing = 0.1
    def forward(self, x, target):
        smoothing = self.smoothing
#        if self.training:
#            self.iter_num += 1
#        if self.iter_num % 4000 == 0:
#            self.smoothing = 0.1 * smoothing
#            print(smoothing)
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
#%%
# loss for MDD method
def get_loss(outputs, outputs_adv):
    class_criterion = nn.CrossEntropyLoss()
    batch_size = 32
#        _, outputs, _, outputs_adv = c_net(inputs)

    target_adv = outputs.max(1)[1]
    target_adv_src = target_adv.narrow(0, 0, batch_size)
    target_adv_tgt = target_adv.narrow(0, batch_size, batch_size) # presudo-label for target smaples

    classifier_loss_adv_src = class_criterion(outputs_adv.narrow(0, 0, batch_size), target_adv_src)

    logloss_tgt = torch.log(1 - F.softmax(outputs_adv.narrow(0, batch_size, batch_size), dim = 1))
    classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)
    
    srcweight = 1
    transfer_loss = srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt
    
    return  transfer_loss 

# loss for DANN method
def log_loss(input_, batch_size=32):
    epsilon = 1e-6
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(DEVICE)
    d_loss = - dc_target * torch.log(input_ + epsilon) -(1 - dc_target) * torch.log(1 - input_ + epsilon)
#    entropy = input_ * torch.log(input_ + epsilon) -(1 - input_) * torch.log(1 - input_ + epsilon)
    return torch.sum(d_loss)/len(d_loss)


def test(model,dataloader):
    total_loss, correct, acc = 0,0,0
    model.eval()
    length = len(dataloader.dataset) 
    for img,label in dataloader:
        img,label = img.to(DEVICE),label.long().to(DEVICE)
        with torch.no_grad():
            _, output = model(img)
            loss = cls_criterion(output, label)
        
        pred = torch.max(output,1)[1]
        correct += torch.sum(pred == label)
        total_loss += float(loss.item()) * img.shape[0]
    acc = correct.double() / length
    return total_loss/length, acc

#%%
parser = argparse.ArgumentParser(description='domain adaptation with cosine similarity')
parser.add_argument('-batch_size', '-b', type=int, help='batch size', default=32)
parser.add_argument('-cuda', '-g', type=int, help='cuda id', default=0)
parser.add_argument('-Epoch', '-e', type=int, default=500)
parser.add_argument('-hidden_node', '-hn', type=int, default=1024)

# learning rate
parser.add_argument('-learning_rate', '-lr', type=float, help='learning rate', default=3e-4)
parser.add_argument('-warm_up_epochs', '-w', type=int, help='warm up epoch for Cosine Schedule', default=5)
parser.add_argument('-momentum', '-m', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('-weight_decay', '-wd', type=float, default=5e-5,
                    help='weight decay for SGD Momentum')
# dataset
parser.add_argument('-train_path', '-path', type=str, default='~/Desktop/PyTorch/dataset/OFFICE31/',
                    help='source and target datasets\'s path')
parser.add_argument('-source', '-src', type=str, default='amazon')
parser.add_argument('-target', '-tar', type=str, default='dslr')

args = parser.parse_args()

#%% record args
save_file_name = os.listdir('./output')
start_name = 'outlog_'+args.source+'_'+args.target
number = 0
for name in save_file_name:
    if name.startswith(start_name):
        number += 1
with open('./output/'+start_name+'_'+str(number)+'.txt', 'w') as f:
    f.write(str(args))
print(str(args))

#%%
batch_size = 32
lr = args.learning_rate #2e-4
n_epoch = args.Epoch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pre_trained = False
T = 1.0 # temperature

# network
dann = DaNNmodel().to(DEVICE)
AD = AdversialDiscreminator().to(DEVICE)
MAD = MAdversialDiscreminator().to(DEVICE)

seed = 2021
print(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



if pre_trained:
    dann_dict = torch.load('dann_webcam_dslr.pth')
    dann.load_state_dict(dann_dict[0], strict=False)
    AD.load_state_dict(dann_dict[1],strict=False)
    del dann_dict
# Define Loss
#cls_criterion = nn.NLLLoss(reduction='none').to(DEVICE)
cls_criterion  = LabelSmoothingCrossEntropy().to(DEVICE)
#cls_criterion  = nn.CrossEntropyLoss().to(DEVICE)
dmn_criterion = nn.CrossEntropyLoss(reduction='none').to(DEVICE)


# setup optimizer
optimizer = optim.SGD([{'params': dann.feature_extractor.parameters(),'lr': lr},
                    {'params': AD.parameters(), 'lr': lr*10},
                    {'params': dann.class_classifier1.parameters(), 'lr': lr*10},
                    {'params': dann.bottleneck.parameters(), 'lr': lr*10}],
            lr=lr, momentum=0.9, weight_decay=0.0005)

# Learning rate update schedulers
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scheduler_)
#warm_up_epochs = 5
#warm_up_with_cosine_lr = lambda epoch:  (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
#    else  0.5 * ( math.cos((epoch - warm_up_epochs) /(n_epoch - warm_up_epochs) * math.pi) + 1)
#    
#lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,warm_up_with_cosine_lr)
#%%
# data loader
# Image transformations
data_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

# Training data loader
dataset_name_s = args.source
path0 = '~/Desktop/PyTorch/dataset/OfficeHome/% s/'
path1 = '~/Desktop/PyTorch/dataset/OFFICE31/% s/'
path = path1
amazon_dataset = datasets.ImageFolder(root= path % dataset_name_s,
                                           transform=data_transform)
dataloader_source = torch.utils.data.DataLoader(amazon_dataset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=8)

dataset_name_t = args.target
webcam_dataset = datasets.ImageFolder(root= path % dataset_name_t,
                                           transform=data_transform)
dataloader_target = torch.utils.data.DataLoader(webcam_dataset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=8)

test_dataset = datasets.ImageFolder(root= path % dataset_name_t,
                                           transform=test_transform)
dataloader_test = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size, shuffle=True,
                                             num_workers=8)
#%%
acc_list, loss_list = [], []
prev = time.time()
iter_num = -1
for epoch in range(n_epoch):
    len_dataloader = min(len(dataloader_source),len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)
    
    for i in range(len(dataloader_source)):
#        iter_num += 1
        # calulate cls_loss
        s_img, s_label = data_source_iter.next()
        s_img, s_label = s_img.to(DEVICE), s_label.long().to(DEVICE)
        domain_label_s = torch.zeros(s_img.shape[0]).long().to(DEVICE)
        
        t_img,_ = data_target_iter.next()
        t_img = t_img.to(DEVICE)
        domain_label_t = torch.ones(t_img.shape[0]).long().to(DEVICE)
        if t_img.shape[0] < batch_size or s_img.shape[0] < batch_size:
            break
        
        dann.train()
        AD.train()
        
        feature_s, cls_output = dann(s_img)
        
        feature_t,cls_output2 = dann(t_img)
        
        # ----------------
        # train G C and D
        # ----------------
        
        # entropy
        domain_feature = torch.cat((feature_s,feature_t))
        domain_output = torch.cat((cls_output,cls_output2))
        domain_label = torch.cat((domain_label_s,domain_label_t))
        
        entropy = Entropy(domain_output/T)
        exp_entropy = torch.exp(-entropy)
        weight = torch.cat((exp_entropy[:batch_size]/torch.sum(exp_entropy[:batch_size]),
                            exp_entropy[batch_size:]/torch.sum(exp_entropy[batch_size:])))
        
        # cls loss
#        cls_loss = torch.sum(cls_criterion(cls_output, s_label)*weight)
#        cls_loss = cls_criterion(F.log_softmax(cls_output/T,dim=1), s_label).mean()
        cls_loss = cls_criterion(cls_output/T,s_label)
        
        # dmn loss
        ad_output = AD(F.softmax(domain_output/T, dim=1))
#        dmn_loss = dmn_criterion(ad_output,domain_label).mean() # num_class of domain = 2
        dmn_loss = log_loss(ad_output) 
        
        
        cls_entropy = torch.sum(entropy)/np.log(num_class)/len(entropy)
        
        entropy = cls_entropy
        
        # mean
        mean = mmd_rbf(cls_output/T, cls_output2/T)
        
        # sign
        sign = ad_output.mean() ** 2 
        
        if epoch < 100:
            alpha = 0
        else:
            alpha = 1
        # total loss
        loss =  cls_loss + dmn_loss + mean + alpha * entropy# + sign 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [cls_loss: %f] [dmn_loss0: %f] [entropy: %f] [mmd_rbf: %f] [sign: %f]"
            % (
                    epoch,n_epoch, cls_loss.item(), dmn_loss.item(),  entropy, mean.item(), sign.item()
            )
        )
        sys.stdout.flush()
    
    lr_scheduler.step()
    
    
    if epoch % 5 == 0:
    #    s_loss, s_acc = test(dann,dataloader_source)
        t_loss, t_acc = test(dann,dataloader_test)
    #    print('\nnepoch: {},s_loss: {:.6f}, s_acc: {:.4f}'.format(epoch, s_loss, s_acc))
        print('\nepoch: {},t_loss: {:.6f}, t_acc: {:.4f}'.format(epoch, t_loss, t_acc))
#        print('time:',time.time()-prev, optimizer.state_dict()['param_groups'][0]['lr'])
#        prev = time.time()
        with open('./output/'+start_name+'_'+str(number)+'.txt', 'a') as f:
            f.write("\r[Epoch %d/%d] [iter: %d/%d] [cls_loss: %f] [dmn_loss: %f]"
            % (epoch, args.Epoch, i, len_dataloader, cls_loss.item(), dmn_loss.item()))
#            f.write('\nepoch: {},tr_loss: {:.6f}, tr_acc: {:.4f}'.format(epoch, tr_loss, tr_acc))
#            f.write('\nepoch: {},v_loss: {:.6f}, v_acc: {:.4f}'.format(epoch, v_loss, v_acc))
            f.write('\nepoch: {},t_loss: {:.6f}, t_acc: {:.4f}'.format(epoch, t_loss, t_acc))
            f.write('; learning rate: '+str(optimizer.state_dict()['param_groups'][0]['lr']))


