import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import network_Gray_ResNet

name = 'ResNet50_Gray_epoch60_BN_batchsize64.pth'
dictname = name[:-4] + '_dict' + name[-4:]
print(dictname)

net = torch.load(name).cuda()
dictnet = net.state_dict()
torch.save(dictnet, dictname)