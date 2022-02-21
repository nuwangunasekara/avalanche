import torch
import network_Gray_ResNet
nn_model_weights_file = 'ResNet50_Gray_epoch60_BN_batchsize64_dict.pth'
abstract_model = network_Gray_ResNet.resnet50()
abstract_model.load_state_dict(torch.load(nn_model_weights_file))
print(abstract_model)