import os.path
import random
import sys
import threading
from collections import OrderedDict

import numpy as np
from copy import deepcopy
from math import log

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from joblib import dump, load

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models.quantization as models
# from torchsummary import summary

# import network_Gray_ResNet
from avalanche.models import network_Gray_ResNet

Sigmoid = 1
Tanh = 2
Relu = 3
LeakyRelu = 4

default_mlp_hidden_layers = [{'neurons': 2 ** 10, 'nonlinearity': Relu}]

OP_TYPE_SGD = 'SGD'
OP_TYPE_SGD_NC = 'SGD-NC'
OP_TYPE_ADAGRAD = 'Adagrad'
OP_TYPE_ADAGRAD_NC = 'Adagrad-NC'
OP_TYPE_RMSPROP = 'RMSprop'
OP_TYPE_RMSPROP_NC = 'RMSprop-NC'
OP_TYPE_ADADELTA = 'Adadelta'
OP_TYPE_ADADELTA_NC = 'Adadelta-NC'
OP_TYPE_ADAM = 'Adam'
OP_TYPE_ADAM_NC = 'Adam-NC'
OP_TYPE_ADAM_AMSG = 'Adam-AMSG'
OP_TYPE_ADAM_AMSG_NC = 'Adam-AMSG-NC'

NETWORK_TYPE_MLP = 0
NETWORK_TYPE_CNN = 1

PREDICT_METHOD_ONE_CLASS = 0
PREDICT_METHOD_MAJORITY_VOTE = 1
PREDICT_METHOD_RANDOM = 2
PREDICT_METHOD_TASK_ID_KNOWN = 3
PREDICT_METHOD_NW_CONFIDENCE = 4
PREDICT_METHOD_NAIVE_BAYES = 5
PREDICT_METHOD_HT = 6

DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END = -1
WITH_ACCUMULATED_INSTANCES = 0
WITH_ACCUMULATED_LEARNED_FEATURES = 1
WITH_ACCUMULATED_STATIC_FEATURES = 2

NO_OF_CHANNELS = 3

POOL_FROZEN = 0
POOL_TRAINING = 1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def print_summary(model, input_dims: tuple):
#     i_dims = input_dims[1:len(input_dims)] if len(input_dims) == 4 else input_dims
#     summary(model, tuple(i_dims))

def create_static_feature_extractor(
        device=None,
        use_single_channel_fx=False,
        single_channel_fx_path='/Scratch/repository/ng98/CL_SSD_Arch/CL/1_channel_fx/ResNet50_Gray_epoch60_BN_batchsize64_dict.pth',
        quantize=True):
    layers = OrderedDict()

    if use_single_channel_fx:
        # conv1 = nn.Conv2d(1, 64, kernel_size=original_model.conv1.kernel_size,
        #                   # stride=1,
        #                   # stride=original_model.conv1.stride,
        #                   #       padding=original_model.conv1.padding,
        #                   bias=False)
        # conv1.weight.data = original_model.state_dict()['conv1.weight'].mean(axis=1).view(64,1,original_model.conv1.kernel_size[0],original_model.conv1.kernel_size[1])
        original_model = network_Gray_ResNet.resnet50()
        original_model.load_state_dict(torch.load(single_channel_fx_path))

        # original_model = torch.load(single_channel_fx_path)
        # original_model.eval()
    else: # 3_channel_data
        # https://keras.io/api/applications/
        # https://pytorch.org/hub/
        # https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html
        # https://pytorch.org/hub/pytorch_vision_resnet/
        original_model = models.resnet18(pretrained=True, progress=True, quantize=quantize)

    # you dont need this if the model is quantized. But we are doing it any way to be on the safe side.
    for param in original_model.parameters():
        param.requires_grad = False

    if use_single_channel_fx:
        if quantize:
            # need to implement
            # layers['quant'] = original_model.quant  # Quantize the input
            pass
        layers['begin'] = original_model.begin
        layers['layer1'] = original_model.layer1
        layers['layer2'] = original_model.layer2
        layers['layer3'] = original_model.layer3
        layers['layer4'] = original_model.layer4
        layers['last'] = original_model.last
        layers['avgpool'] = original_model.avgpool
        if quantize:
            # need to implement
            # layers['dequant'] = original_model.dequant  # Dequantize the output
            pass
    else: # 3_channel_data
        # Step 1. Isolate the feature extractor.
        if quantize:
            layers['quant'] = original_model.quant  # Quantize the input
            device = 'cpu'
        layers['conv1'] = original_model.conv1
        layers['bn1'] = original_model.bn1
        layers['relu'] = original_model.relu
        layers['maxpool'] = original_model.maxpool
        layers['layer1'] = original_model.layer1
        layers['layer2'] = original_model.layer2
        layers['layer3'] = original_model.layer3
        layers['layer4'] = original_model.layer4
        layers['avgpool'] = original_model.avgpool
        if quantize:
            layers['dequant'] = original_model.dequant  # Dequantize the output
    model_fe = nn.Sequential(layers)

    # # Step 2. Create a new "head"
    # new_head = nn.Sequential(
    #     nn.Dropout(p=0.5),
    #     nn.Linear(num_ftrs, 2),
    # )
    #
    # Step 3. Combine, and don't forget the quant stubs.
    new_model = nn.Sequential(
        model_fe,
        nn.Flatten(1),
        # new_head,
    )

    for param in new_model.parameters():
        param.requires_grad = False

    # if single_channel_data:

        # https: // pytorch.org / tutorials / advanced / static_quantization_tutorial.html  # post-training-static-quantization
        # backend = "fbgemm"
        # new_model.qconfig = torch.quantization.get_default_qconfig(backend)
        # torch.backends.quantized.engine = backend
        # new_model.qconfig = torch.quantization.default_qconfig
        # new_model = torch.quantization.prepare(new_model, inplace=False)
        # new_model = torch.quantization.convert(new_model, inplace=False)

    return device, new_model.to(torch.device(device))


def train_one_class_classifier(features, one_class_detector, train_logistic_regression, logistic_regression, scaler=None):
    xx = features.cpu().numpy()
    one_class_detector.partial_fit(xx)
    yy = one_class_detector.predict(xx)
    if train_logistic_regression:
        yy[yy == -1] = 0  # set outlier to be class 0
        df_scores = one_class_detector.decision_function(xx)
        if yy.sum() == 0:  # only class 0 available
            yy[df_scores.argmax()] = 1
        if yy.sum() == yy.shape[0]:  # only class 1 available
            yy[df_scores.argmin()] = 0
        df_scores = df_scores.reshape(-1, 1)
        if scaler is not None:
            scaler.partial_fit(df_scores)
            df_scores = scaler.transform(df_scores)
        logistic_regression.partial_fit(df_scores, yy, classes=np.array([0, 1]))


def train_nb(features, nb, task_id):
    x = features.detach().cpu().numpy()
    y = np.empty((x.shape[0],), dtype=np.int64)
    y.fill(task_id)
    nb.partial_fit(x, y)


preprocessor = transforms.Compose([
    # https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/9
    # expand channels
    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)) if len(x.shape) < 4 else NoneTransform(),
    transforms.Lambda(lambda x: x.repeat(1, NO_OF_CHANNELS, 1, 1)), # repeat channel 1, 3 times
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.Pad(0, fill=3),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

repeat_channel_1 = transforms.Compose([
    # https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/9
    # expand channels
    transforms.Lambda(lambda x: x.repeat(1, NO_OF_CHANNELS, 1, 1)), # repeat channel 1, 3 times
])


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10, num_channels=NO_OF_CHANNELS):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_channels + 1, 32, kernel_size=3, stride=1, padding=1) if num_channels == 0 else nn.Conv2d(
                num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # nn.Conv2d(64, 64, kernel_size=3, padding=0),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=0.25),

            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            # nn.Dropout(p=0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x, learned_features=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        if learned_features:
            # pass learned features from last layer
            if learned_features[0] is None:
                learned_features[0] = x.detach()

        x = self.classifier(x)
        return x


class CNN4(nn.Module):
    def __init__(self, num_classes=10, num_channels=0):
        super(CNN4, self).__init__()

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_channels + 1, 32, kernel_size=3, stride=1, padding=1) if num_channels == 0 else nn.Conv2d(
                num_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=0.25),

            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            # nn.Dropout(p=0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x, learned_features=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        if learned_features:
            # pass learned features from last layer
            if learned_features[0] is None:
                learned_features[0] = x.detach()

        x = self.classifier(x)
        return x


# class PyNet(nn.Module):
#     def __init__(self, hidden_layers: list = None, num_classes: int = None, input_dimensions=None):
#         super(PyNet, self).__init__()
#
#         if hidden_layers is None:
#             return
#         linear = []
#         self.f = []
#
#         # Add hidden layers
#         for h in range(0, len(hidden_layers), 1):
#             if h == 0:  # first hidden layer
#                 in_d = input_dimensions
#                 nonlinearity = Relu
#             else:
#                 in_d = hidden_layers[h - 1]['neurons']
#                 nonlinearity = hidden_layers[h]['nonlinearity']
#             out_d = hidden_layers[h]['neurons']
#             linear.append(nn.Linear(in_d, out_d))
#             if nonlinearity == Tanh:
#                 self.f.append(nn.Tanh())
#             elif nonlinearity == Sigmoid:
#                 self.f.append(nn.Sigmoid())
#             elif nonlinearity == Relu:
#                 self.f.append(nn.ReLU())
#             elif nonlinearity == LeakyRelu:
#                 self.f.append(nn.LeakyReLU())
#             else:
#                 pass
#
#         # output layer
#         linear.append(nn.Linear(hidden_layers[len(hidden_layers) - 1]['neurons'], num_classes))
#         # self.f.append(nn.ReLU())
#
#         self.linear = nn.ModuleList(linear)
#
#     def forward(self, X, learned_features=None):
#         if learned_features:
#             # pass learned features from last layer
#             if learned_features[0] is None:
#                 learned_features[0] = X.detach()
#
#         x = X
#         for i, l in enumerate(self.linear):
#             if i == len(self.linear) - 1:
#                 x = l(x)
#             else:
#                 x = self.f[i](l(x))
#         return x


def init_one_class_detector():
    return linear_model.SGDOneClassSVM(random_state=0, shuffle=False, max_iter=1, warm_start=True)


def init_scaler():
    return StandardScaler()


def init_logistic_regression():
    return SGDClassifier(loss='log', random_state=0, max_iter=1, shuffle=False, warm_start=True)


class ANN:
    def __init__(self,
                 id,
                 learning_rate=0.03,
                 network_type=None,
                 hidden_layers_for_mlp=default_mlp_hidden_layers,
                 num_classes=None,
                 device='cpu',
                 optimizer_type=OP_TYPE_SGD,
                 loss_f=nn.CrossEntropyLoss(),
                 loss_estimator_delta=1e-3,
                 task_detector_type=PREDICT_METHOD_ONE_CLASS,
                 back_prop_skip_loss_threshold=0.6,
                 train_task_predictor_at_the_end=DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END,
                 train_nn_using_ex_static_f=False):
        # configuration variables (which has the same name as init parameters)
        self.id = id
        self.frozen_id = None
        self.model_name = None
        self.learning_rate = learning_rate
        self.network_type = network_type
        # self.hidden_layers = copy.deepcopy(hidden_layers)
        self.hidden_layers_for_MLP = deepcopy(hidden_layers_for_mlp)
        self.num_classes = num_classes
        self.device = device
        self.optimizer_type = optimizer_type
        self.loss_f = loss_f
        self.loss_estimator_delta = loss_estimator_delta
        self.back_prop_skip_loss_threshold = back_prop_skip_loss_threshold
        self.task_detector_type = task_detector_type
        self.train_task_predictor_at_the_end = train_task_predictor_at_the_end
        # self.prediction_pool = prediction_pool
        self.current_loss = None
        self.train_nn_using_ex_static_f = train_nn_using_ex_static_f

        # status variables
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.samples_seen_at_train = 0
        self.trained_count = 0
        self.chosen_for_test = 0
        self.chosen_after_train = 0
        self.loss_estimator: BaseDriftDetector = None
        self.old_loss_estimator: BaseDriftDetector = None
        self.accumulated_loss = 0
        self.last_loss = 0
        self.one_class_detector = None
        self.one_class_detector_fit_called = False
        self.scaler = None
        self.logistic_regression = None
        self.correct_class_predicted = 0
        # self.input_dimensions = 0
        self.x_shape = None
        self.task_detected = False
        self.seen_task_ids_train = {}
        self.correctly_predicted_task_ids_test = {}
        self.correctly_predicted_task_ids_test_at_last = {}
        self.correctly_predicted_task_ids_probas_test_at_last = {}
        self.accumulated_features = None
        self.outputs = None
        self.init_values()

    def init_values(self):
        # init status variables
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.samples_seen_at_train = 0
        self.trained_count = 0
        self.chosen_for_test = 0
        self.chosen_after_train = 0
        self.loss_estimator = ADWIN(delta=self.loss_estimator_delta)
        self.accumulated_loss = 0
        self.one_class_detector = init_one_class_detector()
        self.scaler = init_scaler()
        self.logistic_regression = init_logistic_regression()
        self.correct_class_predicted = 0
        # self.input_dimensions = 0
        self.x_shape = None
        self.task_detected = False
        self.seen_task_ids_train = {}
        self.correctly_predicted_task_ids_test = {}
        self.correctly_predicted_task_ids_test_at_last = {}
        self.correctly_predicted_task_ids_probas_test_at_last = {}
        self.accumulated_features = None
        self.outputs = None

        if self.hidden_layers_for_MLP is None:
            pass
        elif isinstance(self.hidden_layers_for_MLP, nn.Module):
            # assumes input dimension is set properly in the network structure
            self.net = deepcopy(self.hidden_layers_for_MLP)
            self.initialize_net_para()
        elif isinstance(self.hidden_layers_for_MLP, list):
            if self.hidden_layers_for_MLP[0]['neurons'] is None or self.hidden_layers_for_MLP[0][
                'nonlinearity'] is None:
                print('Unknown hidden layer format is passed in: {}'.format(self.hidden_layers_for_MLP))
                print('Expected format :{}'.format(default_mlp_hidden_layers))
                exit(1)
        self.model_name = '{}_{}_{:05f}'.format(
            'CNN' if self.network_type == NETWORK_TYPE_CNN else 'MLP_L1_' + str(
                log(self.hidden_layers_for_MLP[0]['neurons'], 2) // 1),
            self.optimizer_type,
            self.learning_rate)

    def init_optimizer(self):
        self.net.to(self.device)
        if self.optimizer_type == OP_TYPE_ADAGRAD or self.optimizer_type == OP_TYPE_ADAGRAD_NC:
            self.optimizer = optim.Adagrad(self.net.parameters(), lr=self.learning_rate, lr_decay=0, weight_decay=0,
                                           initial_accumulator_value=0, eps=1e-10)
        elif self.optimizer_type == OP_TYPE_ADADELTA or self.optimizer_type == OP_TYPE_ADADELTA_NC:
            self.optimizer = optim.Adadelta(self.net.parameters(), lr=self.learning_rate, eps=1e-10)
        elif self.optimizer_type == OP_TYPE_RMSPROP or self.optimizer_type == OP_TYPE_RMSPROP_NC:
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.learning_rate, alpha=0.99, weight_decay=0,
                                           eps=1e-10)
        elif self.optimizer_type == OP_TYPE_SGD or self.optimizer_type == OP_TYPE_SGD_NC:
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == OP_TYPE_ADAM or self.optimizer_type == OP_TYPE_ADAM_NC:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                        weight_decay=0, amsgrad=False)
        elif self.optimizer_type == OP_TYPE_ADAM_AMSG or self.optimizer_type == OP_TYPE_ADAM_AMSG_NC:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                        weight_decay=0, amsgrad=True)
        else:
            print('Invalid optimizer type = {}'.format(self.optimizer_type))

    def initialize_net_para(self):
        self.init_optimizer()
        print('Network configuration:\n'
              '{}\n'
              '======================================='.format(self))

    def initialize_network(self):
        if self.network_type == NETWORK_TYPE_CNN:
            if len(self.x_shape) == 2:
                number_of_channels = 0
                # number_of_channels = self.x_shape[1]
            else:
                number_of_channels = self.x_shape[1]
            self.net = SimpleCNN(num_classes=self.num_classes, num_channels=number_of_channels)
            # print_summary(self.net, self.x_shape)
            print('Number of parameters: {}'.format(count_parameters(self.net)))
        else:
            pass
            # self.net = PyNet(hidden_layers=self.hidden_layers_for_MLP, num_classes=self.num_classes,
            #                  input_dimensions=self.x_shape[0])
        self.initialize_net_para()

    def train_one_class_classifier(self, features, train_logistic_regression):
        train_one_class_classifier(features,
                                   self.one_class_detector,
                                   train_logistic_regression,
                                   self.logistic_regression,
                                   self.scaler)
        self.one_class_detector_fit_called = True

    def train_net(self, x, y, c, r, true_task_id, use_instances_for_task_detector_training,
                  use_one_class_probas, static_features=None, train_nn_using_ex_static_f=False):
        if self.train_nn_using_ex_static_f and static_features is not None:
            # xx = repeat_channel_1(static_features.view(static_features.shape[0], 64, -1)[:, None, :, :])
            xx = static_features.view(static_features.shape[0], 1, 64, -1)
        else:
            xx = x
        if self.net is None:
            # self.input_dimensions = c
            self.x_shape = deepcopy(xx.shape)
            self.initialize_network()

        self.samples_seen_at_train += r

        if self.seen_task_ids_train.get(true_task_id) is None:
            self.seen_task_ids_train[true_task_id] = r
        else:
            self.seen_task_ids_train[true_task_id] += r

        xx = xx.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()  # zero the gradient buffers
        # forward propagation
        learned_features = None
        use_static_features = False
        if use_instances_for_task_detector_training:
            if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                if self.train_task_predictor_at_the_end == DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END:
                    if static_features is not None:
                        use_static_features = True
                    else:
                        learned_features = [None]  # get learned features
                elif self.train_task_predictor_at_the_end == WITH_ACCUMULATED_LEARNED_FEATURES:
                    learned_features = [None]  # get learned features

        outputs = self.net(xx, learned_features=learned_features)  # trains nn and fills learned_features if necessary

        if use_instances_for_task_detector_training:
            if self.train_task_predictor_at_the_end == DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END:
                if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                    if use_static_features:
                        learned_features = [None]
                        learned_features[0] = static_features
                    self.train_one_class_classifier(learned_features[0].cpu(), use_one_class_probas)
            elif self.train_task_predictor_at_the_end == WITH_ACCUMULATED_LEARNED_FEATURES:
                if self.accumulated_features is None:
                    self.accumulated_features = learned_features[0].cpu()
                else:
                    self.accumulated_features = torch.vstack((self.accumulated_features, learned_features[0].cpu()))

        # backward propagation
        # print(self.net.linear[0].weight.data)
        self.loss = self.loss_f(outputs, y)
        self.current_loss = self.loss.item()
        self.outputs = outputs.detach()
        # if self.prediction_pool == POOL_FROZEN:
        #     self.update_loss_estimator(copy_old=False)
        #     self.call_backprop()
        #     self.reset_loss_and_bp_buffers()

        return outputs

    def update_loss_estimator(self, copy_old=False):
        if copy_old:
            self.old_loss_estimator = deepcopy(self.loss_estimator)
        previous_estimated_loss = self.loss_estimator.estimation
        self.loss_estimator.add_element(self.current_loss)
        self.accumulated_loss += self.current_loss

        self.task_detected = False
        if self.loss_estimator.detected_change() and self.loss_estimator.estimation > previous_estimated_loss:
            print('NEW TASK detected by {}'.format(self.model_name))
            self.task_detected = True
        else:
            self.old_loss_estimator = None

    def call_backprop(self):
        if self.loss.item() > self.back_prop_skip_loss_threshold:
            self.loss.backward()
            self.optimizer.step()  # Does the update
            self.trained_count += 1

    def reset_loss_and_bp_buffers(self):
        self.optimizer.zero_grad()
        self.loss = None
        # self.outputs = None
        self.old_loss_estimator = None

    def get_votes(self, x, x_flatten, static_features=None):
        if self.network_type == NETWORK_TYPE_CNN:
            if self.train_nn_using_ex_static_f and static_features is not None:
                # xx = repeat_channel_1(static_features.view(static_features.shape[0], 64, -1)[:, None, :, :])
                xx = static_features.view(static_features.shape[0], 1, 64, -1)
            else:
                xx = x
        else:
            xx = x_flatten
        return self.net(xx).unsqueeze(0)

    def reset(self):
        # configuration variables (which has the same name as init parameters) should be copied by the caller function
        self.init_values()
        return self

    def get_loss_estimation(self):
        return self.loss_estimator.estimation

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


def flatten_dimensions(mb_x):
    c = 1
    # flatten all dimensions
    for i in range(1, len(mb_x.shape), 1):
        c *= mb_x.shape[i]
    # c = strategy.mb_x.shape[1] * strategy.mb_x.shape[2] * strategy.mb_x.shape[3]
    x = mb_x.contiguous()
    return c, x.view(x.size(0), c)


def net_train(net: ANN, x: np.ndarray, r, c, y: np.ndarray, true_task_id,
              use_instances_for_task_detector_training,
              use_one_class_probas,
              static_features,
              train_nn_using_ex_static_f):
    net.train_net(x, y, c, r, true_task_id,use_instances_for_task_detector_training,
                  use_one_class_probas,
                  static_features=static_features,
                  train_nn_using_ex_static_f=train_nn_using_ex_static_f)

def update_loss_estimator(net: ANN, copy_old):
    net.update_loss_estimator(copy_old)

def call_backprop(net: ANN):
    net.call_backprop()

def reset_loss_and_bp_buffers(net: ANN):
    net.reset_loss_and_bp_buffers()


def save_model(best_model: ANN, abstract_model_file_name, nn_model_file_name, preserve_net=False):
    # set unwanted attributes to None

    net = best_model.net
    best_model.net = None

    # Save abstract model with one class classifier
    # https://scikit-learn.org/stable/modules/model_persistence.html
    # save model at self.model_dump_dir with name model_id_task_id one_class
    # Save one_class_classifier
    dump(best_model, abstract_model_file_name)

    # Save NN model
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
    torch.save(net.state_dict(), nn_model_file_name)
    # load model and one_class_classifier and  add to frozen nets

    if preserve_net:
        best_model.net = net


def load_model(abstract_model_file_name, nn_model_file_name):
    abstract_model: ANN = load(abstract_model_file_name)
    abstract_model.initialize_network()
    abstract_model.net.load_state_dict(torch.load(nn_model_file_name))
    abstract_model.net.eval()
    return abstract_model


class MultiMLP(nn.Module):
    def __init__(self,
                 num_classes=None,
                 use_threads=False,
                 stats_file=sys.stdout,
                 # number_of_mlps_to_train=10,
                 number_of_instances_to_train_using_all_mlps_at_start=1000,
                 loss_estimator_delta=1e-3,
                 back_prop_skip_loss_threshold=0.6,
                 stats_print_frequency=0,
                 nn_pool_type='30MLP',
                 predict_method=PREDICT_METHOD_ONE_CLASS,
                 train_task_predictor_at_the_end=DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END,
                 device='cpu',
                 model_dump_dir=None,
                 reset_training_pool=True,
                 use_one_class_probas=False,
                 vote_weight_bias=1e-3,
                 use_weights_from_task_detectors=False,
                 auto_detect_tasks=False,
                 n_experiences=None,
                 use_static_f_ex=False,
                 train_nn_using_ex_static_f=True,
                 train_only_the_best_nn=False,
                 use_1_channel_pretrained_for_1_channel=False,
                 use_quantized=False,
                 prediction_pool=None,
                 train_pool_max=None,
                 random_train_frozen_if_best=False):
        super().__init__()

        # configuration variables (which has the same name as init parameters)
        self.num_classes = num_classes
        self.use_threads = use_threads
        self.stats_file = stats_file
        # self.number_of_mlps_to_train = number_of_mlps_to_train
        self.number_of_instances_to_train_using_all_mlps_at_start = number_of_instances_to_train_using_all_mlps_at_start
        self.loss_estimator_delta = loss_estimator_delta
        self.stats_print_frequency = stats_print_frequency
        self.back_prop_skip_loss_threshold = back_prop_skip_loss_threshold
        self.nn_pool_type = nn_pool_type
        self.task_detector_type = predict_method
        self.train_task_predictor_at_the_end = train_task_predictor_at_the_end
        self.device = device
        self.model_dump_dir = model_dump_dir
        self.reset_training_pool = reset_training_pool
        self.use_one_class_probas = use_one_class_probas
        self.vote_weight_bias = vote_weight_bias
        self.use_weights_from_task_detectors = use_weights_from_task_detectors
        self.auto_detect_tasks = auto_detect_tasks
        self.n_experiences = n_experiences
        self.use_static_f_ex = use_static_f_ex
        self.one_class_stats_file = sys.stdout
        self.one_class_stats_header_printed = False
        self.nb_stats_file = sys.stdout
        self.train_nn_using_ex_static_f = train_nn_using_ex_static_f
        self.train_only_the_best_nn = train_only_the_best_nn
        self.use_1_channel_pretrained_for_1_channel = use_1_channel_pretrained_for_1_channel
        self.use_quantized = use_quantized
        self.prediction_pool = prediction_pool
        self.train_pool_max = train_pool_max if train_pool_max is not None and train_pool_max > 6 else train_pool_max
        self.random_train_frozen_if_best = random_train_frozen_if_best

        # status variables
        self.train_nets = []  # type: List[ANN]
        self.frozen_nets = []  # type: List[ANN]
        self.frozen_net_module_paths = []
        self.heading_printed = False
        self.samples_seen_for_train_after_dd = 0
        self.total_samples_seen_for_train = 0
        self.samples_per_each_task_at_train = {}
        self.samples_seen_for_test = 0
        self.correct_network_selected_count = 0
        self.correct_network_selected_count_at_last = 0
        self.correct_class_predicted = 0
        self.call_predict = False
        self.mb_yy = None
        self.mb_task_id = None
        self.training_exp = 0
        self.available_nn_id = 0
        self.detected_task_id = 0
        self.accumulated_x_or_features = [None]
        # self.learned_tasks = [0]
        self.instances_per_task_at_last = {}
        self.f_ex = None
        self.f_ex_device = None
        self.nb_or_ht = NaiveBayes() if self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES else HoeffdingTreeClassifier() if self.task_detector_type == PREDICT_METHOD_HT else None
        self.nb_preds = None

        self.init_values()

    def init_values(self):
        # init status variables

        self.heading_printed = False
        self.nb_stats_file = sys.stdout if self.stats_file is sys.stdout else \
            self.stats_file.replace('.csv', '_NB') if self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES else self.stats_file.replace('.csv', '_HT') if self.task_detector_type == PREDICT_METHOD_HT else sys.stdout
        self.one_class_stats_file = sys.stdout if self.stats_file is sys.stdout else \
            open(self.stats_file.replace('.csv', '_OC.csv'), 'w') if self.task_detector_type == PREDICT_METHOD_ONE_CLASS else sys.stdout
        self.stats_file = sys.stdout if self.stats_file is sys.stdout else open(self.stats_file, 'w')

        self.create_nn_pool()

        print(self)

    def create_nn_pool(self):
        if self.nn_pool_type == '30MLP':
            neurons_in_log2_start_include = 8
            neurons_in_log2_stop_exclude = 11

            lr_denominator_in_log10_start_include = 1
            lr_denominator_in_log10_stop_exclude = 6

            optimizer_types = (OP_TYPE_SGD_NC, OP_TYPE_ADAM_NC)
        elif self.nn_pool_type == '10MLP':
            neurons_in_log2_start_include = 10
            neurons_in_log2_stop_exclude = 11

            lr_denominator_in_log10_start_include = 1
            lr_denominator_in_log10_stop_exclude = 6

            optimizer_types = (OP_TYPE_SGD_NC, OP_TYPE_ADAM_NC)
        elif self.nn_pool_type == '6MLP':
            neurons_in_log2_start_include = 10
            neurons_in_log2_stop_exclude = 11

            lr_denominator_in_log10_start_include = 1
            lr_denominator_in_log10_stop_exclude = 7

            optimizer_types = ([OP_TYPE_ADAM_NC])
        elif self.nn_pool_type == '6CNN':
            neurons_in_log2_start_include = 10
            neurons_in_log2_stop_exclude = 11

            lr_denominator_in_log10_start_include = 1
            lr_denominator_in_log10_stop_exclude = 7

            optimizer_types = ([OP_TYPE_ADAM_NC])

        for number_of_neurons_in_log2 in range(neurons_in_log2_start_include, neurons_in_log2_stop_exclude):
            for lr_denominator_in_log10 in range(lr_denominator_in_log10_start_include,
                                                 lr_denominator_in_log10_stop_exclude):
                for optimizer_type in optimizer_types:
                    hidden_layers_for_mlp = None
                    network_type = NETWORK_TYPE_CNN
                    if self.nn_pool_type != '6CNN':
                        hidden_layers_for_mlp = [{'neurons': 2 ** number_of_neurons_in_log2, 'nonlinearity': Relu}]
                        network_type = NETWORK_TYPE_MLP
                    tmp_ann = ANN(id=self.available_nn_id,
                                  hidden_layers_for_mlp=hidden_layers_for_mlp,
                                  learning_rate=5 / (10 ** lr_denominator_in_log10),
                                  network_type=network_type,
                                  optimizer_type=optimizer_type,
                                  loss_estimator_delta=self.loss_estimator_delta,
                                  back_prop_skip_loss_threshold=self.back_prop_skip_loss_threshold,
                                  task_detector_type=self.task_detector_type,
                                  num_classes=self.num_classes,
                                  device=self.device,
                                  train_task_predictor_at_the_end=self.train_task_predictor_at_the_end,
                                  train_nn_using_ex_static_f= self.train_nn_using_ex_static_f)
                    self.train_nets.append(tmp_ann)
                    self.available_nn_id += 1

    def get_majority_vote_from_nets(self, x, x_flatten, mini_batch_size, weights_for_each_network=None, add_best_training_nn_votes=False, predictor=None, static_features=None):
        votes = None

        if self.prediction_pool == POOL_FROZEN:
            nn_list = self.frozen_nets
        elif self.prediction_pool == POOL_TRAINING:
            nn_list = self.train_nets

        for i in range(len(nn_list)):
            v = nn_list[i].get_votes(x, x_flatten, static_features)

            if weights_for_each_network is not None:
                v *= weights_for_each_network[i]

            if votes is None:
                votes = v
            else:
                votes = torch.cat((votes, v), dim=0)
            nn_list[i].chosen_for_test += mini_batch_size

        if self.prediction_pool == POOL_FROZEN:
            if add_best_training_nn_votes:
                i = self.get_nn_index_with_lowest_loss(self.train_nets, use_estimated_loss=True)
                nn_list = self.train_nets
                v = nn_list[i].get_votes(x, x_flatten)

                if weights_for_each_network is not None:
                    if votes is not None and nn_list[i].one_class_detector_fit_called:
                        w, _ = self.get_best_matched_nn_index_and_weights_via_predictor(x, x_flatten, predictor, [nn_list[i]], static_features)
                        v *= w[i]

                if votes is None:
                    votes = v
                else:
                    votes = torch.cat((votes, v), dim=0)
                nn_list[i].chosen_for_test += mini_batch_size

        return torch.mean(votes, dim=0)

    def get_network_with_best_confidence(self, x, x_flatten):
        predictions = None
        for i in range(len(self.frozen_nets)):
            if self.frozen_nets[i].network_type == NETWORK_TYPE_CNN:
                xx = x
            else:
                xx = x_flatten
            if i == 0:
                predictions = self.frozen_nets[i].net(xx).unsqueeze(0)
            else:
                predictions = torch.cat((predictions, self.frozen_nets[i].net(xx).unsqueeze(0)), dim=0)
        return np.argmax(predictions.sum(axis=1).sum(axis=1), axis=0)


    def get_one_class_probas_for_nn(self, x, x_flatten, n, feature_extractor=None, static_features=None):
        one_class_y = None
        one_class_p = None
        one_class_df = None

        if feature_extractor is not None:
            if static_features is not None:
                xx = static_features
            else:
                xx = self.get_static_features(x, feature_extractor, self.f_ex_device)
        else:
            if n.network_type == NETWORK_TYPE_CNN:
                xx = n.net.features(x)
                xx = xx.view(xx.size(0), -1)
            else:
                xx = x_flatten

        xxx = xx.cpu().numpy()
        one_class_y = [0.0]
        if n.one_class_detector_fit_called:
            one_class_y = n.one_class_detector.predict(xxx)
            one_class_df = n.one_class_detector.decision_function(xxx)
            one_class_df = one_class_df.reshape(-1, 1)
            one_class_p = n.logistic_regression.predict_proba(one_class_df)
            one_class_p = one_class_p[:, 1]  # get probabilities for class 1 (inlier)

        return one_class_df, one_class_y, one_class_p

    def get_best_matched_nn_index_and_weights_via_predictor(self, x, x_flatten, predictor, net_list, static_features=None):
        if predictor == PREDICT_METHOD_ONE_CLASS:
            predictions = []
            for i in range(len(net_list)):
                one_class_df, one_class_y, one_class_p = self.get_one_class_probas_for_nn(
                    x,
                    x_flatten,
                    net_list[i],
                    self.f_ex if self.use_static_f_ex else None,
                    static_features)
                if self.use_one_class_probas:
                    p = one_class_p
                else:
                    p = one_class_y

                if p is not None:
                    predictions.append(p)
            if len(predictions) > 0:
                predictions = np.array(predictions)
                best_matched_frozen_nn_index = np.argmax(predictions.mean(axis=1), axis=0).item()
                weights_for_each_network = np.mean(predictions, axis=1) + self.vote_weight_bias
            else:
                weights_for_each_network = None
                best_matched_frozen_nn_index = -1

        elif predictor == PREDICT_METHOD_NAIVE_BAYES or predictor == PREDICT_METHOD_HT:
            predictions = self.nb_or_ht_predict(x, static_features)
            best_matched_frozen_nn_index = np.argmax(predictions.mean(axis=0), axis=0).item()
            weights_for_each_network = np.mean(predictions, axis=0) + self.vote_weight_bias

        return weights_for_each_network, best_matched_frozen_nn_index

    def nb_train(self, features, task_id):
        train_nb(features, self.nb_or_ht, task_id)

    def nb_or_ht_predict(self, x, static_features=None):
        p = None
        if self.use_static_f_ex:
            if static_features is not None:
                ex_f = static_features
            else:
                ex_f = self.get_static_features(x, self.f_ex, self.f_ex_device)
            p = self.nb_or_ht.predict_proba(ex_f.cpu())
        return p

    def get_nn_index_with_lowest_loss(self, nn_list, use_estimated_loss=True):
        idx = 0
        min_loss = float("inf")
        for i in range(len(nn_list)):
            tmp_loss = nn_list[i].loss_estimator.estimation if use_estimated_loss else nn_list[i].current_loss
            if tmp_loss < min_loss:
                idx = i
                min_loss = tmp_loss
        # self.train_nets.sort(key=lambda ann: ann.loss_estimator.estimation)
        return idx

    def save_best_model_and_append_to_paths(self, best_model_idx):
        best_model: ANN = self.train_nets[best_model_idx]
        outputs = best_model.outputs
        best_model.outputs = None
        frozen_id = str(self.training_exp) + '_' + str(self.detected_task_id) + '_' + str(best_model.id)
        model_save_name = str(self.detected_task_id) + '_' + str(best_model.id) + '_' + best_model.model_name

        abstract_model_file_name = os.path.join(self.model_dump_dir, model_save_name)
        nn_model_file_name = os.path.join(self.model_dump_dir, model_save_name + '_nn')

        best_model.frozen_id = frozen_id
        save_model(best_model, abstract_model_file_name, nn_model_file_name, preserve_net=True)
        best_model.frozen_id = None
        best_model.outputs = outputs

        self.frozen_net_module_paths.append({'abstract_model_file_name': abstract_model_file_name,
                                     'nn_model_file_name': nn_model_file_name})

    def clear_frozen_pool(self):
        if len(self.frozen_nets) == 0:
            return
        try:
            for i in range(len(self.frozen_net_module_paths)):
                save_model(self.frozen_nets[i],
                           self.frozen_net_module_paths[i]['abstract_model_file_name'],
                           self.frozen_net_module_paths[i]['nn_model_file_name'])
                self.frozen_nets[i] = None
        except:
            print("Exception thrown. frozen_nets:\n{}", self.frozen_nets)
            print("Exception thrown. frozen_net_module_paths:\n{}", self.frozen_net_module_paths)
            print("Exception thrown. self:\n{}", self)

        self.frozen_nets = []

    def load_frozen_pool(self):
        if len(self.frozen_nets) != 0:
            print('Frozen pool is not empty')
            return

        for i in range(len(self.frozen_net_module_paths)):
            self.frozen_nets.append(load_model(self.frozen_net_module_paths[i]['abstract_model_file_name'],
                                               self.frozen_net_module_paths[i]['nn_model_file_name']))

    def reset(self):
        if self.reset_training_pool:
            # configuration variables (which has the same name as init parameters) should be copied by the caller function
            for i in range(len(self.train_nets)):
                self.train_nets[i] = None
            self.train_nets = None
            self.train_nets = []  # type: List[ANN]
            self.create_nn_pool()
        return self

    def reset_one_class_detectors_and_loss_estimators_seen_task_ids(self):
        for n in self.train_nets:
            n.one_class_detector = init_one_class_detector()
            n.scaler = init_scaler()
            n.logistic_regression = init_logistic_regression()
            n.one_class_detector_fit_called = False
            n.loss_estimator = ADWIN(delta=n.loss_estimator_delta)
            n.seen_task_ids_train = {}
            n.correctly_predicted_task_ids_test = {}
            n.correctly_predicted_task_ids_test_at_last = {}
            n.correctly_predicted_task_ids_probas_test_at_last = {}

    @torch.no_grad()
    def add_nn_with_lowest_loss_to_frozen_list(self):
        idx = self.get_nn_index_with_lowest_loss(self.train_nets, use_estimated_loss=True)
        self.print_nn_list([self.train_nets[idx]], list_type='train_net', dumped_at='task_detect')

        for i in range(len(self.train_nets)):
            if i != idx:  # clear other nets memory
                if self.reset_training_pool:
                    self.train_nets[i] = None
                else:
                    self.train_nets[i].accumulated_features = None

        if self.train_task_predictor_at_the_end != DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END:
            learned_features = [None]
            device = torch.device('cpu')
            self.train_nets[idx].net.to(device)  # move the model to cpu

            # fills learned_features
            if self.train_task_predictor_at_the_end == WITH_ACCUMULATED_INSTANCES:
                outputs = self.train_nets[idx].net(self.accumulated_x_or_features[0].to(device), learned_features=learned_features)
            elif self.train_task_predictor_at_the_end == WITH_ACCUMULATED_STATIC_FEATURES:
                learned_features[0] = self.accumulated_x_or_features[0].to(device)
            else:
                learned_features[0] = self.train_nets[idx].accumulated_features

            if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                self.train_nets[idx].train_one_class_classifier(learned_features[0],
                                                                self.use_one_class_probas)
            elif self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_HT:
                self.nb_train(learned_features[0], self.detected_task_id)
            self.train_nets[idx].net.to(self.train_nets[idx].device)   # move the model back to original device
        else:
            if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                pass

        self.save_best_model_and_append_to_paths(idx)

        if self.reset_training_pool:
            self.train_nets[idx] = None  # clear chosen net's memory

        self.detected_task_id += 1
        self.accumulated_x_or_features = None
        self.accumulated_x_or_features = [None]

    def save_nb_predictions(self):
        if self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_HT:
            if self.training_exp == self.n_experiences:
                np.save(self.nb_stats_file, self.nb_preds)

    def check_accuracy_of_nb(self, x, x_flatten):
        for i in range(len(x)):
            p = self.nb_or_ht_predict(x[None, i, :])

            p_row = np.concatenate((p, self.mb_task_id[i].cpu().numpy().reshape((1, 1))), axis=1)
            if self.nb_preds is None:
                self.nb_preds = p_row
            else:
                self.nb_preds = np.concatenate((self.nb_preds, p_row))

    def check_accuracy_of_one_class_classifiers(self, x, x_flatten):
        if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
            for i in range(len(x)):
                task_id = self.mb_task_id[i].item()
                if self.instances_per_task_at_last.get(task_id) is None:
                    self.instances_per_task_at_last[task_id] = 1
                else:
                    self.instances_per_task_at_last[task_id] += 1
                for j in range(len(self.frozen_nets)):
                    is_nw_trained_on_task_id = 0 if self.frozen_nets[j].seen_task_ids_train.get(task_id) is None else 1

                    one_class_df, one_class_y, one_class_p = self.get_one_class_probas_for_nn(
                        x[None, i, :],
                        x_flatten[None, i, :],
                        self.frozen_nets[j],
                        self.f_ex if self.use_static_f_ex else None)
                    # get in-class or not
                    if one_class_y is not None and one_class_y.item() > 0.0:
                        self.correct_network_selected_count_at_last += 1
                        if self.frozen_nets[j].correctly_predicted_task_ids_test_at_last.get(task_id) is None:
                            self.frozen_nets[j].correctly_predicted_task_ids_test_at_last[task_id] = 1
                        else:
                            self.frozen_nets[j].correctly_predicted_task_ids_test_at_last[task_id] += 1
                    # get probas
                    if one_class_p is not None and one_class_p.item() > 0.0:
                        if self.frozen_nets[j].correctly_predicted_task_ids_probas_test_at_last.get(task_id) is None:
                            self.frozen_nets[j].correctly_predicted_task_ids_probas_test_at_last[task_id] = one_class_p.item()
                        else:
                            self.frozen_nets[j].correctly_predicted_task_ids_probas_test_at_last[task_id] += one_class_p.item()

                    if not self.one_class_stats_header_printed:
                        print('{},{},{},{},{},{},{}'.format(
                            'task_id',
                            'nw_id',
                            'frozen_id',
                            'is_nw_trained_on_task_id',
                            'one_class_df',
                            'one_class_y',
                            'one_class_p'
                        ), file=self.one_class_stats_file, flush=True)
                        self.one_class_stats_header_printed = True

                    print('{},{},{},{},{},{},{}'.format(
                        task_id,
                        self.frozen_nets[j].id,
                        self.frozen_nets[j].frozen_id,
                        is_nw_trained_on_task_id,
                        one_class_df.item(),
                        one_class_y.item(),
                        one_class_p.item()
                    ), file=self.one_class_stats_file, flush=True)

    def get_static_features(self, x, feature_extractor, fx_device):
        x = x.to(fx_device)
        if x.shape[1] == 1 and not self.use_1_channel_pretrained_for_1_channel:  # has les than 3 channels
            preprocessed_x = preprocessor(x)  # repeat channel 1
            # preprocessed_x = x
        else:
            preprocessed_x = x
        return feature_extractor(preprocessed_x).cpu()

    def forward_pass(self, nn_list, **kwargs):
        xx = kwargs["xx"]
        r = kwargs["r"]
        c = kwargs["c"]
        y = kwargs["y"]
        true_task_id = kwargs["true_task_id"]
        use_instances_for_task_detector_training = kwargs["use_instances_for_task_detector_training"]
        static_features = kwargs["static_features"]
        t = []
        for i in range(len(nn_list)):
            if self.use_threads:
                t.append(threading.Thread(target=net_train, args=(
                nn_list[i], xx, r, c, y, true_task_id, use_instances_for_task_detector_training,
                self.use_one_class_probas, static_features, self.train_nn_using_ex_static_f,)))
            else:
                nn_list[i].train_net(xx, y, c, r, true_task_id, use_instances_for_task_detector_training,
                                             self.use_one_class_probas,
                                             static_features=static_features,
                                             train_nn_using_ex_static_f=self.train_nn_using_ex_static_f)
        if self.use_threads:
            for i in range(len(t)):
                t[i].start()
            for i in range(len(t)):
                t[i].join()

    def update_loss_estimator(self, nn_list):
        t = []
        for i in range(len(nn_list)):
            if self.use_threads:
                copy_old = False
                t.append(threading.Thread(target=update_loss_estimator, args=(nn_list[i], copy_old,)))
            else:
                nn_list[i].update_loss_estimator(copy_old=False)
        if self.use_threads:
            for i in range(len(t)):
                t[i].start()
            for i in range(len(t)):
                t[i].join()

    def call_backprop(self, nn_list):
        t = []
        for i in range(len(nn_list)):
            if self.use_threads:
                t.append(threading.Thread(target=call_backprop, args=(nn_list[i],)))
            else:
                nn_list[i].call_backprop()
        if self.use_threads:
            for i in range(len(t)):
                t[i].start()
            for i in range(len(t)):
                t[i].join()

    def reset_loss_and_bp_buffers(self, nn_list):
        t = []
        for i in range(len(nn_list)):
            if self.use_threads:
                t.append(threading.Thread(target=reset_loss_and_bp_buffers, args=(nn_list[i],)))
            else:
                nn_list[i].reset_loss_and_bp_buffers()
        if self.use_threads:
            for i in range(len(t)):
                t[i].start()
            for i in range(len(t)):
                t[i].join()

    def forward(self, x):
        r = x.shape[0]
        c_flatten, x_flatten = flatten_dimensions(x)
        y = self.mb_yy
        true_task_id = self.mb_task_id.sum(dim=0).item() // self.mb_task_id.shape[0]
        static_features = None
        # if x.shape[1] < 3:
        #     x = repeat_channel_1(x)
        if self.use_static_f_ex:
            if self.f_ex is None:
                self.f_ex_device, self.f_ex = create_static_feature_extractor(
                    device=self.device,
                    use_single_channel_fx = True if x.shape[1] == 1 and self.use_1_channel_pretrained_for_1_channel else False
                )
            static_features = self.get_static_features(x, self.f_ex, fx_device=self.f_ex_device)

        if self.call_predict:
            self.samples_seen_for_test += r
            final_votes = None
            best_matched_frozen_nn_idx = -1
            weights_for_frozen_nns = None
            if self.task_detector_type == PREDICT_METHOD_ONE_CLASS \
                    or self.task_detector_type == PREDICT_METHOD_RANDOM \
                    or self.task_detector_type == PREDICT_METHOD_TASK_ID_KNOWN \
                    or self.task_detector_type == PREDICT_METHOD_NW_CONFIDENCE \
                    or self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES \
                    or self.task_detector_type == PREDICT_METHOD_HT:
                if self.task_detector_type == PREDICT_METHOD_ONE_CLASS \
                        or self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES \
                        or self.task_detector_type == PREDICT_METHOD_HT:
                    weights_for_frozen_nns, best_matched_frozen_nn_idx = \
                        self.get_best_matched_nn_index_and_weights_via_predictor(x, x_flatten,
                                                                                 self.task_detector_type,
                                                                                 self.frozen_nets if self.prediction_pool == POOL_FROZEN else self.train_nets,
                                                                                 static_features)
                    if self.training_exp == self.n_experiences:
                        if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                            self.check_accuracy_of_one_class_classifiers(x, x_flatten)
                        elif self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_HT:
                            self.check_accuracy_of_nb(x, x_flatten)

                elif self.task_detector_type == PREDICT_METHOD_RANDOM:
                    best_matched_frozen_nn_idx = random.randrange(0, len(self.frozen_nets))
                elif self.task_detector_type == PREDICT_METHOD_TASK_ID_KNOWN:
                    best_matched_frozen_nn_idx = true_task_id
                elif self.task_detector_type == PREDICT_METHOD_NW_CONFIDENCE:
                    best_matched_frozen_nn_idx = self.get_network_with_best_confidence(x, x_flatten)

                if (self.prediction_pool == POOL_FROZEN  and (0 <= best_matched_frozen_nn_idx < len(self.frozen_nets))) or \
                        (self.prediction_pool == POOL_TRAINING and (0 <= best_matched_frozen_nn_idx < len(self.train_nets))):
                    if self.prediction_pool == POOL_FROZEN:
                        nn_list = self.frozen_nets
                    else:
                        nn_list = self.train_nets

                    nn_list[best_matched_frozen_nn_idx].chosen_for_test += r

                    if nn_list[best_matched_frozen_nn_idx].seen_task_ids_train.get(true_task_id) is not None:
                        self.correct_network_selected_count += r
                        if nn_list[best_matched_frozen_nn_idx].correctly_predicted_task_ids_test.get(true_task_id) is None:
                            nn_list[best_matched_frozen_nn_idx].correctly_predicted_task_ids_test[true_task_id] = r
                        else:
                            nn_list[best_matched_frozen_nn_idx].correctly_predicted_task_ids_test[true_task_id] += r
                else:
                    if len(self.frozen_nets) >= 0:
                        print('Index error for best_matched_frozen_nn_index ({}) frozen_nets size ({}) '.format(best_matched_frozen_nn_idx, len(self.frozen_nets)))
                    else:
                        print('No frozen nets. may use best training net for prediction')

                if self.prediction_pool == POOL_FROZEN:
                    if self.use_weights_from_task_detectors and (
                            self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or
                            self.task_detector_type == PREDICT_METHOD_HT or
                            self.task_detector_type == PREDICT_METHOD_ONE_CLASS):
                        if best_matched_frozen_nn_idx < 0:
                            print(
                                'No frozen nets. may use best training net for prediction for PREDICT_METHOD_NAIVE_BAYES or PREDICT_METHOD_HT or PREDICT_METHOD_ONE_CLASS')
                        final_votes = self.get_majority_vote_from_nets(
                            x, x_flatten, r,
                            weights_for_each_network=weights_for_frozen_nns,
                            add_best_training_nn_votes=True if self.auto_detect_tasks and (
                                        len(self.frozen_nets) == 0 or best_matched_frozen_nn_idx == len(
                                    self.frozen_nets)) else False,
                            predictor=self.task_detector_type,
                            static_features=static_features)
                    else: # No weights from predictors
                        final_votes = \
                            self.frozen_nets[best_matched_frozen_nn_idx].net(x if self.frozen_nets[best_matched_frozen_nn_idx].network_type == NETWORK_TYPE_CNN else x_flatten)
                else: # POOL_TRAINING
                    # if self.use_weights_from_task_detectors
                    #     not implemented yet
                    final_votes = \
                        self.train_nets[best_matched_frozen_nn_idx].net(x if self.train_nets[best_matched_frozen_nn_idx].network_type == NETWORK_TYPE_CNN else x_flatten)

            elif self.task_detector_type == PREDICT_METHOD_MAJORITY_VOTE:
                final_votes = self.get_majority_vote_from_nets(
                    x, x_flatten, r,
                    weights_for_each_network=None,
                    add_best_training_nn_votes=False,
                    predictor=None,
                    static_features=static_features)

            correct_class_predicted = torch.eq(torch.argmax(final_votes, dim=1), y).sum().item()
            self.correct_class_predicted += correct_class_predicted

            if self.task_detector_type != PREDICT_METHOD_MAJORITY_VOTE:
                if self.use_weights_from_task_detectors:
                    pass
                else:
                    if best_matched_frozen_nn_idx >= 0:
                        if self.prediction_pool == POOL_FROZEN:
                            self.frozen_nets[best_matched_frozen_nn_idx].correct_class_predicted += correct_class_predicted
                        else:
                            self.train_nets[best_matched_frozen_nn_idx].correct_class_predicted += correct_class_predicted

            return final_votes

        else:  # train
            self.samples_seen_for_train_after_dd += r
            self.total_samples_seen_for_train += r
            if self.samples_per_each_task_at_train.get(true_task_id) is None:
                self.samples_per_each_task_at_train[true_task_id] = r
            else:
                self.samples_per_each_task_at_train[true_task_id] += r

        use_instances_for_task_detector_training = True if random.randint(0, 0) == 0 else False

        x_or_features = None
        if use_instances_for_task_detector_training:
            if self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or \
                    self.task_detector_type == PREDICT_METHOD_HT or \
                    self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                if self.train_task_predictor_at_the_end == DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END:
                    if self.use_static_f_ex:
                        x_or_features = static_features
                        if self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_HT:
                            if self.prediction_pool == POOL_FROZEN:
                                self.nb_train(x_or_features, self.detected_task_id)
                else:  # TRAIN_TASK_PREDICTOR_AT_THE_END
                    if self.train_task_predictor_at_the_end == WITH_ACCUMULATED_INSTANCES:
                        x_or_features = x.cpu()
                    elif self.train_task_predictor_at_the_end == WITH_ACCUMULATED_STATIC_FEATURES:
                        x_or_features = static_features
                    if x_or_features is not None:
                        if self.accumulated_x_or_features[0] is None:
                            self.accumulated_x_or_features[0] = x_or_features
                        else:
                            self.accumulated_x_or_features[0] = torch.cat([self.accumulated_x_or_features[0], x_or_features],
                                                                          dim=0)
                    x_or_features = None

        # number_of_mlps_to_train = self.number_of_mlps_to_train
        # number_of_top_mlps_to_train = self.number_of_mlps_to_train // 2

        # if self.samples_seen_for_train_after_dd < self.number_of_instances_to_train_using_all_mlps_at_start:
        #     number_of_mlps_to_train = len(self.train_nets)
        #     number_of_top_mlps_to_train = len(self.train_nets) // 2

        if self.train_nets[0].network_type == NETWORK_TYPE_CNN:
            xx = x
            c = None
        else:
            xx = x_flatten
            c = c_flatten

        self.clear_frozen_pool()
        nn_list = []
        if self.train_only_the_best_nn and self.total_samples_seen_for_train > 1000:
            for i in range(len(self.train_nets)):
                nn_list.append(self.train_nets[i])
            nn_with_lowest_loss = self.get_nn_index_with_lowest_loss(nn_list, use_estimated_loss=True)
            self.train_nets[nn_with_lowest_loss].train_net(xx, y, c, r, true_task_id, use_instances_for_task_detector_training,
                                                self.use_one_class_probas,
                                                static_features=static_features,
                                                train_nn_using_ex_static_f=self.train_nn_using_ex_static_f)
        else:
            if self.random_train_frozen_if_best and random.randint(0, 9) < 3:
                forward_pass_both_lists = True
                for i in range (len(self.train_nets)):
                    nn_list.append(self.train_nets[i])
                self.load_frozen_pool()
                for i in range (len(self.frozen_nets)):
                    nn_list.append(self.frozen_nets[i])
            else:
                forward_pass_both_lists = False
                nn_list = self.train_nets

            self.forward_pass(nn_list,
                              xx=xx,
                              r=r,
                              c=c,
                              y=y,
                              true_task_id=true_task_id,
                              use_instances_for_task_detector_training=use_instances_for_task_detector_training,
                              static_features=static_features)

            nn_with_lowest_loss = self.get_nn_index_with_lowest_loss(nn_list,
                                                                     use_estimated_loss= True if self.prediction_pool == POOL_FROZEN and not forward_pass_both_lists else False)
            # outputs = deepcopy(self.train_nets[nn_with_lowest_loss].outputs.detach())

            if self.prediction_pool == POOL_FROZEN or self.total_samples_seen_for_train <= 1000:
                self.update_loss_estimator(nn_list)
                if forward_pass_both_lists and nn_with_lowest_loss <= len(self.train_nets): # trained both training and frozen and frozen has the lowest loss.
                    nn_list[nn_with_lowest_loss].call_backprop()
                else:
                    self.call_backprop(nn_list)
                self.reset_loss_and_bp_buffers(nn_list)

            if self.prediction_pool == POOL_FROZEN:
                if self.auto_detect_tasks:
                    task_detected = False
                    for m in self.train_nets:
                        if m.task_detected:
                            task_detected = True
                    if task_detected:
                        self.samples_seen_for_train_after_dd = 0
                        self.add_to_frozen_pool()
            else: # POOL_TRAINING
                if self.total_samples_seen_for_train > 1000:
                    nn_list[nn_with_lowest_loss].update_loss_estimator(copy_old=False)
                    selected_network = nn_with_lowest_loss
                    task_detected = False
                    # task_detected = self.train_nets[nn_with_lowest_loss].task_detected
                    for m in nn_list:
                        if m.task_detected:
                            task_detected = True
                    if self.auto_detect_tasks and self.samples_seen_for_train_after_dd > 100 and task_detected and len(self.train_nets) < self.train_pool_max:
                        self.samples_seen_for_train_after_dd = 0
                        self.add_to_train_pool(nn_with_lowest_loss)
                        selected_network = len(self.train_nets) -1

                    nn_list[nn_with_lowest_loss].old_loss_estimator = None
                    nn_list[nn_with_lowest_loss].call_backprop()
                    self.nb_train(x_or_features, selected_network)

                    for i in range(len(nn_list)):
                        nn_list[i].reset_loss_and_bp_buffers()

        # if (self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_HT or self.task_detector_type == PREDICT_METHOD_ONE_CLASS) and use_instances_for_task_detector_training:
        #     pass

        nn_list[nn_with_lowest_loss].chosen_after_train += r
        return nn_list[nn_with_lowest_loss].outputs

    def add_to_train_pool(self, nn_with_lowest_loss):
        print("Adding item to training pool===")
        # Append a copy of nn_with_lowest_loss to the training pool
        model_save_name = 'tmp'
        abstract_model_file_name = os.path.join(self.model_dump_dir, model_save_name)
        nn_model_file_name = os.path.join(self.model_dump_dir, model_save_name + '_nn')

        save_model(self.train_nets[nn_with_lowest_loss], abstract_model_file_name, nn_model_file_name, preserve_net=True)
        tmp_model = load_model(abstract_model_file_name, nn_model_file_name)

        self.train_nets.append(tmp_model)
        os.remove(abstract_model_file_name)
        os.remove(nn_model_file_name)

        #self.train_nets.append(deepcopy(self.train_nets[nn_with_lowest_loss]))

        # self.train_nets.append(self.train_nets[nn_with_lowest_loss])

        if self.train_nets[nn_with_lowest_loss].task_detected:
            # preserve the old loss estimator on this NW as we are going to train NB with newly added NW
            # self.train_nets[nn_with_lowest_loss].loss_estimator = self.train_nets[nn_with_lowest_loss].old_loss_estimator
            pass
        self.train_nets[nn_with_lowest_loss].old_loss_estimator = None

        self.detected_task_id += 1

    def add_to_frozen_pool(self):
        self.add_nn_with_lowest_loss_to_frozen_list()
        self.reset_one_class_detectors_and_loss_estimators_seen_task_ids()
        self.reset()

    def add_to_pool(self):
        if self.prediction_pool == POOL_FROZEN:
            self.add_to_frozen_pool()
        else: # POOL_TRAINING
            nn_with_lowest_loss = self.get_nn_index_with_lowest_loss(self.train_nets, use_estimated_loss=False)
            self.add_to_train_pool(nn_with_lowest_loss)

    def print_stats_hader(self):
        print('Training pool size {}'.format(len(self.train_nets)))
        if not self.heading_printed:
            print('training_exp,'
                  'dumped_at,'
                  'detected_task_id,'
                  'list_type,'
                  'total_samples_seen_for_train,'
                  'samples_per_each_task_at_train,'
                  'samples_seen_for_train_after_dd,'
                  'this_name,'
                  'this_id,'
                  'this_frozen_id,'
                  'this_samples_seen_at_train,'
                  'this_trained_count,'
                  'this_seen_task_ids_train,'
                  'this_avg_loss,'
                  'this_estimated_loss,'
                  'this_chosen_after_train,'
                  'total_samples_seen_for_test,'
                  'this_chosen_for_test,'
                  'this_correctly_predicted_task_ids_test,'
                  'this_correctly_predicted_task_ids_test_at_last,'
                  'this_correctly_predicted_task_ids_probas_test_at_last,'
                  'correct_network_selected,'
                  'correct_network_selected_count_at_last,'
                  'instances_per_task_at_last,'
                  'this_correct_class_predicted,'
                  'correct_class_predicted',
                  file=self.stats_file, flush=True)
            self.heading_printed = True

    def print_stats(self, dumped_at=None):
        # print('---train_nets---', file=self.stats_file)
        self.print_nn_list(self.train_nets, list_type='train_net', dumped_at=dumped_at)
        # print('---frozen_nets---', file=self.stats_file)
        if len(self.frozen_nets) > 0:
            self.print_nn_list(self.frozen_nets, list_type='frozen_net', dumped_at=dumped_at)

    def print_nn_list(self, nn_l, list_type=None, dumped_at=None):
        self.print_stats_hader()

        for i in range(len(nn_l)):
            print('{},{},{},{},{},"{}",{},{},{},{},{},{},"{}",{},{},{},{},{},"{}","{}","{}",{},{},"{}",{},{}'.format(
                self.training_exp - 1 if dumped_at == 'after_eval' else self.training_exp,
                dumped_at,
                self.detected_task_id,
                list_type,
                self.total_samples_seen_for_train,
                self.samples_per_each_task_at_train,
                self.samples_seen_for_train_after_dd,
                nn_l[i].model_name,
                nn_l[i].id,
                nn_l[i].frozen_id,
                nn_l[i].samples_seen_at_train,
                nn_l[i].trained_count,
                nn_l[i].seen_task_ids_train,
                0 if nn_l[i].samples_seen_at_train == 0 else nn_l[i].accumulated_loss / nn_l[i].samples_seen_at_train,
                nn_l[i].loss_estimator.estimation,
                nn_l[i].chosen_after_train,
                self.samples_seen_for_test,
                nn_l[i].chosen_for_test,
                nn_l[i].correctly_predicted_task_ids_test,
                nn_l[i].correctly_predicted_task_ids_test_at_last,
                nn_l[i].correctly_predicted_task_ids_probas_test_at_last,
                self.correct_network_selected_count,
                self.correct_network_selected_count_at_last,
                self.instances_per_task_at_last,
                nn_l[i].correct_class_predicted,
                self.correct_class_predicted
            ), file=self.stats_file, flush=True)
