import random
import sys
import threading

import numpy as np
from copy import deepcopy
from math import log

from sklearn.svm import OneClassSVM

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.drift_detection import ADWIN
from skmultiflow.bayes import NaiveBayes

import torch
import torch.nn as nn
import torch.optim as optim

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


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10, num_channels=0):
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


class PyNet(nn.Module):
    def __init__(self, hidden_layers: list = None, num_classes: int = None, input_dimensions=None):
        super(PyNet, self).__init__()

        if hidden_layers is None:
            return
        linear = []
        self.f = []

        # Add hidden layers
        for h in range(0, len(hidden_layers), 1):
            if h == 0:  # first hidden layer
                in_d = input_dimensions
                nonlinearity = Relu
            else:
                in_d = hidden_layers[h - 1]['neurons']
                nonlinearity = hidden_layers[h]['nonlinearity']
            out_d = hidden_layers[h]['neurons']
            linear.append(nn.Linear(in_d, out_d))
            if nonlinearity == Tanh:
                self.f.append(nn.Tanh())
            elif nonlinearity == Sigmoid:
                self.f.append(nn.Sigmoid())
            elif nonlinearity == Relu:
                self.f.append(nn.ReLU())
            elif nonlinearity == LeakyRelu:
                self.f.append(nn.LeakyReLU())
            else:
                pass

        # output layer
        linear.append(nn.Linear(hidden_layers[len(hidden_layers) - 1]['neurons'], num_classes))
        # self.f.append(nn.ReLU())

        self.linear = nn.ModuleList(linear)

    def forward(self, X, learned_features=None):
        if learned_features:
            # pass learned features from last layer
            if learned_features[0] is None:
                learned_features[0] = X.detach()

        x = X
        for i, l in enumerate(self.linear):
            if i == len(self.linear) - 1:
                x = l(x)
            else:
                x = self.f[i](l(x))
        return x


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
                 adwin_delta=1e-3,
                 use_adwin=False,
                 task_detector_type=PREDICT_METHOD_ONE_CLASS,
                 back_prop_skip_loss_threshold=0.6,
                 train_task_predictor_at_the_end=True):
        # configuration variables (which has the same name as init parameters)
        self.id = id
        self.model_name = None
        self.learning_rate = learning_rate
        self.network_type = network_type
        # self.hidden_layers = copy.deepcopy(hidden_layers)
        self.hidden_layers_for_MLP = deepcopy(hidden_layers_for_mlp)
        self.num_classes = num_classes
        self.device = device
        self.optimizer_type = optimizer_type
        self.loss_f = loss_f
        self.adwin_delta = adwin_delta
        self.back_prop_skip_loss_threshold = back_prop_skip_loss_threshold
        self.use_adwin = use_adwin
        self.task_detector_type = task_detector_type
        self.train_task_predictor_at_the_end = train_task_predictor_at_the_end

        # status variables
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.samples_seen_at_train = 0
        self.trained_count = 0
        self.chosen_for_test = 0
        self.chosen_after_train = 0
        self.estimator: BaseDriftDetector = None
        self.accumulated_loss = 0
        self.learned_features_x = [None]
        self.one_class_detector = None
        self.naive_bayes = None
        self.correct_network_selected_count = 0
        self.correct_class_predicted = 0
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
        self.estimator = ADWIN(delta=self.adwin_delta)
        self.accumulated_loss = 0
        self.learned_features_x = [None]
        self.one_class_detector = OneClassSVM(gamma='auto')
        self.naive_bayes = NaiveBayes()
        self.correct_network_selected_count = 0
        self.correct_class_predicted = 0

        if self.hidden_layers_for_MLP is None:
            pass
        elif isinstance(self.hidden_layers_for_MLP, nn.Module):
            # assumes input dimention is set properly in the network structure
            self.net = deepcopy(self.hidden_layers_for_MLP)
            self.initialize_net_para()
        elif isinstance(self.hidden_layers_for_MLP, list):
            if self.hidden_layers_for_MLP[0]['neurons'] is None or self.hidden_layers_for_MLP[0][
                'nonlinearity'] is None:
                print('Unknown hidden layer format is passed in: {}'.format(self.hidden_layers_for_MLP))
                print('Expected format :{}'.format(default_mlp_hidden_layers))
                exit(1)
        self.model_name = '{}_{}_{:05f}_{}'.format(
            'CNN' if self.network_type == NETWORK_TYPE_CNN else 'MLP_L1_' + str(
                log(self.hidden_layers_for_MLP[0]['neurons'], 2) // 1),
            self.optimizer_type,
            self.learning_rate,
            self.adwin_delta)

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

    def initialize_network(self, x, input_dimensions=None):
        if self.network_type == NETWORK_TYPE_CNN:
            number_of_channels = 0
            if len(x.shape) == 2:
                number_of_channels = 0
            else:
                number_of_channels = x.shape[1]
            self.net = SimpleCNN(num_channels=number_of_channels)
        else:
            self.net = PyNet(hidden_layers=self.hidden_layers_for_MLP, num_classes=self.num_classes,
                             input_dimensions=input_dimensions)
        self.initialize_net_para()

    def train_net(self, x, y, c, r, task_id, use_instances_for_task_detector_training):
        if self.net is None:
            self.initialize_network(x, input_dimensions=c)

        self.samples_seen_at_train += r

        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()  # zero the gradient buffers
        # forward propagation
        learned_features = None
        if (self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_ONE_CLASS) and use_instances_for_task_detector_training and not self.train_task_predictor_at_the_end:
            learned_features = [None]

        outputs = self.net(x, learned_features=learned_features)

        if self.train_task_predictor_at_the_end:
            pass
        else: # train task predictor online using current learned_features
            if self.task_detector_type == PREDICT_METHOD_ONE_CLASS and use_instances_for_task_detector_training:
                if self.learned_features_x[0] is None:
                    self.learned_features_x[0] = learned_features[0].cpu()
                else:
                    self.learned_features_x[0] = torch.cat([self.learned_features_x[0], learned_features[0].cpu()], dim=0)
            elif self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES:
                nb_y = np.empty((learned_features[0].shape[0],), dtype=np.int64)
                nb_y.fill(task_id)
                self.naive_bayes.partial_fit(learned_features[0].cpu().numpy(), nb_y)



        # backward propagation
        # print(self.net.linear[0].weight.data)
        self.loss = self.loss_f(outputs, y)
        if self.loss.item() > self.back_prop_skip_loss_threshold:
            self.loss.backward()
            self.optimizer.step()  # Does the update
            self.trained_count += 1

        self.estimator.add_element(self.loss.item())
        self.accumulated_loss += self.loss.item()

        # if self.estimator.detected_change():
        #     print('drift detected by {}'.format(self.model_name))
        #     pass

        return outputs

    def reset(self):
        # configuration variables (which has the same name as init parameters) should be copied by the caller function
        self.init_values()
        return self

    def get_loss_estimation(self):
        if self.use_adwin:
            return self.estimator.estimation
        else:
            return self.accumulated_loss / self.samples_seen_at_train if self.samples_seen_at_train != 0 else 0.0

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


def net_train(net: ANN, x: np.ndarray, r, c, y: np.ndarray, task_id, use_instances_for_task_detector_training):
    net.train_net(x, y, c, r, task_id, use_instances_for_task_detector_training)


def sort_by_loss(k: ANN):
    return k.get_loss_estimation()


class MultiMLP(nn.Module):
    def __init__(self,
                 num_classes=None,
                 use_threads=False,
                 stats_file=sys.stdout,
                 number_of_mlps_to_train=10,
                 number_of_instances_to_train_using_all_mlps_at_start=1000,
                 use_adwin=True,
                 adwin_delta=1e-3,
                 back_prop_skip_loss_threshold=0.6,
                 stats_print_frequency=0,
                 nn_pool_type='30MLP',
                 predict_method=PREDICT_METHOD_ONE_CLASS,
                 train_task_predictor_at_the_end=True,
                 device='cpu'):
        super().__init__()

        # configuration variables (which has the same name as init parameters)
        self.num_classes = num_classes
        self.use_threads = use_threads
        self.stats_file = stats_file
        self.number_of_mlps_to_train = number_of_mlps_to_train
        self.number_of_instances_to_train_using_all_mlps_at_start = number_of_instances_to_train_using_all_mlps_at_start
        self.adwin_delta = adwin_delta
        self.use_adwin = use_adwin
        self.stats_print_frequency = stats_print_frequency
        self.back_prop_skip_loss_threshold = back_prop_skip_loss_threshold
        self.nn_pool_type = nn_pool_type
        self.task_detector_type = predict_method
        self.train_task_predictor_at_the_end = train_task_predictor_at_the_end
        self.device = device

        # status variables
        self.train_nets = []  # type: List[ANN]
        self.frozen_nets = []  # type: List[ANN]
        self.heading_printed = False
        self.samples_seen_for_train_after_drift = 0
        self.total_samples_seen_for_train = 0
        self.samples_seen_for_test = 0
        self.test_samples_seen_for_learned_tasks = 0
        self.correct_network_selected_count = 0
        self.correct_class_predicted = 0
        self.call_predict = False
        self.mb_yy = None
        self.mb_task_id = None
        self.available_nn_id = 0
        self.task_id = 0
        self.accumulated_x = [None]
        # self.learned_tasks = [0]

        self.init_values()

    def init_values(self):
        # init status variables

        self.heading_printed = False
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
                                  adwin_delta=self.adwin_delta,
                                  use_adwin=self.use_adwin,
                                  back_prop_skip_loss_threshold=self.back_prop_skip_loss_threshold,
                                  task_detector_type=self.task_detector_type,
                                  num_classes=self.num_classes,
                                  device=self.device,
                                  train_task_predictor_at_the_end=self.train_task_predictor_at_the_end)
                    self.train_nets.append(tmp_ann)
                    self.available_nn_id += 1

    def get_majority_vote_from_frozen_nets(self, x, x_flatten, mini_batch_size):
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
            self.frozen_nets[i].chosen_for_test += mini_batch_size
        return predictions.sum(axis=0) / len(self.frozen_nets)

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

    # def get_best_matched_frozen_nn_index_using_a_predictor(self, x, x_flatten, predictor):
    #     predictions = []
    #     score_samples = []
    #     for i in range(len(self.frozen_nets)):
    #         if self.frozen_nets[i].network_type == NETWORK_TYPE_CNN:
    #             xx = self.frozen_nets[i].net.features(x)
    #             xx = xx.view(xx.size(0), -1)
    #         else:
    #             xx = x_flatten
    #         if predictor == PREDICT_METHOD_ONE_CLASS:
    #             predictions.append(self.frozen_nets[i].one_class_detector.predict(xx))
    #         elif predictor == PREDICT_METHOD_NAIVE_BAYES:
    #             predictions.append(self.frozen_nets[i].naive_bayes.predict(xx))
    #         # score_samples.append(self.train_nets[i].one_class_detector.score_samples(xx))
    #     predictions = np.array(predictions)
    #     # score_samples = np.array(score_samples)
    #     # score_samples, predictions = strategy.model.get_belongingness(strategy.mb_x, x_flatten)
    #     best_matched_frozen_nn_index = np.argmax(predictions.sum(axis=1), axis=0).item()
    #     return best_matched_frozen_nn_index

    def get_best_matched_frozen_nn_index_using_a_predictor(self, x, x_flatten, predictor):
        predictions = []
        # score_samples = []
        for i in range(len(self.frozen_nets)):
            if self.frozen_nets[i].network_type == NETWORK_TYPE_CNN:
                xx = self.frozen_nets[i].net.features(x)
                xx = xx.view(xx.size(0), -1)
            else:
                xx = x_flatten
            if predictor == PREDICT_METHOD_ONE_CLASS:
                predictions.append(self.frozen_nets[i].one_class_detector.predict(xx.cpu().numpy()))
            elif predictor == PREDICT_METHOD_NAIVE_BAYES:
                # pre and post pad 0's to get array length len(self.frozen_nets)
                predictions.append(np.pad(self.frozen_nets[i].naive_bayes.predict_proba(xx.cpu().numpy()).sum(axis=0), (0, len(self.frozen_nets) - 1 - i), 'constant', constant_values=(0, 0)))

        if predictor == PREDICT_METHOD_ONE_CLASS:
            best_matched_frozen_nn_index = np.argmax(np.array(predictions).sum(axis=1), axis=0).item()
        elif predictor == PREDICT_METHOD_NAIVE_BAYES:
            best_matched_frozen_nn_index = np.argmax(np.array(predictions)).item()

        return best_matched_frozen_nn_index

    def get_train_nn_index_with_lowest_loss(self):
        self.train_nets.sort(key=sort_by_loss)
        return 0

    def reset(self):
        # configuration variables (which has the same name as init parameters) should be copied by the caller function
        for i in range(len(self.train_nets)):
            self.train_nets[i] = None
        self.train_nets = None
        self.train_nets = []  # type: List[ANN]
        self.create_nn_pool()
        return self

    @torch.no_grad()
    def add_nn_with_lowest_loss_to_frozen_list(self):
        idx = self.get_train_nn_index_with_lowest_loss()

        for i in range(len(self.train_nets)):
            if i != idx:  # clear other nets memory
                self.train_nets[i] = None

        if self.train_task_predictor_at_the_end:
            self.train_nets[idx].learned_features_x[0] = None  # clear this memory

            learned_features = [None]
            device = torch.device('cpu')
            self.train_nets[idx].net.to(device)  # move the model to cpu

            # fills learned_features
            outputs = self.train_nets[idx].net(self.accumulated_x[0].to(device), learned_features=learned_features)

            if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                self.train_nets[idx].one_class_detector.fit(learned_features[0].cpu().numpy())
            elif self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES:
                nb_y = np.empty((learned_features[0].shape[0],), dtype=np.int64)
                nb_y.fill(self.task_id)
                self.train_nets[idx].naive_bayes.partial_fit(learned_features[0].cpu().numpy(), nb_y)
            self.train_nets[idx].net.to(self.train_nets[idx].device)   # move the model back to original device
        else:
            if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                self.train_nets[idx].one_class_detector.fit(self.train_nets[idx].learned_features_x[0].numpy())
                self.train_nets[idx].learned_features_x[0] = None
        self.frozen_nets.append(self.train_nets[idx])
        self.task_id += 1
        self.accumulated_x = None
        self.accumulated_x = [None]

    def forward(self, x):
        r = x.shape[0]
        c_flatten, x_flatten = flatten_dimensions(x)
        y = self.mb_yy

        if self.call_predict:
            self.samples_seen_for_test += r
            true_task_id = self.mb_task_id.sum(dim=0).item() // self.mb_task_id.shape[0]
            if true_task_id < len(self.frozen_nets):
                self.test_samples_seen_for_learned_tasks += r
            class_votes = []
            if self.task_detector_type == PREDICT_METHOD_ONE_CLASS \
                    or self.task_detector_type == PREDICT_METHOD_RANDOM \
                    or self.task_detector_type == PREDICT_METHOD_TASK_ID_KNOWN \
                    or self.task_detector_type == PREDICT_METHOD_NW_CONFIDENCE \
                    or self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES:
                if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                    best_matched_frozen_nn_index = self.get_best_matched_frozen_nn_index_using_a_predictor(x, x_flatten, self.task_detector_type)
                elif self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES:
                    best_matched_frozen_nn_index = self.get_best_matched_frozen_nn_index_using_a_predictor(x, x_flatten, self.task_detector_type)
                elif self.task_detector_type == PREDICT_METHOD_RANDOM:
                    best_matched_frozen_nn_index = random.randrange(0, len(self.frozen_nets))
                elif self.task_detector_type == PREDICT_METHOD_TASK_ID_KNOWN:
                    best_matched_frozen_nn_index = true_task_id
                    if best_matched_frozen_nn_index >= len(self.frozen_nets):
                        best_matched_frozen_nn_index = len(self.frozen_nets) - 1
                elif self.task_detector_type == PREDICT_METHOD_NW_CONFIDENCE:
                    best_matched_frozen_nn_index = self.get_network_with_best_confidence(x, x_flatten)

                if best_matched_frozen_nn_index == true_task_id:
                    self.correct_network_selected_count += r
                    self.frozen_nets[best_matched_frozen_nn_index].correct_network_selected_count += r
                try:
                    self.frozen_nets[best_matched_frozen_nn_index].chosen_for_test += r
                except IndexError:
                    print('Index error reached best_matched_frozen_nn_index {}'.format(best_matched_frozen_nn_index))
                    best_matched_frozen_nn_index = len(self.frozen_nets) - 1
                class_votes = self.frozen_nets[best_matched_frozen_nn_index].net(
                    x if self.frozen_nets[best_matched_frozen_nn_index].network_type == NETWORK_TYPE_CNN else x_flatten)
            elif self.task_detector_type == PREDICT_METHOD_MAJORITY_VOTE:
                class_votes = self.get_majority_vote_from_frozen_nets(x, x_flatten, r)

            correct_class_predicted = (np.argmax(class_votes, axis=1) == y).sum().item()
            self.correct_class_predicted += correct_class_predicted
            self.frozen_nets[best_matched_frozen_nn_index].correct_class_predicted += correct_class_predicted

            return class_votes

        else:  # train
            self.samples_seen_for_train_after_drift += r
            self.total_samples_seen_for_train += r

        use_instances_for_task_detector_training = True if random.randint(0, 0) == 0 else False
        t = []
        number_of_mlps_to_train = self.number_of_mlps_to_train
        number_of_top_mlps_to_train = self.number_of_mlps_to_train // 2

        if self.samples_seen_for_train_after_drift < self.number_of_instances_to_train_using_all_mlps_at_start:
            number_of_mlps_to_train = len(self.train_nets)
            number_of_top_mlps_to_train = len(self.train_nets) // 2

        self.train_nets.sort(key=sort_by_loss)
        for i in range(number_of_mlps_to_train):
            if i < number_of_top_mlps_to_train:
                # top most train
                nn_index = i
            else:
                # Random train
                off_set = ((self.samples_seen_for_train_after_drift + i) % (
                            len(self.train_nets) - number_of_top_mlps_to_train))
                nn_index = number_of_top_mlps_to_train + off_set

            if self.train_nets[nn_index].network_type == NETWORK_TYPE_CNN:
                xx = x
                c = None
            else:
                xx = x_flatten
                c = c_flatten

            if self.use_threads:
                t.append(threading.Thread(target=net_train, args=(self.train_nets[nn_index], xx, r, c, y, self.task_id, use_instances_for_task_detector_training,)))
            else:
                self.train_nets[nn_index].train_net(xx, y, c, r, self.task_id, use_instances_for_task_detector_training)
        if self.use_threads:
            for i in range(len(t)):
                t[i].start()
            for i in range(len(t)):
                t[i].join()

        nn_with_lowest_loss = self.get_train_nn_index_with_lowest_loss()

        if (self.train_task_predictor_at_the_end and
                self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_ONE_CLASS) and use_instances_for_task_detector_training:
            if self.accumulated_x[0] is None:
                self.accumulated_x[0] = x.cpu()
            else:
                self.accumulated_x[0] = torch.cat([self.accumulated_x[0], x.cpu()], dim=0)

        self.train_nets[nn_with_lowest_loss].chosen_after_train += r
        return self.train_nets[nn_with_lowest_loss].net(
            x if self.train_nets[nn_with_lowest_loss].network_type == NETWORK_TYPE_CNN else x_flatten)

    def print_stats(self, after_eval=False):
        if not self.heading_printed:
            print('training_exp,'
                  'list_type,'
                  'total_samples_seen_for_train,'
                  'samples_seen_for_train_after_drift,'
                  'this_name,'
                  'this_id,'
                  'this_samples_seen_at_train,'
                  'this_trained_count,'
                  'this_avg_loss,'
                  'this_estimated_loss,'
                  'this_chosen_after_train,'
                  'total_samples_seen_for_test,'
                  'test_samples_seen_for_learned_tasks,'
                  'this_chosen_for_test,'
                  'this_correct_network_selected,'
                  'correct_network_selected,'
                  'this_acc,'
                  'acc',
                  file=self.stats_file, flush=True)
            self.heading_printed = True

        # print('---train_nets---', file=self.stats_file)
        self.print_nn_list(self.train_nets, list_type='train_net')
        # print('---frozen_nets---', file=self.stats_file)
        self.print_nn_list(self.frozen_nets, list_type='frozen_net')

    def print_nn_list(self, l, list_type=None):
        for i in range(len(l)):
            print('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
                self.task_id - 1,
                list_type,
                self.total_samples_seen_for_train,
                self.samples_seen_for_train_after_drift,
                l[i].model_name,
                l[i].id,
                l[i].samples_seen_at_train,
                l[i].trained_count,
                0 if l[i].samples_seen_at_train == 0 else l[i].accumulated_loss / l[i].samples_seen_at_train,
                l[i].estimator.estimation,
                l[i].chosen_after_train,
                self.samples_seen_for_test,
                self.test_samples_seen_for_learned_tasks,
                l[i].chosen_for_test,
                l[i].correct_network_selected_count / self.test_samples_seen_for_learned_tasks * 100 if self.test_samples_seen_for_learned_tasks != 0 else 0.0,
                self.correct_network_selected_count / self.test_samples_seen_for_learned_tasks * 100 if self.test_samples_seen_for_learned_tasks != 0 else 0.0,
                l[i].correct_class_predicted / self.test_samples_seen_for_learned_tasks * 100 if self.test_samples_seen_for_learned_tasks != 0 else 0.0,
                self.correct_class_predicted / self.test_samples_seen_for_learned_tasks * 100 if self.test_samples_seen_for_learned_tasks != 0 else 0.0
            ), file=self.stats_file, flush=True)
