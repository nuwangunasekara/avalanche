import os.path
import random
import sys
import threading

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
        self.loss_estimator_delta = loss_estimator_delta
        self.back_prop_skip_loss_threshold = back_prop_skip_loss_threshold
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
        self.loss_estimator: BaseDriftDetector = None
        self.accumulated_loss = 0
        self.one_class_detector = None
        self.one_class_detector_fit_called = False
        self.scaler = None
        self.logistic_regression = None
        self.naive_bayes = None
        self.correct_class_predicted = 0
        self.input_dimensions = 0
        self.x_shape = None
        self.task_detected = False
        self.seen_task_ids_train = {}
        self.seen_task_ids_test = {}
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
        self.naive_bayes = NaiveBayes()
        self.correct_class_predicted = 0
        self.input_dimensions = 0
        self.x_shape = None
        self.task_detected = False
        self.seen_task_ids_train = {}
        self.seen_task_ids_test = {}

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
            number_of_channels = 0
            if len(self.x_shape) == 2:
                number_of_channels = 0
            else:
                number_of_channels = self.x_shape[1]
            self.net = SimpleCNN(num_channels=number_of_channels)
        else:
            self.net = PyNet(hidden_layers=self.hidden_layers_for_MLP, num_classes=self.num_classes,
                             input_dimensions=self.input_dimensions)
        self.initialize_net_para()

    def train_one_class_classifier(self, features, train_logistic_regression):
        train_one_class_classifier(features,
                                   self.one_class_detector,
                                   train_logistic_regression,
                                   self.logistic_regression,
                                   self.scaler)
        self.one_class_detector_fit_called = True

    def train_net(self, x, y, c, r, true_task_id, detected_task_id, use_instances_for_task_detector_training, use_one_class_probas):
        if self.net is None:
            self.input_dimensions = c
            self.x_shape = deepcopy(x.shape)
            self.initialize_network()

        self.samples_seen_at_train += r

        if self.seen_task_ids_train.get(true_task_id) is None:
            self.seen_task_ids_train[true_task_id] = r
        else:
            self.seen_task_ids_train[true_task_id] += r

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
                self.train_one_class_classifier(learned_features[0].cpu(), use_one_class_probas)
            elif self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES:
                nb_y = np.empty((learned_features[0].shape[0],), dtype=np.int64)
                nb_y.fill(detected_task_id)
                self.naive_bayes.partial_fit(learned_features[0].cpu().numpy(), nb_y)

        # backward propagation
        # print(self.net.linear[0].weight.data)
        self.loss = self.loss_f(outputs, y)
        if self.loss.item() > self.back_prop_skip_loss_threshold:
            self.loss.backward()
            self.optimizer.step()  # Does the update
            self.trained_count += 1

        previous_estimated_loss = self.loss_estimator.estimation
        self.loss_estimator.add_element(self.loss.item())
        current_estimated_loss = self.loss.item()
        self.accumulated_loss += current_estimated_loss

        self.task_detected = False
        if self.loss_estimator.detected_change() and current_estimated_loss > previous_estimated_loss:
            print('NEW TASK detected by {}'.format(self.model_name))
            self.task_detected = True

        return outputs

    def get_votes(self, x, x_flatten):
        if self.network_type == NETWORK_TYPE_CNN:
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


def net_train(net: ANN, x: np.ndarray, r, c, y: np.ndarray, true_task_id, detected_task_id, use_instances_for_task_detector_training, use_one_class_probas):
    net.train_net(x, y, c, r, true_task_id, detected_task_id, use_instances_for_task_detector_training, use_one_class_probas)


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
                 number_of_mlps_to_train=10,
                 number_of_instances_to_train_using_all_mlps_at_start=1000,
                 loss_estimator_delta=1e-3,
                 back_prop_skip_loss_threshold=0.6,
                 stats_print_frequency=0,
                 nn_pool_type='30MLP',
                 predict_method=PREDICT_METHOD_ONE_CLASS,
                 train_task_predictor_at_the_end=True,
                 device='cpu',
                 model_dump_dir=None,
                 reset_training_pool=True,
                 use_one_class_probas=False,
                 vote_weight_bias=1e-3,
                 use_weights_from_task_detectors=False,
                 auto_detect_tasks=False):
        super().__init__()

        # configuration variables (which has the same name as init parameters)
        self.num_classes = num_classes
        self.use_threads = use_threads
        self.stats_file = stats_file
        self.number_of_mlps_to_train = number_of_mlps_to_train
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

        # status variables
        self.train_nets = []  # type: List[ANN]
        self.frozen_nets = []  # type: List[ANN]
        self.frozen_net_module_paths = []
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
        self.training_exp = -1
        self.available_nn_id = 0
        self.detected_task_id = 0
        self.accumulated_x = [None]
        # self.learned_tasks = [0]
        self.task_detected = False

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
                                  loss_estimator_delta=self.loss_estimator_delta,
                                  back_prop_skip_loss_threshold=self.back_prop_skip_loss_threshold,
                                  task_detector_type=self.task_detector_type,
                                  num_classes=self.num_classes,
                                  device=self.device,
                                  train_task_predictor_at_the_end=self.train_task_predictor_at_the_end)
                    self.train_nets.append(tmp_ann)
                    self.available_nn_id += 1

    def get_majority_vote_from_nets(self, x, x_flatten, mini_batch_size, weights_for_each_network=None, add_best_training_nn_votes=False, predictor=None):
        votes = None
        nn_list = self.frozen_nets
        for i in range(len(nn_list)):
            v = nn_list[i].get_votes(x, x_flatten)

            if weights_for_each_network is not None:
                v *= weights_for_each_network[i]

            if votes is None:
                votes = v
            else:
                votes = torch.cat((votes, v), dim=0)
            nn_list[i].chosen_for_test += mini_batch_size

        if add_best_training_nn_votes:
            i = self.get_train_nn_index_with_lowest_loss()
            nn_list = self.train_nets
            v = nn_list[i].get_votes(x, x_flatten)

            if weights_for_each_network is not None:
                if votes is not None:
                    w, _ = self.get_best_matched_nn_index_and_weights_via_predictor(x, x_flatten, predictor, [nn_list[i]])
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

    def get_task_predictor_probas_for_nn(self, x, x_flatten, predictor, n, n_idx):
        yyy = None
        if n.network_type == NETWORK_TYPE_CNN:
            xx = n.net.features(x)
            xx = xx.view(xx.size(0), -1)
        else:
            xx = x_flatten
        xxx = xx.cpu().numpy()
        if predictor == PREDICT_METHOD_ONE_CLASS:
            yyy = [0.0]
            if n.one_class_detector_fit_called:
                if self.use_one_class_probas:
                    df_scores = n.one_class_detector.decision_function(xxx)
                    df_scores = df_scores.reshape(-1, 1)
                    yyy = n.logistic_regression.predict_proba(df_scores)
                    yyy = yyy[:, 1]  # get probabilities for class 1 (inlier)
                else:
                    yyy = n.one_class_detector.predict(xxx)
        elif predictor == PREDICT_METHOD_NAIVE_BAYES:
            # pre and post pad 0's to get array length len(net_list)
            yyy = n.naive_bayes.predict_proba(xxx)
            yyy = yyy[:, n_idx]  # get probas for n_idx th task id from n_idx th (or n) network
        return yyy

    def get_best_matched_nn_index_and_weights_via_predictor(self, x, x_flatten, predictor, net_list):
        predictions = []
        # score_samples = []
        for i in range(len(net_list)):
            p = self.get_task_predictor_probas_for_nn(x, x_flatten, predictor, net_list[i], i)
            if p is not None:
                predictions.append(p)

        if len(predictions) > 0:
            predictions = np.array(predictions)
            best_matched_frozen_nn_index = np.argmax(predictions.sum(axis=1), axis=0).item()
            weights_for_each_network = np.mean(predictions, axis=1) + self.vote_weight_bias
        else:
            weights_for_each_network = None
            best_matched_frozen_nn_index = -1

        return weights_for_each_network, best_matched_frozen_nn_index

    def get_train_nn_index_with_lowest_loss(self):
        self.train_nets.sort(key=lambda ann: ann.loss_estimator.estimation)
        return 0

    def save_best_model_and_append_to_paths(self, best_model_idx):
        best_model: ANN = self.train_nets[best_model_idx]
        model_save_name = str(self.detected_task_id) + '_' + str(best_model.id) + '_' + best_model.model_name

        abstract_model_file_name = os.path.join(self.model_dump_dir, model_save_name)
        nn_model_file_name = os.path.join(self.model_dump_dir, model_save_name + '_nn')

        save_model(best_model, abstract_model_file_name, nn_model_file_name, preserve_net=True)

        self.frozen_net_module_paths.append({'abstract_model_file_name': abstract_model_file_name,
                                     'nn_model_file_name': nn_model_file_name})

    def clear_frozen_pool(self):
        for i in range(len(self.frozen_net_module_paths)):
            save_model(self.frozen_nets[i],
                       self.frozen_net_module_paths[i]['abstract_model_file_name'],
                       self.frozen_net_module_paths[i]['nn_model_file_name'])
            self.frozen_nets[i] = None
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
            n.loss_estimator = ADWIN(delta=n.loss_estimator_delta)
            n.seen_task_ids_train = {}
            n.seen_task_ids_test = {}

    @torch.no_grad()
    def add_nn_with_lowest_loss_to_frozen_list(self):
        idx = self.get_train_nn_index_with_lowest_loss()
        self.print_nn_list([self.train_nets[idx]], list_type='train_net', dumped_at='task_detect')

        if self.reset_training_pool:
            for i in range(len(self.train_nets)):
                if i != idx:  # clear other nets memory
                    self.train_nets[i] = None

        if self.train_task_predictor_at_the_end:
            learned_features = [None]
            device = torch.device('cpu')
            self.train_nets[idx].net.to(device)  # move the model to cpu

            # fills learned_features
            outputs = self.train_nets[idx].net(self.accumulated_x[0].to(device), learned_features=learned_features)

            if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                self.train_nets[idx].train_one_class_classifier(learned_features[0],
                                                                self.use_one_class_probas)
            elif self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES:
                nb_y = np.empty((learned_features[0].shape[0],), dtype=np.int64)
                nb_y.fill(self.detected_task_id)
                self.train_nets[idx].naive_bayes.partial_fit(learned_features[0].cpu().numpy(), nb_y)
            self.train_nets[idx].net.to(self.train_nets[idx].device)   # move the model back to original device
        else:
            if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                pass

        self.save_best_model_and_append_to_paths(idx)

        if self.reset_training_pool:
            self.train_nets[idx] = None  # clear chosen net's memory

        self.detected_task_id += 1
        self.accumulated_x = None
        self.accumulated_x = [None]

    def forward(self, x):
        r = x.shape[0]
        c_flatten, x_flatten = flatten_dimensions(x)
        y = self.mb_yy
        true_task_id = self.mb_task_id.sum(dim=0).item() // self.mb_task_id.shape[0]

        if self.call_predict:
            self.samples_seen_for_test += r
            self.test_samples_seen_for_learned_tasks += r
            final_votes = None
            best_matched_frozen_nn_idx = -1
            weights_for_frozen_nns = None
            if self.task_detector_type == PREDICT_METHOD_ONE_CLASS \
                    or self.task_detector_type == PREDICT_METHOD_RANDOM \
                    or self.task_detector_type == PREDICT_METHOD_TASK_ID_KNOWN \
                    or self.task_detector_type == PREDICT_METHOD_NW_CONFIDENCE \
                    or self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES:
                if self.task_detector_type == PREDICT_METHOD_ONE_CLASS \
                        or self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES:
                    weights_for_frozen_nns, best_matched_frozen_nn_idx = \
                        self.get_best_matched_nn_index_and_weights_via_predictor(x, x_flatten,
                                                                                 self.task_detector_type,
                                                                                 self.frozen_nets)
                elif self.task_detector_type == PREDICT_METHOD_RANDOM:
                    best_matched_frozen_nn_idx = random.randrange(0, len(self.frozen_nets))
                elif self.task_detector_type == PREDICT_METHOD_TASK_ID_KNOWN:
                    best_matched_frozen_nn_idx = true_task_id
                elif self.task_detector_type == PREDICT_METHOD_NW_CONFIDENCE:
                    best_matched_frozen_nn_idx = self.get_network_with_best_confidence(x, x_flatten)

                if 0 <= best_matched_frozen_nn_idx < len(self.frozen_nets):
                    self.frozen_nets[best_matched_frozen_nn_idx].chosen_for_test += r

                    if self.frozen_nets[best_matched_frozen_nn_idx].seen_task_ids_train.get(true_task_id) is not None:
                        self.correct_network_selected_count += r
                        if self.frozen_nets[best_matched_frozen_nn_idx].seen_task_ids_test.get(true_task_id) is None:
                            self.frozen_nets[best_matched_frozen_nn_idx].seen_task_ids_test[true_task_id] = r
                        else:
                            self.frozen_nets[best_matched_frozen_nn_idx].seen_task_ids_test[true_task_id] += r
                else:
                    if len(self.frozen_nets) >= 0:
                        print('Index error for best_matched_frozen_nn_index ({})'.format(best_matched_frozen_nn_idx))
                    else:
                        print('No frozen nets. may use best training net for prediction')

                if self.use_weights_from_task_detectors and (
                        self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or
                        self.task_detector_type == PREDICT_METHOD_ONE_CLASS):
                    if best_matched_frozen_nn_idx < 0:
                        print(
                            'No frozen nets. may use best training net for prediction for PREDICT_METHOD_NAIVE_BAYES or PREDICT_METHOD_ONE_CLASS')
                    final_votes = self.get_majority_vote_from_nets(
                        x, x_flatten, r,
                        weights_for_each_network=weights_for_frozen_nns,
                        add_best_training_nn_votes=True,
                        predictor=self.task_detector_type)
                else:
                    final_votes = \
                        self.frozen_nets[best_matched_frozen_nn_idx].net(x if self.frozen_nets[best_matched_frozen_nn_idx].network_type == NETWORK_TYPE_CNN else x_flatten)

            elif self.task_detector_type == PREDICT_METHOD_MAJORITY_VOTE:
                final_votes = self.get_majority_vote_from_nets(
                    x, x_flatten, r,
                    weights_for_each_network=None,
                    add_best_training_nn_votes=False,
                    predictor=None)

            correct_class_predicted = torch.eq(torch.argmax(final_votes, dim=1), y).sum().item()
            self.correct_class_predicted += correct_class_predicted

            if self.task_detector_type != PREDICT_METHOD_MAJORITY_VOTE:
                if self.use_weights_from_task_detectors:
                    pass
                else:
                    if best_matched_frozen_nn_idx >= 0:
                        self.frozen_nets[best_matched_frozen_nn_idx].correct_class_predicted += correct_class_predicted

            return final_votes

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

        self.train_nets.sort(key=lambda ann: ann.loss_estimator.estimation)
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
                t.append(threading.Thread(target=net_train, args=(self.train_nets[nn_index], xx, r, c, y, true_task_id, self.detected_task_id, use_instances_for_task_detector_training, self.use_one_class_probas,)))
            else:
                self.train_nets[nn_index].train_net(xx, y, c, r, true_task_id, self.detected_task_id, use_instances_for_task_detector_training, self.use_one_class_probas)
        if self.use_threads:
            for i in range(len(t)):
                t[i].start()
            for i in range(len(t)):
                t[i].join()

        nn_with_lowest_loss = self.get_train_nn_index_with_lowest_loss()

        self.task_detected = False
        for m in self.train_nets:
            if m.task_detected:
                self.task_detected = True
        if self.task_detected:
            self.add_nn_with_lowest_loss_to_frozen_list()
            self.reset_one_class_detectors_and_loss_estimators_seen_task_ids()

        if (self.train_task_predictor_at_the_end and
                self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_ONE_CLASS) and use_instances_for_task_detector_training:
            if self.accumulated_x[0] is None:
                self.accumulated_x[0] = x.cpu()
            else:
                self.accumulated_x[0] = torch.cat([self.accumulated_x[0], x.cpu()], dim=0)

        self.train_nets[nn_with_lowest_loss].chosen_after_train += r
        return self.train_nets[nn_with_lowest_loss].net(
            x if self.train_nets[nn_with_lowest_loss].network_type == NETWORK_TYPE_CNN else x_flatten)

    def print_stats_hader(self):
        if not self.heading_printed:
            print('training_exp,'
                  'dumped_at,'
                  'detected_task_id,'
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
                  'this_seen_task_ids_test,'
                  'correct_network_selected,'
                  'this_acc,'
                  'acc',
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
            print('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},"{}",{},{},{}'.format(
                self.training_exp,
                dumped_at,
                self.detected_task_id,
                list_type,
                self.total_samples_seen_for_train,
                self.samples_seen_for_train_after_drift,
                nn_l[i].model_name,
                nn_l[i].id,
                nn_l[i].samples_seen_at_train,
                nn_l[i].trained_count,
                0 if nn_l[i].samples_seen_at_train == 0 else nn_l[i].accumulated_loss / nn_l[i].samples_seen_at_train,
                nn_l[i].loss_estimator.estimation,
                nn_l[i].chosen_after_train,
                self.samples_seen_for_test,
                self.test_samples_seen_for_learned_tasks,
                nn_l[i].chosen_for_test,
                nn_l[i].seen_task_ids_test,
                self.correct_network_selected_count / self.test_samples_seen_for_learned_tasks * 100 if self.test_samples_seen_for_learned_tasks != 0 else 0.0,
                nn_l[i].correct_class_predicted / self.test_samples_seen_for_learned_tasks * 100 if self.test_samples_seen_for_learned_tasks != 0 else 0.0,
                self.correct_class_predicted / self.test_samples_seen_for_learned_tasks * 100 if self.test_samples_seen_for_learned_tasks != 0 else 0.0
            ), file=self.stats_file, flush=True)
