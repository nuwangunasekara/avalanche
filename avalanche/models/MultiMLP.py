import os.path
import random
import sys
import threading
from collections import OrderedDict
from collections import defaultdict
from typing import (
    # TYPE_CHECKING,
    Any,
    # ClassVar,
    Dict,
    List,
    # Optional,
    Tuple,
    # Type,
)
from random import sample

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
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models.quantization as models
# from torchsummary import summary

# import network_Gray_ResNet
from avalanche.models import network_Gray_ResNet

# from torchinfo import summary

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

TRAIN_FROZEN_NONE = 0
TRAIN_FROZEN_MOST_CONFIDENT = 1
TRAIN_FROZEN_ROUND_ROBIN = 2

NO_OF_CHANNELS = 3

augmentor = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()])


def lr_decay_or_increase(decay, lr, optimizer, num_iter, alpha = 1.0, loss_decreasing = False):
    factor = (decay ** num_iter)
    factor = factor if loss_decreasing else factor + alpha
    learn_rate = lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = learn_rate

def lr_decay_or_increase_2(decay, lr, optimizer, loss_at_start_of_drift, current_loss, alpha = 1.0, loss_decreasing = False):
    factor = current_loss/loss_at_start_of_drift
    factor = (decay ** factor)
    factor = factor if loss_decreasing else factor + alpha
    learn_rate = lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = learn_rate


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class NewBuffer:
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.I = {}
        self.attributes = ['examples', 'labels', 'task_labels']
        self.items_in_buffer = 0
        if self.buffer_size <= 0:
            return

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                     *attr.shape[1:]), dtype=typ, device=self.device))

    def add_item(self, x_i, y_i=None, t_i=None, index=None):
        self.examples[index] = x_i.to(self.device)
        self.labels[index] = y_i.to(self.device)
        self.task_labels[index] = t_i.to(self.device)
        t = t_i.item()
        if not t in self.I:
            self.I[t] = {}
        if not y_i.item() in self.I[t]:
            self.I[t][y_i.item()] = []
        self.I[t][y_i.item()].append(index)
        self.items_in_buffer += 1

    def add_or_replace_most_represented_tasks_most_represented_class(self, examples_i, labels_i=None, task_labels_i=None):
        if self.items_in_buffer < self.buffer_size:  # space available in the buffer
            index = self.items_in_buffer
            self.add_item(examples_i, labels_i, task_labels_i, index)
        else:  # replace the most representted task's most represented class's instance with this instance
            max_t = None
            max_t_count = 0
            for t in self.I.keys():
                n = 0
                for c in self.I[t].keys():
                    n += len(self.I[t][c])
                if max_t_count < n:
                    max_t_count = n
                    max_t = t
            max_t_c = None
            max_t_c_count = 0
            for c in self.I[max_t].keys():
                c_count = len(self.I[max_t][c])
                if max_t_c_count < c_count:
                    max_t_c = c
                    max_t_c_count = c_count
            j = random.randint(0, max_t_c_count - 1)
            index = self.I[max_t][max_t_c][j]

            del self.I[max_t][max_t_c][j]
            self.items_in_buffer -= 1
            self.add_item(examples_i, labels_i, task_labels_i, index)

    def add_data(self, examples, labels=None, task_labels=None):
        if self.buffer_size <= 0:
            return
        self.init_tensors(examples, labels, task_labels)

        for i in range(examples.shape[0]):
            # Reservoir sampling strategy
            use_example = False
            if self.num_seen_examples < self.buffer_size:
                use_example = True
            elif np.random.randint(0, self.num_seen_examples) < self.buffer_size:
                use_example = True
            self.num_seen_examples += 1
            if use_example:
                self.add_or_replace_most_represented_tasks_most_represented_class(examples[i], labels[i], task_labels[i])

    def data_available_for_task(self, task_id):
        if task_id in self.I:
           return True
        return False

    def return_tuples_for_chosen_indexes(self, chosen, transform: transforms = None, return_indexes=False):
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.to(self.device))
                                  for ee in self.examples[chosen]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[chosen],)
        if not return_indexes:
            return ret_tuple
        else:
            return ret_tuple + (chosen,)

    def get_data(self, size: int, transform: transforms = None, return_indexes=False, task_id=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if self.buffer_size <= 0:
            return

        if not self.data_available_for_task(task_id):
            return

        indexes = []
        for c in self.I[task_id].keys():
            for idx in self.I[task_id][c]:
                indexes.append(idx)
        size = min(len(indexes), size)
        chosen = sample(indexes, size)

        return self.return_tuples_for_chosen_indexes(chosen, transform, return_indexes)

    def get_data_from_all_tasks(self, size: int, transform: transforms = None, return_indexes=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if self.buffer_size <= 0:
            return

        indexes = []
        for t in self.I.keys():
            for c in self.I[t].keys():
                for idx in self.I[t][c]:
                    indexes.append(idx)
        size = min(len(indexes), size)
        chosen = sample(indexes, size)

        return self.return_tuples_for_chosen_indexes(chosen, transform, return_indexes)

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.items_in_buffer == 0:
            return True
        else:
            return False

    def to_string(self):
        s = ''
        for t in self.I.keys():
            ss = ''
            for c in self.I[t].keys():
                ss += '{}:{},'.format(c, len(self.I[t][c]))
            s += '{{{}: {{{}}},'.format(t, ss)
        return s


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']
        self.dict = {}

        if self.buffer_size <= 0:
            return

        # scores for lossoir
        self.importance_scores = torch.ones(self.buffer_size).to(self.device) * -float('inf')
        # scores for balancoir
        self.balance_scores = torch.ones(self.buffer_size).to(self.device) * -float('inf')
        # merged scores
        self.scores = torch.ones(self.buffer_size).to(self.device) * -float('inf')

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                     *attr.shape[1:]), dtype=typ, device=self.device))

    def merge_scores(self):
        scaling_factor = self.importance_scores.abs().mean() * self.balance_scores.abs().mean()
        norm_importance = self.importance_scores / scaling_factor
        presoftscores = 0.5 * norm_importance + 0.5 * self.balance_scores

        if presoftscores.max() - presoftscores.min() != 0:
            presoftscores = (presoftscores - presoftscores.min()) / (presoftscores.max() - presoftscores.min())
        self.scores = presoftscores / presoftscores.sum()

    def update_scores(self, indexes, values):
        self.importance_scores[indexes] = values

    def update_all_scores(self):
        self.balance_scores = torch.tensor([self.dict[x.item()] for x in self.labels]).float().to(self.device)

    def functionalReservoir(self, N, m):
        if N < m:
            return N

        rn = np.random.randint(0, N)
        if rn < m:
            self.update_all_scores()
            self.merge_scores()
            index = np.random.choice(range(m), p=self.scores.cpu().numpy(), size=1)
            return index
        else:
            return -1

    def add_data(self, examples, labels=None, logits=None, task_labels=None, loss_scores=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if self.buffer_size <= 0:
            return

        self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = self.functionalReservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    if self.num_seen_examples >= self.buffer_size:
                        self.dict[self.labels[index].item()] -= 1
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                self.importance_scores[index] = -float('inf') if loss_scores is None else loss_scores[i]
                if labels[i].item() in self.dict:
                    self.dict[labels[i].item()] += 1
                else:
                    self.dict[labels[i].item()] = 1

    def get_data(self, size: int, transform: transforms = None, return_indexes=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if self.buffer_size <= 0:
            return

        if size > self.num_seen_examples:
            size = self.num_seen_examples

        choice = np.random.choice(self.examples.shape[0], size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        if not return_indexes:
            return ret_tuple
        else:
            return ret_tuple + (choice,)

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, None)

    def to_string(self) -> str:
        return '[]'


class InstanceBuffer:
    def __init__(self, mem_size: int = 5):
        self.mem_size = mem_size
        self.ext_mem: Dict[Any, Tuple[List[Tensor], List[Tensor]]] = {}
        # count occurrences for each class
        self.counter: Dict[Any, Dict[Any, int]] = {}

    def add_items(self, x, y, t_id):
        if self.mem_size <= 0:
            return
        xx = x.detach().clone()
        yy = y.detach().clone()
        # for each pattern, add it to the memory or not
        for i in range(len(xx)):
            pattern = xx[i]
            task_id = t_id
            target = yy[i]
            target_value = target.item()

            if len(pattern.size()) == 1:
                pattern = pattern.unsqueeze(0)

            current_counter = self.counter.setdefault(task_id, defaultdict(int))
            current_mem = self.ext_mem.setdefault(task_id, ([], []))

            if current_counter == {}:
                # any positive (>0) number is ok
                patterns_per_class = 1
            else:
                patterns_per_class = int(
                    self.mem_size / len(current_counter.keys())
                )

            if (
                    target_value not in current_counter
                    or current_counter[target_value] < patterns_per_class
            ):
                # add new pattern into memory
                if sum(current_counter.values()) >= self.mem_size:
                    # full memory: replace item from most represented class
                    # with current pattern
                    to_remove = max(current_counter, key=current_counter.get)

                    dataset_size = len(current_mem[0])
                    for j in range(dataset_size):
                        if current_mem[1][j].item() == to_remove:
                            current_mem[0][j] = pattern
                            current_mem[1][j] = target.reshape(1)
                            break
                    current_counter[to_remove] -= 1
                else:
                    # memory not full: add new pattern
                    current_mem[0].append(pattern)
                    current_mem[1].append(target.reshape(1))

                # Indicate that we've changed the number of stored instances of
                # this class.
                current_counter[target_value] += 1

    def get_count(self, task_id):
        counts = self.counter.get(task_id, None)
        if counts is None:
            return 0
        else:
            return sum(counts.values())

    def to_string(self):
        str = '['
        if self.counter == {}:
            return str + ']'
        else:
            for key, counts in self.counter.items():
                str += '{} : {:.2f},'.format(key, sum(counts.values()) / self.mem_size)
            return str + ']'

    def get_all_task_buffer_sample(self, sample_size, device):
        task_count = len(self.counter.keys())
        if task_count == 0:
            return

        per_task_sample_size = sample_size // task_count
        last_task_sample_size = per_task_sample_size + (sample_size % task_count)
        task_set = list(self.counter.keys())

        xx = []
        yy = []
        for i in range(len(task_set) - 1):
            x, y = self.get_mixed_sample(None, None, task_set[i], only_from_buffer=True,
                                         only_from_buffer_size=per_task_sample_size, only_from_buffer_device=device)
            xx.append(x)
            yy.append(y)

        x, y = self.get_mixed_sample(None, None, task_set[-1], only_from_buffer=True,
                                     only_from_buffer_size=last_task_sample_size, only_from_buffer_device=device)
        xx.append(x)
        yy.append(y)

        return torch.cat(tuple(xx), 0), torch.cat(tuple(yy), 0)

    def get_mixed_sample(self, x, y, task_id, only_from_buffer=False, only_from_buffer_size=None,
                         only_from_buffer_device=None):
        if self.mem_size <= 0:
            return x, y

        if only_from_buffer and only_from_buffer_size is not None:
            device = only_from_buffer_device
            x_size = only_from_buffer_size
        else:
            device = x.device
            xx = x.detach().clone()
            yy = y.detach().clone()
            x_size = len(x)

        if self.ext_mem.get(task_id) is not None:
            patterns, targets = self.ext_mem[task_id]
            patterns = torch.stack(patterns)
            targets = torch.stack(targets)
            targets = targets.view(targets.shape[0], )

            buffer_size = len(patterns)
            if only_from_buffer:
                buffer_copy_size = min(x_size, buffer_size)
            else:
                buffer_copy_size = min(x_size // 2, buffer_size)
                replace_indices = sample(range(x_size), buffer_copy_size)
                replace_indices = torch.tensor(replace_indices).to(torch.long).to(device)

            indices_to_copy = sample(range(buffer_size), buffer_copy_size)
            indices_to_copy = torch.tensor(indices_to_copy).to(torch.long).to(device)

            if only_from_buffer:
                xx = torch.index_select(patterns, 0, indices_to_copy).detach().clone()
                yy = torch.index_select(targets, 0, indices_to_copy).detach().clone()
            else:
                try:
                    xx = xx.index_copy_(0, replace_indices,
                                        torch.index_select(patterns, 0, indices_to_copy).detach().clone())
                    yy = yy.index_copy_(0, replace_indices,
                                        torch.index_select(targets, 0, indices_to_copy).detach().clone())
                except Exception as e:
                    print('Replace exception: {}'.format(e))

        return xx.to(device), yy.to(device)


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
    else:  # 3_channel_data
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
    else:  # 3_channel_data
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


def train_one_class_classifier(features, one_class_detector, train_logistic_regression, logistic_regression,
                               scaler=None):
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
    transforms.Lambda(lambda x: x.repeat(1, NO_OF_CHANNELS, 1, 1)),  # repeat channel 1, 3 times
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.Pad(0, fill=3),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

repeat_channel_1 = transforms.Compose([
    # https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/9
    # expand channels
    transforms.Lambda(lambda x: x.repeat(1, NO_OF_CHANNELS, 1, 1)),  # repeat channel 1, 3 times
])


class SimpleCNN(nn.Module):

    def __init__(self, num_classes=10, num_channels=0):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x


class CNN4(nn.Module):
    def __init__(self, num_classes=10, num_channels=0):
        super(CNN4, self).__init__()

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1),
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

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
                 lr_decay=1.0,
                 network_type=None,
                 num_classes=None,
                 device='cpu',
                 optimizer_type=OP_TYPE_SGD,
                 # loss_f=nn.CrossEntropyLoss(),
                 loss_f=nn.functional.cross_entropy,
                 loss_estimator_delta=1e-3,
                 task_detector_type=PREDICT_METHOD_ONE_CLASS,
                 back_prop_skip_loss_threshold=0.6,
                 train_nn_using_ex_static_f=False,
                 cnn_type='SimpleCNN',
                 dl=True):
        # configuration variables (which has the same name as init parameters)
        self.id = id
        self.frozen_id = None
        self.frozen_index = -1
        self.model_name = None
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate
        self.network_type = network_type
        self.num_classes = num_classes
        self.device = device
        self.optimizer_type = optimizer_type
        self.loss_f = loss_f
        self.loss_estimator_delta = loss_estimator_delta
        self.back_prop_skip_loss_threshold = back_prop_skip_loss_threshold
        self.task_detector_type = task_detector_type
        self.current_loss = None
        self.train_nn_using_ex_static_f = train_nn_using_ex_static_f
        self.cnn_type = cnn_type
        self.dl = dl

        # status variables
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.loss_scores = None
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
        self.selected_for_prediction = 0
        self.correct_class_predicted = 0
        # self.input_dimensions = 0
        self.x_shape = None
        self.task_detected = False
        self.loss_decreasing = False
        self.loss_at_start_of_drift = None
        self.samples_seen_since_last_drift = 0
        self.seen_task_ids_train = {}
        self.correctly_predicted_task_ids_test = {}
        self.correctly_predicted_task_ids_test_at_last = {}
        self.correctly_predicted_task_ids_probas_test_at_last = {}
        self.outputs = None
        self.init_values()

    def init_values(self):
        # init status variables
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.loss_scores = None
        self.samples_seen_at_train = 0
        self.trained_count = 0
        self.chosen_for_test = 0
        self.chosen_after_train = 0
        self.loss_estimator = ADWIN(delta=self.loss_estimator_delta)
        self.accumulated_loss = 0
        self.one_class_detector = init_one_class_detector()
        self.scaler = init_scaler()
        self.logistic_regression = init_logistic_regression()
        self.selected_for_prediction = 0
        self.correct_class_predicted = 0
        # self.input_dimensions = 0
        self.x_shape = None
        self.task_detected = False
        self.loss_decreasing = False
        self.loss_at_start_of_drift = None
        self.samples_seen_since_last_drift = 0
        self.seen_task_ids_train = {}
        self.correctly_predicted_task_ids_test = {}
        self.correctly_predicted_task_ids_test_at_last = {}
        self.correctly_predicted_task_ids_probas_test_at_last = {}
        self.outputs = None

        self.model_name = '{}_{}_{:05f}'.format(
            self.cnn_type,
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
        # print('Network configuration:\n'
        #       '{}\n'
        #       '======================================='.format(self))

    def initialize_network(self):
        if self.network_type == NETWORK_TYPE_CNN:
            number_of_channels = self.x_shape[1]
            if self.cnn_type == 'SimpleCNN':
                self.net = SimpleCNN(num_classes=self.num_classes, num_channels=number_of_channels)
            elif self.cnn_type == 'CNN4':
                self.net = CNN4(num_classes=self.num_classes, num_channels=number_of_channels)
            else:
                print('Unsupported cnn type {}'.format(self.cnn_type))
                exit(0)
            # print_summary(self.net, self.x_shape)
            # print('Number of parameters: {}'.format(count_parameters(self.net)))
        else:
            pass
        self.initialize_net_para()

    def train_one_class_classifier(self, features, train_logistic_regression):
        train_one_class_classifier(features,
                                   self.one_class_detector,
                                   train_logistic_regression,
                                   self.logistic_regression,
                                   self.scaler)
        self.one_class_detector_fit_called = True

    def increment_counters_after_train(self, r, true_task_id):
        self.samples_seen_at_train += r

        if self.seen_task_ids_train.get(true_task_id) is None:
            self.seen_task_ids_train[true_task_id] = r
        else:
            self.seen_task_ids_train[true_task_id] += r

    def forward_pass(self, x, y, static_features=None):
        if self.train_nn_using_ex_static_f and static_features is not None:
            # xx = repeat_channel_1(static_features.view(static_features.shape[0], 64, -1)[:, None, :, :])
            xx = static_features.view(static_features.shape[0], 1, 64, -1)
        else:
            xx = x
        if self.net is None:
            self.x_shape = deepcopy(xx.shape)
            self.initialize_network()

        xx = xx.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()  # zero the gradient buffers
        # forward propagation
        outputs = self.net(xx)

        # if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
        #     self.train_one_class_classifier(static_features.cpu(), use_one_class_probas)

        # backward propagation
        # print(self.net.linear[0].weight.data)
        self.loss_scores = self.loss_f(outputs, y, reduction='none')
        self.loss = self.loss_scores.mean()
        self.current_loss = self.loss.item()
        self.outputs = outputs.detach()

        return outputs

    def update_loss_estimator(self):
        previous_estimated_loss = self.loss_estimator.estimation
        self.loss_estimator.add_element(self.current_loss)
        self.accumulated_loss += self.current_loss
        if self.loss_at_start_of_drift is None:
            self.loss_at_start_of_drift = self.current_loss
        if self.loss_estimator.detected_change():
            self.loss_at_start_of_drift = self.current_loss
            self.samples_seen_since_last_drift = 0
            if self.loss_estimator.estimation > previous_estimated_loss:
                print('NEW TASK detected by {} {} {}'.format(self.model_name, 'T' if self.frozen_id is None else 'F', self.frozen_id))
                self.task_detected = True
                self.loss_decreasing = False
            elif self.loss_estimator.estimation < previous_estimated_loss:
                print('Estd LOSS DECREASING in {} {} {}'.format(self.model_name, 'T' if self.frozen_id is None else 'F', self.frozen_id))
                self.loss_decreasing = True
                self.task_detected = False
        else:
            self.samples_seen_since_last_drift += 1

    def call_backprop(self):
        if self.dl:
            lr_decay_or_increase(self.lr_decay, self.learning_rate, self.optimizer, self.samples_seen_since_last_drift,
                                 alpha=1.0,
                                 loss_decreasing=self.loss_decreasing)
            # lr_decay_or_increase_2(self.lr_decay, self.learning_rate, self.optimizer,
            #                        self.loss_at_start_of_drift,
            #                        self.current_loss,
            #                        alpha=1.0,
            #                        loss_decreasing=self.loss_decreasing)
        if self.loss.item() > self.back_prop_skip_loss_threshold:
            self.loss.backward()
            self.optimizer.step()  # Does the update
            self.trained_count += 1

    def reset_loss_and_bp_buffers(self):
        self.optimizer.zero_grad()
        self.loss = None
        self.loss_scores = None

    @torch.no_grad()
    def get_votes(self, x, static_features=None):
        if self.train_nn_using_ex_static_f and static_features is not None:
            # xx = repeat_channel_1(static_features.view(static_features.shape[0], 64, -1)[:, None, :, :])
            xx = static_features.view(static_features.shape[0], 1, 64, -1)
        else:
            xx = x
        return self.net(xx.to(self.device)).unsqueeze(0)

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


def forward_pass_and_increment_counters(net: ANN, x: np.ndarray, y: np.ndarray, true_task_id,
              static_features):
    net.forward_pass(x, y, static_features=static_features)
    net.update_loss_estimator()
    net.increment_counters_after_train(x.shape[0], true_task_id)

@torch.no_grad()
def forward_pass_without_gradiants(net: ANN, x: np.ndarray, y: np.ndarray,
              static_features):
    net.forward_pass(x, y, static_features=static_features)


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


def load_model(abstract_model_file_name, nn_model_file_name, load_eval_mode=True):
    abstract_model: ANN = load(abstract_model_file_name)
    abstract_model.initialize_network()
    abstract_model.net.load_state_dict(torch.load(nn_model_file_name))
    if load_eval_mode:
        for param in abstract_model.net.parameters():
            param.requires_grad = False
        abstract_model.net.eval()
    else: # load model for training
        for param in abstract_model.net.parameters():
            param.requires_grad = True
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
                 use_1_channel_pretrained_for_1_channel=False,
                 use_quantized=False,
                 max_frozen_pool_size=-1,
                 instance_buffer_size_per_frozen_nw=200,
                 cnn_type='SimpleCNN',
                 lr_decay=1.0,
                 dl=True,
                 tf='N'
                 ):
        super().__init__()

        # configuration variables (which has the same name as init parameters)
        self.num_classes = num_classes
        self.use_threads = use_threads
        self.stats_file = stats_file
        self.number_of_instances_to_train_using_all_mlps_at_start = number_of_instances_to_train_using_all_mlps_at_start
        self.loss_estimator_delta = loss_estimator_delta
        self.stats_print_frequency = stats_print_frequency
        self.back_prop_skip_loss_threshold = back_prop_skip_loss_threshold
        self.lr_decay = lr_decay
        self.nn_pool_type = nn_pool_type
        self.task_detector_type = predict_method
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
        self.nb_stats_file = None
        self.train_nn_using_ex_static_f = train_nn_using_ex_static_f
        self.use_1_channel_pretrained_for_1_channel = use_1_channel_pretrained_for_1_channel
        self.use_quantized = use_quantized
        self.max_frozen_pool_size = max_frozen_pool_size
        self.instance_buffer_size_per_frozen_nw = instance_buffer_size_per_frozen_nw
        self.cnn_type = cnn_type
        self.dl = dl
        self.train_frozen = None
        if tf == 'N':
            self.train_frozen = TRAIN_FROZEN_NONE
        elif tf == 'MC':
            self.train_frozen = TRAIN_FROZEN_MOST_CONFIDENT
        elif tf == 'RR':
            self.train_frozen = TRAIN_FROZEN_ROUND_ROBIN
        else:
            print('Unhandled tf type: {}'.format(tf))
            exit(1)

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
        self.correct_nn_with_lowest_loss_predicted = defaultdict(int)
        self.correct_nn_with_lowest_loss_predicted_acc = defaultdict(int)
        self.samples_seen_for_test_by_task = defaultdict(int)
        self.call_predict = False
        self.mb_yy = None
        self.mb_task_id = None
        self.training_exp = 0
        self.available_nn_id = 0
        self.detected_task_id = 0
        self.instances_per_task_at_last = {}
        self.f_ex = None
        self.f_ex_device = None
        self.nb_or_ht = NaiveBayes() if self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES else HoeffdingTreeClassifier() if self.task_detector_type == PREDICT_METHOD_HT else None
        self.nb_preds = None
        self.buffer = None
        self.buffer_used_for_train_count = defaultdict(int)
        self.incoming_training_batch = 0
        self.last_trained_frozen_for_RR = 0
        self.correct_class_predicted_at_test_after_training_each_task = defaultdict(int)
        self.samples_seen_at_test_after_training_each_task = defaultdict(int)
        self.correct_class_predicted_at_test_acc_after_training_each_task = defaultdict(int)

        self.init_values()

    def init_values(self):
        # init status variables

        self.heading_printed = False
        self.one_class_stats_file = sys.stdout if self.stats_file is sys.stdout else \
            open(self.stats_file.replace('.csv', '_OC.csv'),
                 'w') if self.task_detector_type == PREDICT_METHOD_ONE_CLASS else sys.stdout
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
        elif self.nn_pool_type == '1CNN':
            neurons_in_log2_start_include = 10
            neurons_in_log2_stop_exclude = 11
            # l_rate = '0.0005'
            lr_denominator_in_log10_start_include = 4
            lr_denominator_in_log10_stop_exclude = 5

            optimizer_types = ([OP_TYPE_ADAM_NC])

        for number_of_neurons_in_log2 in range(neurons_in_log2_start_include, neurons_in_log2_stop_exclude):
            for lr_denominator_in_log10 in range(lr_denominator_in_log10_start_include,
                                                 lr_denominator_in_log10_stop_exclude):
                for optimizer_type in optimizer_types:
                    network_type = NETWORK_TYPE_CNN
                    tmp_ann = ANN(id=self.available_nn_id,
                                  learning_rate=5 / (10 ** lr_denominator_in_log10),
                                  lr_decay=self.lr_decay,
                                  network_type=network_type,
                                  optimizer_type=optimizer_type,
                                  loss_estimator_delta=self.loss_estimator_delta,
                                  back_prop_skip_loss_threshold=self.back_prop_skip_loss_threshold,
                                  task_detector_type=self.task_detector_type,
                                  num_classes=self.num_classes,
                                  device=self.device,
                                  train_nn_using_ex_static_f=self.train_nn_using_ex_static_f,
                                  cnn_type=self.cnn_type,
                                  dl=self.dl)
                    self.train_nets.append(tmp_ann)
                    self.available_nn_id += 1

    @torch.no_grad()
    def get_majority_vote_from_nets(self, x, mini_batch_size, weights_for_each_network=None,
                                    add_best_training_nn_votes=False, predictor=None, static_features=None):
        votes = None

        nn_list = self.frozen_nets

        for i in range(len(nn_list)):
            v = nn_list[i].get_votes(x, static_features)

            if weights_for_each_network is not None:
                v *= weights_for_each_network[i]

            if votes is None:
                votes = v
            else:
                votes = torch.cat((votes, v), dim=0)
            nn_list[i].chosen_for_test += mini_batch_size

        if add_best_training_nn_votes:
            i = self.get_nn_index_with_lowest_or_highest_loss(self.train_nets, use_estimated_loss=True)
            nn_list = self.train_nets
            v = nn_list[i].get_votes(x)

            if weights_for_each_network is not None:
                if votes is not None and nn_list[i].one_class_detector_fit_called:
                    w, _ = self.get_best_matched_nn_index_and_weights_via_predictor(predictor, [nn_list[i]],
                                                                                    static_features)
                    v *= w[i]

            if votes is None:
                votes = v
            else:
                votes = torch.cat((votes, v), dim=0)
            nn_list[i].chosen_for_test += mini_batch_size

        return torch.mean(votes, dim=0)

    def get_network_with_best_confidence(self, x):
        predictions = None
        for i in range(len(self.frozen_nets)):
            xx = x
            if i == 0:
                predictions = self.frozen_nets[i].net(xx).unsqueeze(0)
            else:
                predictions = torch.cat((predictions, self.frozen_nets[i].net(xx).unsqueeze(0)), dim=0)
        return np.argmax(predictions.sum(axis=1).sum(axis=1), axis=0)

    def get_one_class_probas_for_nn(self, n, static_features=None):
        one_class_p = None
        one_class_df = None

        xxx = static_features.cpu().numpy()
        one_class_y = [0.0]
        if n.one_class_detector_fit_called:
            one_class_y = n.one_class_detector.predict(xxx)
            one_class_df = n.one_class_detector.decision_function(xxx)
            one_class_df = one_class_df.reshape(-1, 1)
            one_class_p = n.logistic_regression.predict_proba(one_class_df)
            one_class_p = one_class_p[:, 1]  # get probabilities for class 1 (inlier)

        return one_class_df, one_class_y, one_class_p

    def get_nb_or_ht_predicted_nn_index(self, static_features):
        predictions = self.nb_or_ht_predict(static_features)
        index = np.argmax(predictions.mean(axis=0), axis=0).item()
        confidence = predictions.mean(axis=0)[index]
        return index, confidence

    def get_best_matched_nn_index_and_weights_via_predictor(self, predictor, net_list, static_features=None):
        if predictor == PREDICT_METHOD_ONE_CLASS:
            predictions = []
            for i in range(len(net_list)):
                one_class_df, one_class_y, one_class_p = self.get_one_class_probas_for_nn(
                    net_list[i],
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
            predictions = self.nb_or_ht_predict(static_features)
            best_matched_frozen_nn_index = np.argmax(predictions.mean(axis=0), axis=0).item()
            weights_for_each_network = np.mean(predictions, axis=0) + self.vote_weight_bias

        return weights_for_each_network, best_matched_frozen_nn_index

    def nb_train(self, features, task_id):
        train_nb(features, self.nb_or_ht, task_id)

    def nb_or_ht_predict(self, static_features):
        return self.nb_or_ht.predict_proba(static_features.detach().cpu().numpy())

    def get_nn_index_with_lowest_or_highest_loss(self, nn_list, use_estimated_loss=True, lowest_loss=True):
        idx = 0
        min_loss = float("inf")
        max_loss = float('-inf')
        for i in range(len(nn_list)):
            tmp_loss = nn_list[i].loss_estimator.estimation if use_estimated_loss else nn_list[i].current_loss
            if lowest_loss:
                if tmp_loss < min_loss:
                    idx = i
                    min_loss = tmp_loss
            else:  # highest loss
                if tmp_loss > max_loss:
                    idx = i
                    max_loss = tmp_loss

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
        best_model.frozen_index = len(self.frozen_net_module_paths)
        save_model(best_model, abstract_model_file_name, nn_model_file_name, preserve_net=True)
        best_model.frozen_id = None
        best_model.frozen_index = -1
        best_model.outputs = outputs

        self.frozen_net_module_paths.append({'abstract_model_file_name': abstract_model_file_name,
                                             'nn_model_file_name': nn_model_file_name})

    def clear_frozen_pool(self):
        if len(self.frozen_nets) == 0:
            return
        try:
            for i in range(len(self.frozen_nets)):
                save_model(self.frozen_nets[i],
                           self.frozen_net_module_paths[i]['abstract_model_file_name'],
                           self.frozen_net_module_paths[i]['nn_model_file_name'])
                self.frozen_nets[i] = None
        except:
            print("Exception thrown. frozen_nets:\n{}", self.frozen_nets)
            print("Exception thrown. frozen_net_module_paths:\n{}", self.frozen_net_module_paths)
            print("Exception thrown. self:\n{}", self)

        self.frozen_nets = []

    def load_frozen_pool(self, load_eval_mode=True):
        if len(self.frozen_nets) != 0:
            # print('Frozen pool is not empty')
            return

        for i in range(len(self.frozen_net_module_paths)):
            self.frozen_nets.append(load_model(self.frozen_net_module_paths[i]['abstract_model_file_name'],
                                               self.frozen_net_module_paths[i]['nn_model_file_name'],
                                               load_eval_mode=load_eval_mode))

    def reset_training_pool_f(self):
        if self.reset_training_pool:
            for i in range(len(self.train_nets)):
                del self.train_nets[i]  # clear chosen net's memory
            self.train_nets = []  # type: List[ANN]
            self.create_nn_pool()

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
        idx = self.get_nn_index_with_lowest_or_highest_loss(self.train_nets, use_estimated_loss=True)
        self.print_nn_list([self.train_nets[idx]], list_type='train_net', dumped_at='task_detect')
        self.save_best_model_and_append_to_paths(idx)
        self.detected_task_id += 1

    def save_nb_predictions(self):
        if self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_HT:
            if self.nb_stats_file is None:
                suffix = '_{}'.format(str(self.training_exp - 1))
                suffix += '_NB' if self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES else '_HT'
                self.nb_stats_file = self.stats_file.name.replace('.csv', suffix)
                np.save(self.nb_stats_file, self.nb_preds)
                self.nb_stats_file = None
                self.nb_preds = None

    def check_accuracy_of_nb(self, x):
        for i in range(len(x)):
            static_features = self.get_static_features(x[None, i, :], self.f_ex, fx_device=self.f_ex_device)
            p = self.nb_or_ht_predict(static_features)

            p_row = np.concatenate((p, self.mb_task_id[i].cpu().numpy().reshape((1, 1))), axis=1)
            if self.nb_preds is None:
                self.nb_preds = p_row
            else:
                self.nb_preds = np.concatenate((self.nb_preds, p_row))

    def check_accuracy_of_one_class_classifiers(self, x):
        if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
            for i in range(len(x)):
                static_features = self.get_static_features(x, self.f_ex, fx_device=self.f_ex_device)
                task_id = self.mb_task_id[i].item()
                if self.instances_per_task_at_last.get(task_id) is None:
                    self.instances_per_task_at_last[task_id] = 1
                else:
                    self.instances_per_task_at_last[task_id] += 1
                for j in range(len(self.frozen_nets)):
                    is_nw_trained_on_task_id = 0 if self.frozen_nets[j].seen_task_ids_train.get(task_id) is None else 1

                    one_class_df, one_class_y, one_class_p = self.get_one_class_probas_for_nn(
                        self.frozen_nets[j],
                        static_features)
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
                            self.frozen_nets[j].correctly_predicted_task_ids_probas_test_at_last[
                                task_id] = one_class_p.item()
                        else:
                            self.frozen_nets[j].correctly_predicted_task_ids_probas_test_at_last[
                                task_id] += one_class_p.item()

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

    def forward_pass_and_increment(self, nn_list, **kwargs):
        xx = kwargs["xx"]
        y = kwargs["y"]
        true_task_id = kwargs["true_task_id"]
        static_features = kwargs["static_features"]
        t = []
        for i in range(len(nn_list)):
            if self.use_threads:
                t.append(threading.Thread(target=forward_pass_and_increment_counters, args=(
                    nn_list[i], xx[i], y[i], true_task_id, static_features,)))
            else:
                nn_list[i].forward_pass(xx[i], y[i], static_features=static_features)
                nn_list[i].update_loss_estimator()
                nn_list[i].increment_counters_after_train(xx[i].shape[0], true_task_id)
        if self.use_threads:
            for i in range(len(t)):
                t[i].start()
            for i in range(len(t)):
                t[i].join()

    @torch.no_grad()
    def forward_pass_without_grad(self, nn_list, **kwargs):
        xx = kwargs["xx"]
        y = kwargs["y"]
        static_features = kwargs["static_features"]
        t = []
        for i in range(len(nn_list)):
            if self.use_threads:
                t.append(threading.Thread(target=forward_pass_without_gradiants, args=(
                    nn_list[i], xx, y, static_features,)))
            else:
                nn_list[i].forward_pass(xx, y, static_features=static_features)
        if self.use_threads:
            for i in range(len(t)):
                t[i].start()
            for i in range(len(t)):
                t[i].join()

    @torch.no_grad()
    def increment_NW_accuraccy_counters_for_test(self, nn_list, x=None, y=None):
        for i in range(len(nn_list)):
            net = nn_list[i]
            self.correct_class_predicted_at_test_after_training_each_task['t{}_f{}'.format(self.training_exp - 1, net.frozen_index)] += (
                    net.outputs.argmax(1) == y).type(torch.float).sum().item()
            self.samples_seen_at_test_after_training_each_task['t{}_f{}'.format(self.training_exp - 1, net.frozen_index)] += x.shape[0]

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

    def calculate_nw_accuracy(self):
        for k in self.correct_nn_with_lowest_loss_predicted.keys():
            self.correct_nn_with_lowest_loss_predicted_acc[k] = 0.0 if self.samples_seen_for_test_by_task[k] == 0.0 else self.correct_nn_with_lowest_loss_predicted[k] / self.samples_seen_for_test_by_task[k]

        for k in self.correct_class_predicted_at_test_after_training_each_task.keys():
            self.correct_class_predicted_at_test_acc_after_training_each_task[k] = 0.0 if self.samples_seen_at_test_after_training_each_task[k] == 0.0 else self.correct_class_predicted_at_test_after_training_each_task[k] / self.samples_seen_at_test_after_training_each_task[k]

    def forward(self, x):
        r = x.shape[0]
        y = self.mb_yy
        true_task_id = self.mb_task_id.sum(dim=0).item() // self.mb_task_id.shape[0]
        static_features = None
        # if x.shape[1] < 3:
        #     x = repeat_channel_1(x)
        if self.use_static_f_ex:
            if self.f_ex is None:
                self.f_ex_device, self.f_ex = create_static_feature_extractor(
                    device=self.device,
                    use_single_channel_fx=False,
                    quantize=self.use_quantized
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
                        self.get_best_matched_nn_index_and_weights_via_predictor(
                            self.task_detector_type, self.frozen_nets, static_features)
                    # if self.training_exp == self.n_experiences:
                    if self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                        self.check_accuracy_of_one_class_classifiers(x)
                    elif self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_HT:
                        self.check_accuracy_of_nb(x)

                elif self.task_detector_type == PREDICT_METHOD_RANDOM:
                    best_matched_frozen_nn_idx = random.randrange(0, len(self.frozen_nets))
                elif self.task_detector_type == PREDICT_METHOD_TASK_ID_KNOWN:
                    best_matched_frozen_nn_idx = true_task_id
                elif self.task_detector_type == PREDICT_METHOD_NW_CONFIDENCE:
                    best_matched_frozen_nn_idx = self.get_network_with_best_confidence(x)

                if (0 <= best_matched_frozen_nn_idx < len(self.frozen_nets)):
                    train_nn_list = self.frozen_nets

                    train_nn_list[best_matched_frozen_nn_idx].chosen_for_test += r

                    if train_nn_list[best_matched_frozen_nn_idx].seen_task_ids_train.get(true_task_id) is not None:
                        self.correct_network_selected_count += r
                        if train_nn_list[best_matched_frozen_nn_idx].correctly_predicted_task_ids_test.get(
                                true_task_id) is None:
                            train_nn_list[best_matched_frozen_nn_idx].correctly_predicted_task_ids_test[
                                true_task_id] = r
                        else:
                            train_nn_list[best_matched_frozen_nn_idx].correctly_predicted_task_ids_test[
                                true_task_id] += r

                if self.use_weights_from_task_detectors and (
                        self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or
                        self.task_detector_type == PREDICT_METHOD_HT or
                        self.task_detector_type == PREDICT_METHOD_ONE_CLASS):
                    if best_matched_frozen_nn_idx < 0:
                        print(
                            'No frozen nets. may use best training net for prediction for PREDICT_METHOD_NAIVE_BAYES or PREDICT_METHOD_HT or PREDICT_METHOD_ONE_CLASS')
                    final_votes = self.get_majority_vote_from_nets(
                        x, r,
                        weights_for_each_network=weights_for_frozen_nns,
                        add_best_training_nn_votes=True if self.auto_detect_tasks and (
                                len(self.frozen_nets) == 0 or best_matched_frozen_nn_idx == len(
                            self.frozen_nets)) else False,
                        predictor=self.task_detector_type,
                        static_features=static_features)
                else:  # No weights from predictors
                    final_votes = \
                        self.frozen_nets[best_matched_frozen_nn_idx].net(x)

            elif self.task_detector_type == PREDICT_METHOD_MAJORITY_VOTE:
                final_votes = self.get_majority_vote_from_nets(
                    x, r,
                    weights_for_each_network=None,
                    add_best_training_nn_votes=False,
                    predictor=None,
                    static_features=static_features)

            correct_class_predicted = torch.eq(torch.argmax(final_votes, dim=1), y).sum().item()
            self.correct_class_predicted += correct_class_predicted

            self.forward_pass_without_grad(self.frozen_nets,
                                            xx=x,
                                            y=y,
                                            static_features=None)
            self.increment_NW_accuraccy_counters_for_test(self.frozen_nets, x=x, y=y)

            nn_with_lowest_loss_at_pred = self.get_nn_index_with_lowest_or_highest_loss(self.frozen_nets, use_estimated_loss=False, lowest_loss=True)
            self.samples_seen_for_test_by_task['t{}_e{}'.format(self.training_exp-1, true_task_id)] += r
            if self.task_detector_type != PREDICT_METHOD_MAJORITY_VOTE:
                if nn_with_lowest_loss_at_pred == best_matched_frozen_nn_idx == true_task_id:
                    self.correct_nn_with_lowest_loss_predicted['t{}_e{}'.format(self.training_exp-1, true_task_id)] += r
                if self.use_weights_from_task_detectors:
                    pass
                else:
                    if best_matched_frozen_nn_idx >= 0:
                        self.frozen_nets[best_matched_frozen_nn_idx].selected_for_prediction += r
                        self.frozen_nets[best_matched_frozen_nn_idx].correct_class_predicted += correct_class_predicted

            return final_votes

        else:  # train
            self.incoming_training_batch += 1
            self.samples_seen_for_train_after_dd += r
            self.total_samples_seen_for_train += r
            if self.samples_per_each_task_at_train.get(true_task_id) is None:
                self.samples_per_each_task_at_train[true_task_id] = r
            else:
                self.samples_per_each_task_at_train[true_task_id] += r

        c = None
        train_nn_list = []
        x_to_use = []
        y_to_use = []
        frozen_pool_full = False
        if self.max_frozen_pool_size > 0 and len(self.frozen_net_module_paths) >= self.max_frozen_pool_size:
            detected_task_id, detected_task_id_confidence = self.get_nb_or_ht_predicted_nn_index(static_features)
            nw_id = detected_task_id
            frozen_pool_full = True
        else:  # max_frozen_pool_size is infinite or frozen pool is not fully filled
            if self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_HT:
                detected_task_id, detected_task_id_confidence = self.get_nb_or_ht_predicted_nn_index(static_features)
            elif self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
                pass
            else:
                detected_task_id = self.detected_task_id
            nw_id = self.detected_task_id

        if self.buffer is None:
            self.buffer = NewBuffer(self.instance_buffer_size_per_frozen_nw, x.device)

        trained_frozen_index = []
        if frozen_pool_full:
            self.load_frozen_pool(load_eval_mode=False)
            train_nn_list.append(self.frozen_nets[nw_id])
            trained_frozen_index.append(nw_id)
            x_to_use.append(x)
            y_to_use.append(y)
        else:
            # train training NW/s
            for i in range(len(self.train_nets)):
                train_nn_list.append(self.train_nets[i])
                x_to_use.append(x)
                y_to_use.append(y)

        # train frozen
        if self.train_frozen != TRAIN_FROZEN_NONE and len(self.frozen_net_module_paths) > 0:
            self.load_frozen_pool(load_eval_mode=False)
            frozen_indexes_to_train = []
            if self.train_frozen == TRAIN_FROZEN_MOST_CONFIDENT and detected_task_id < len(self.frozen_net_module_paths):
                frozen_indexes_to_train = [detected_task_id]
            elif self.train_frozen == TRAIN_FROZEN_ROUND_ROBIN and len(self.frozen_net_module_paths) > 0:
                if self.last_trained_frozen_for_RR + 1 < len(self.frozen_net_module_paths):
                    self.last_trained_frozen_for_RR += 1
                else:
                    self.last_trained_frozen_for_RR = 0
                frozen_indexes_to_train = [self.last_trained_frozen_for_RR]
            # frozen_indexes_to_train = [sample(frozen_indexes_to_train, 1)[0]]  # random train frozen
            # frozen_indexes_to_train = [i for i in range(len(self.frozen_nets))] # train all
            # remove trained_frozen_index from  frozen_indexes_to_train
            frozen_indexes_to_train = list(set(frozen_indexes_to_train).difference(set(trained_frozen_index)))
            for i in frozen_indexes_to_train:
                self.buffer_used_for_train_count['f_{}'.format(i)] += 1
                train_nn_list.append(self.frozen_nets[i])
                if self.buffer.data_available_for_task(i):
                    buf_x, buf_y, _, buf_indx = self.buffer.get_data(
                        r,  # batch size
                        transform=None,
                        return_indexes=True,
                        task_id=i)
                    x_to_use.append(buf_x)
                    y_to_use.append(buf_y)

        if self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or \
                self.task_detector_type == PREDICT_METHOD_HT or \
                self.task_detector_type == PREDICT_METHOD_ONE_CLASS:
            if self.use_static_f_ex:
                if self.task_detector_type == PREDICT_METHOD_NAIVE_BAYES or self.task_detector_type == PREDICT_METHOD_HT:
                    self.nb_train(static_features, nw_id)

        self.forward_pass_and_increment(train_nn_list,
                          xx=x_to_use,
                          y=y_to_use,
                          true_task_id=true_task_id,
                          static_features=None)

        nn_with_lowest_loss = self.get_nn_index_with_lowest_or_highest_loss(train_nn_list, use_estimated_loss=True)
        nn_with_highest_loss = self.get_nn_index_with_lowest_or_highest_loss(train_nn_list, use_estimated_loss=False,
                                                                             lowest_loss=False)
        self.call_backprop(train_nn_list)
        loss_scores = train_nn_list[nn_with_highest_loss].loss_scores
        # if not self.buffer.is_empty():
        #     self.buffer.update_scores(buf_indexes, -loss_scores.detach()[r:])
        self.buffer.add_data(examples=x,
                             labels=y,
                             task_labels=torch.ones_like(y)*nw_id
                             )
        self.reset_loss_and_bp_buffers(train_nn_list)

        train_nn_list[nn_with_lowest_loss].chosen_after_train += r
        outputs = train_nn_list[nn_with_lowest_loss].outputs.clone()

        self.clear_frozen_pool()

        if self.auto_detect_tasks:
            task_detected = False
            for m in train_nn_list:
                if m.task_detected and m.samples_seen_since_last_drift == 0:
                    task_detected = True
            if task_detected:
                self.samples_seen_for_train_after_dd = 0
                self.add_to_frozen_paths()
                self.reset_training_pool_f()

        return outputs[:r]

    def add_to_frozen_paths(self):
        if self.max_frozen_pool_size == -1 or (
                self.max_frozen_pool_size > 0 and len(self.frozen_net_module_paths) < self.max_frozen_pool_size):
            self.add_nn_with_lowest_loss_to_frozen_list()
            self.reset_one_class_detectors_and_loss_estimators_seen_task_ids()
            # self.reset()

    def print_stats_hader(self):
        print('Training pool size {}'.format(len(self.train_nets)))
        print('Frozen pool size {}'.format(len(self.frozen_net_module_paths)))
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
                  'this_selected_for_prediction,'
                  'correct_class_predicted,'
                  'buffer,'
                  'buffer_used_for_train_count,'
                  'incoming_training_batch,'
                  'correct_nn_with_lowest_loss_predicted_acc,'
                  'correct_class_predicted_at_test_acc_after_training_each_task',
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
            print('{},{},{},{},{},"{}",{},{},{},{},{},{},"{}",{},{},{},{},{},"{}","{}","{}",{},{},"{}",{},{},{},"{}","{}",{},"{}","{}"'.format(
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
                nn_l[i].selected_for_prediction,
                self.correct_class_predicted,
                self.buffer.to_string(),
                self.buffer_used_for_train_count,
                self.incoming_training_batch,
                self.correct_nn_with_lowest_loss_predicted_acc,
                self.correct_class_predicted_at_test_acc_after_training_each_task
            ), file=self.stats_file, flush=True)
