################################################################################
# Copyright (c) 2020 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-02-2021                                                             #
# Author(s): Tyler L. Hayes                                                    #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader

from typing_extensions import Literal

from avalanche.benchmarks.datasets import Stream51
from avalanche.benchmarks.scenarios.generic_benchmark_creation import \
    create_generic_benchmark_from_paths
from torchvision import transforms
import math
import os

from avalanche.benchmarks.utils import AvalancheDatasetType

_mu = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]
_default_stream51_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_mu,
                         std=_std)
])


def _adjust_bbox(img_shapes, bbox, ratio=1.1):
    """
    Adapts bounding box coordinates so that they can be used by
    torchvision.transforms.functional.crop function.

    This also pads each bounding box according to the `ratio` parameter.

    :param img_shapes: a list of shapes, with each element in the format
        "[img.shape[0], img.shape[1]]".
    :param bbox: A list of elements in the format "[right, left, top, bottom]".
    :param ratio: The amount of padding. Defaults to "1.1".

    :returns: A list of adapted bounding box coordinates.
    """
    cw = bbox[0] - bbox[1]
    ch = bbox[2] - bbox[3]
    center = [int(bbox[1] + cw / 2), int(bbox[3] + ch / 2)]
    bbox = [
        min([int(center[0] + (cw * ratio / 2)), img_shapes[0]]),
        max([int(center[0] - (cw * ratio / 2)), 0]),
        min([int(center[1] + (ch * ratio / 2)), img_shapes[1]]),
        max([int(center[1] - (ch * ratio / 2)), 0])]
    return [bbox[3], bbox[1], bbox[2] - bbox[3], bbox[0] - bbox[1]]


def CLStream51(
        *,
        scenario: Literal[
            'iid', 'class_iid', 'instance',
            'class_instance'] = "class_instance",
        seed=10,
        eval_num=None,
        bbox_crop=True,
        ratio: float = 1.10,
        download=True,
        train_transform=_default_stream51_transform,
        eval_transform=_default_stream51_transform,
        dataset_root: Union[str, Path] = None,
        no_novelity_detection=False):
    """
    Creates a CL benchmark for Stream-51.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    This generator can be used to obtain the 'iid', 'class_iid', 'instance', and
    'class_instance' scenarios.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Avalanche will support the
    "out of distribution" stream in the near future!

    Each Experience contains the `dataset` and the associated task label, which
    is always 0 for Stream51.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param scenario: A string defining which Stream-51 scenario to return.
        Can be chosen between 'iid', 'class_iid', 'instance', and
        'class_instance'. Defaults to 'class_instance'.
    :param bbox_crop: If True, crops the images by using the bounding boxes
        defined by Stream51. This is needed to ensure that images depict only
        the required object (for classification purposes). Defaults to True.
    :param ratio: A floating point value (>= 1.0) that controls the amount of
        padding for bounding boxes crop (default: 1.10).
    :param seed: Random seed for shuffling classes or instances. Defaults to 10.
    :param eval_num: How many samples to see before evaluating the network for
        instance ordering and how many classes to see before evaluating the
        network for the class_instance ordering. Defaults to None, which means
        that "30000" will be used for the 'instance' scenario and "10" for the
        'class_instance' scenario.
    :param download: If True, the dataset will automatically downloaded.
        Defaults to True.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default eval transformation
        will be used.
    :param dataset_root: The root path of the dataset.
        Defaults to None, which means that the default location for
        'stream51' will be used.

    :returns: A properly initialized :class:`GenericCLScenario` instance.
    """

    # get train and test sets and order them by benchmark
    train_set = Stream51(root=dataset_root, train=True, download=download)
    test_set = Stream51(root=dataset_root, train=False, download=download)
    samples = Stream51.make_dataset(train_set.samples, ordering=scenario,
                                    seed=seed)
    dataset_root = train_set.root

    # set appropriate train parameters
    train_set.samples = samples
    train_set.targets = [s[0] for s in samples]

    # compute number of tasks
    if eval_num is None and scenario == 'instance':
        eval_num = 30000
        num_tasks = math.ceil(
            len(train_set) / eval_num)  # evaluate every 30000 samples
    elif eval_num is None and scenario == 'class_instance':
        eval_num = 10
        num_tasks = math.ceil(
            51 / eval_num)  # evaluate every 10 classes
    elif scenario == 'instance':
        num_tasks = math.ceil(
            len(train_set) / eval_num)  # evaluate every eval_num samples
    else:
        num_tasks = math.ceil(
            51 / eval_num)  # evaluate every eval_num classes

    if scenario == 'instance':
        # break files into task lists based on eval_num samples
        train_filelists_paths = []
        start = 0
        for i in range(num_tasks):
            end = min(start + eval_num, len(train_set))
            train_filelists_paths.append(
                [(os.path.join(dataset_root, train_set.samples[j][-1]),
                  train_set.samples[j][0],
                  _adjust_bbox(train_set.samples[j][-3],
                               train_set.samples[j][-2],
                               ratio)) for j in
                 range(start, end)])
            start = end

        # use all test data for instance ordering
        test_filelists_paths = \
            [(os.path.join(dataset_root, test_set.samples[j][-1]),
              test_set.samples[j][0],
              _adjust_bbox(test_set.samples[j][-3],
                           test_set.samples[j][-2],
                           ratio)) for j in range(len(test_set))]
        test_ood_filelists_paths = None  # no ood testing for instance ordering
    elif scenario == 'class_instance':
        # break files into task lists based on classes
        train_filelists_paths = []
        test_filelists_paths = []
        test_ood_filelists_paths = []
        class_change = [i for i in range(1, len(train_set.targets)) if
                        train_set.targets[i] != train_set.targets[i - 1]]
        unique_so_far = []
        start = 0
        for i in range(num_tasks):
            if i == num_tasks - 1:
                end = len(train_set)
            else:
                end = class_change[
                    min(eval_num + eval_num * i - 1, len(class_change) - 1)]
            unique_labels = [train_set.targets[k] for k in range(start, end)]
            unique_labels = list(set(unique_labels))
            unique_so_far += unique_labels
            test_files = []
            test_ood_files = []
            for ix, test_label in enumerate(test_set.targets):
                if test_label in unique_so_far:
                    test_files.append(ix)
                else:
                    test_ood_files.append(ix)
            test_filelists_paths.append(
                [(os.path.join(dataset_root, test_set.samples[j][-1]),
                  test_set.samples[j][0],
                  _adjust_bbox(test_set.samples[j][-3], test_set.samples[j][-2],
                               ratio)) for j in
                 test_files])
            test_ood_filelists_paths.append(
                [(os.path.join(dataset_root, test_set.samples[j][-1]),
                  test_set.samples[j][0],
                  _adjust_bbox(test_set.samples[j][-3], test_set.samples[j][-2],
                               ratio)) for j in
                 test_ood_files])
            train_filelists_paths.append(
                [(os.path.join(dataset_root, train_set.samples[j][-1]),
                  train_set.samples[j][0],
                  _adjust_bbox(train_set.samples[j][-3],
                               train_set.samples[j][-2],
                               ratio)) for j in
                 range(start, end)])
            start = end
    else:
        raise NotImplementedError

    if not bbox_crop:
        # remove bbox coordinates from lists
        train_filelists_paths = [[[j[0], j[1]] for j in i] for i in
                                 train_filelists_paths]
        test_filelists_paths = [[[j[0], j[1]] for j in i] for i in
                                test_filelists_paths]
        if scenario == 'class_instance':
            test_ood_filelists_paths = [[[j[0], j[1]] for j in i] for i in
                                        test_ood_filelists_paths]
    if no_novelity_detection:
        intersection, remove_label_list = get_remove_label_lists(None, train_filelists_paths, test_filelists_paths,
                                                   complete_test_set_only=scenario == 'instance')
        remove_filepaths_with_label(remove_label_list, train_filelists_paths, test_filelists_paths,
                                    complete_test_set_only=scenario == 'instance')
        intersection, get_remove_label_lists(None, train_filelists_paths, test_filelists_paths,
                               complete_test_set_only=scenario == 'instance')


    benchmark_obj = create_generic_benchmark_from_paths(
        train_lists_of_files=train_filelists_paths,
        test_lists_of_files=test_filelists_paths,
        task_labels=[0 for _ in range(num_tasks)],
        complete_test_set_only=scenario == 'instance',
        train_transform=train_transform,
        eval_transform=eval_transform,
        dataset_type=AvalancheDatasetType.CLASSIFICATION)

    if no_novelity_detection:
        benchmark_obj.n_classes = len(intersection)
    return benchmark_obj


def remove_filepaths_with_label(remove_label_list, train_filelists_paths, test_filelists_paths, complete_test_set_only=False):
    for i in range(len(train_filelists_paths)):
        for l in remove_label_list['train']:
            j = 0
            while j < len(train_filelists_paths[i]):
                if train_filelists_paths[i][j][1] == l:
                    del train_filelists_paths[i][j]
                    continue
                j += 1

    if complete_test_set_only:
        for l in remove_label_list['test']:
            i = 0
            while i < len(test_filelists_paths):
                if test_filelists_paths[i][1] == l:
                    del test_filelists_paths[i]
                    continue
                i += 1
    else:
        for i in range(len(test_filelists_paths)):
            for l in remove_label_list['test']:
                j = 0
                while j < len(test_filelists_paths[i]):
                    if test_filelists_paths[i][j][1] == l:
                        del test_filelists_paths[i][j]
                        continue
                    j += 1

def get_remove_label_lists(scenario, train_filelists_paths, test_filelists_paths, complete_test_set_only=False):
    sets = {}
    sets['train'] = set()
    sets['test'] = set()
    if scenario is not None:
        for key, stream in scenario.streams.items():
            l = None
            for task_stream in stream:
                batch_size = len(task_stream.dataset)
                dl = DataLoader(task_stream.dataset, batch_size=batch_size)
                for x, y, t in dl:
                    # x, y, t = next(iter(dl))
                    # y = y.item()
                    sets[key].update(y.unique().tolist())
    else:
        for i in range(len(train_filelists_paths)):
            for j in range(len(train_filelists_paths[i])):
                y = train_filelists_paths[i][j][1]
                sets ['train'].add(y)
        if complete_test_set_only:
            for i in range(len(test_filelists_paths)):
                y = test_filelists_paths[i][1]
                sets['test'].add(y)
        else:
            for i in range(len(test_filelists_paths)):
                for j in range(len(test_filelists_paths[i])):
                    y = test_filelists_paths[i][j][1]
                    sets['test'].add(y)
    print('sets-------')
    print(sets)
    print('intersection-------')
    intersection = sets['train'].intersection(sets['test'])
    print(intersection)

    remove_label_list = {}
    print('difference-------------')
    remove_label_list['train'] = sets['train'].difference(sets['test'])
    remove_label_list['test'] = sets['test'].difference(sets['train'])
    print('remove list-------------')
    print(remove_label_list)
    return intersection, remove_label_list

__all__ = [
    'Stream51'
]

if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    benchmark = CLStream51(scenario="class_instance", seed=10,
                           bbox_crop=True)

    train_imgs_count = 0
    for i, batch in enumerate(benchmark.train_stream):
        print(i, batch)
        dataset, _ = batch.dataset, batch.task_label
        train_imgs_count += len(dataset)
        dl = DataLoader(dataset, batch_size=1)

        for j, mb in enumerate(dl):
            if j == 2:
                break
            x, y, *_ = mb

            # show a few un-normalized images from data stream
            # this code is for debugging purposes
            x_np = x[0, :, :, :].numpy().transpose(1, 2, 0)
            x_np = x_np * _std + _mu
            plt.imshow(x_np)
            plt.show()

            print(x.shape)
            print(y.shape)

    # make sure all of the training data was loaded properly
    assert train_imgs_count == 150736
