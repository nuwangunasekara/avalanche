################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 26-06-2020                                                             #
# Author(s): Gabriele Graffieti                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from pathlib import Path
from typing import Optional, Sequence, Union, Any
import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize, \
    RandomRotation
import numpy as np

from avalanche.benchmarks import NCScenario, nc_benchmark, dataset_benchmark
from avalanche.benchmarks.classic.classic_benchmarks_utils import \
    check_vision_benchmark
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.utils import AvalancheDataset

_default_mnist_train_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

_default_mnist_eval_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])


class PixelsPermutation(object):
    """
    Apply a fixed permutation to the pixels of the given image.

    Works with both Tensors and PIL images. Returns an object of the same type
    of the input element.
    """

    def __init__(self, index_permutation: Sequence[int]):
        self.permutation = index_permutation
        self._to_tensor = ToTensor()
        self._to_image = ToPILImage()

    def __call__(self, img: Union[Image, Tensor]):
        is_image = isinstance(img, Image)
        if (not is_image) and (not isinstance(img, Tensor)):
            raise ValueError('Invalid input: must be a PIL image or a Tensor')

        if is_image:
            img = self._to_tensor(img)

        img = img.view(-1)[self.permutation].view(*img.shape)

        if is_image:
            img = self._to_image(img)

        return img


def SplitMNIST(
        n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None):
    """
    Creates a CL benchmark using the MNIST dataset.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of incremental experiences in the current
        benchmark.
        The value of this parameter should be a divisor of 10.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to false.
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
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)

    if return_task_id:
        return nc_benchmark(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_benchmark(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform)


def PermutedMNIST(
        n_experiences: int,
        *,
        seed: Optional[int] = None,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None) -> NCScenario:
    """
    Creates a Permuted MNIST benchmark.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    Random pixel permutations are used to permute the MNIST images in
    ``n_experiences`` different manners. This means that each experience is
    composed of all the original 10 MNIST classes, but the pixel in the images
    are permuted in a different way.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    A progressive task label, starting from "0", is applied to each experience.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences (tasks) in the current
        benchmark. It indicates how many different permutations of the MNIST
        dataset have to be created.
        The value of this parameter should be a divisor of 10.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param train_transform: The transformation to apply to the training data
        before the random permutation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data
        before the random permutation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    list_train_dataset = []
    list_test_dataset = []
    rng_permute = np.random.RandomState(seed)

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)

    # for every incremental experience
    for _ in range(n_experiences):
        # choose a random permutation of the pixels in the image
        idx_permute = torch.from_numpy(rng_permute.permutation(784)).type(
            torch.int64)

        permutation = PixelsPermutation(idx_permute)

        permutation_transforms = dict(
            train=(permutation, None),
            eval=(permutation, None)
        )

        # Freeze the permutation
        permuted_train = AvalancheDataset(
            mnist_train,
            transform_groups=permutation_transforms,
            initial_transform_group='train').freeze_transforms()

        permuted_test = AvalancheDataset(
            mnist_test,
            transform_groups=permutation_transforms,
            initial_transform_group='eval').freeze_transforms()

        list_train_dataset.append(permuted_train)
        list_test_dataset.append(permuted_test)

    return nc_benchmark(
        list_train_dataset,
        list_test_dataset,
        n_experiences=len(list_train_dataset),
        task_labels=True,
        shuffle=False,
        class_ids_from_zero_in_each_exp=True,
        one_dataset_per_exp=True,
        train_transform=train_transform,
        eval_transform=eval_transform)


def RotatedMNIST(
        n_experiences: int,
        *,
        seed: Optional[int] = None,
        rotations_list: Optional[Sequence[int]] = None,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None) -> NCScenario:
    """
    Creates a Rotated MNIST benchmark.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    Random angles are used to rotate the MNIST images in ``n_experiences``
    different manners. This means that each experience is composed of all the
    original 10 MNIST classes, but each image is rotated in a different way.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    A progressive task label, starting from "0", is applied to each experience.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences (tasks) in the current
        benchmark. It indicates how many different rotations of the MNIST
        dataset have to be created.
        The value of this parameter should be a divisor of 10.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param rotations_list: A list of rotations values in degrees (from -180 to
        180) used to define the rotations. The rotation specified in position
        0 of the list will be applied to the task 0, the rotation specified in
        position 1 will be applied to task 1 and so on.
        If None, value of ``seed`` will be used to define the rotations.
        If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param train_transform: The transformation to apply to the training data
        after the random rotation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data
        after the random rotation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    if rotations_list is not None and len(rotations_list) != n_experiences:
        raise ValueError("The number of rotations should match the number"
                         " of incremental experiences.")

    if rotations_list is not None and any(180 < rotations_list[i] < -180
                                          for i in range(len(rotations_list))):
        raise ValueError("The value of a rotation should be between -180"
                         " and 180 degrees.")

    list_train_dataset = []
    list_test_dataset = []
    rng_rotate = np.random.RandomState(seed)

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)

    # for every incremental experience
    for exp in range(n_experiences):
        if rotations_list is not None:
            rotation_angle = rotations_list[exp]
        else:
            # choose a random rotation of the pixels in the image
            rotation_angle = rng_rotate.randint(-180, 181)

        rotation = RandomRotation(degrees=(rotation_angle, rotation_angle))

        rotation_transforms = dict(
            train=(rotation, None),
            eval=(rotation, None)
        )

        # Freeze the rotation
        rotated_train = AvalancheDataset(
            mnist_train,
            transform_groups=rotation_transforms,
            initial_transform_group='train').freeze_transforms()

        rotated_test = AvalancheDataset(
            mnist_test,
            transform_groups=rotation_transforms,
            initial_transform_group='eval').freeze_transforms()

        list_train_dataset.append(rotated_train)
        list_test_dataset.append(rotated_test)

    return nc_benchmark(
        list_train_dataset,
        list_test_dataset,
        n_experiences=len(list_train_dataset),
        task_labels=True,
        shuffle=False,
        class_ids_from_zero_in_each_exp=True,
        one_dataset_per_exp=True,
        train_transform=train_transform,
        eval_transform=eval_transform)


def RotatedMNIST_di(
        n_experiences: int,
        *,
        seed: Optional[int] = None,
        rotations_list: Optional[Sequence[int]] = None,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None) -> NCScenario:
    """
    Creates a Rotated MNIST .

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    Random angles are used to rotate the MNIST images in ``n_experiences``
    different manners. This means that each experience is composed of all the
    original 10 MNIST classes, but each image is rotated in a different way.

    The  instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    A progressive task label, starting from "0", is applied to each experience.

    The  API is quite simple and is uniform across all 
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences (tasks) in the current
        . It indicates how many different rotations of the MNIST
        dataset have to be created.
        The value of this parameter should be a divisor of 10.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param rotations_list: A list of rotations values in degrees (from -180 to
        180) used to define the rotations. The rotation specified in position
        0 of the list will be applied to the task 0, the rotation specified in
        position 1 will be applied to task 1 and so on.
        If None, value of ``seed`` will be used to define the rotations.
        If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param train_transform: The transformation to apply to the training data
        after the random rotation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data
        after the random rotation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    if rotations_list is not None and len(rotations_list) != n_experiences:
        raise ValueError("The number of rotations should match the number"
                         " of incremental experiences.")

    if rotations_list is not None and any(180 < rotations_list[i] < -180
                                          for i in range(len(rotations_list))):
        raise ValueError("The value of a rotation should be between -180"
                         " and 180 degrees.")

    list_train_dataset = []
    list_test_dataset = []
    rng_rotate = np.random.RandomState(seed)

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)

    # for every incremental experience
    for exp in range(n_experiences):
        if rotations_list is not None:
            rotation_angle = rotations_list[exp]
        else:
            # choose a random rotation of the pixels in the image
            rotation_angle = rng_rotate.randint(-180, 181)

        rotation = RandomRotation(degrees=(rotation_angle, rotation_angle))

        rotation_transforms = dict(
            train=(rotation, None),
            eval=(rotation, None)
        )

        # Freeze the rotation
        rotated_train = AvalancheDataset(
            mnist_train,
            task_labels=exp,
            transform_groups=rotation_transforms,
            initial_transform_group='train').freeze_transforms()

        rotated_test = AvalancheDataset(
            mnist_test,
            task_labels=exp,
            transform_groups=rotation_transforms,
            initial_transform_group='eval').freeze_transforms()

        list_train_dataset.append(rotated_train)
        list_test_dataset.append(rotated_test)

    tmp_scenario = dataset_benchmark(
            train_datasets=list_train_dataset,
            test_datasets=list_test_dataset,
            complete_test_set_only=False,
            train_transform=train_transform,
            eval_transform=eval_transform)

    for j in range(len(tmp_scenario.train_stream)):
        tmp_scenario.train_stream[j].classes_in_this_experience = torch.as_tensor(
            tmp_scenario.train_stream[j].classes_in_this_experience, dtype=torch.int32).unique().tolist()
        tmp_scenario.train_stream[j].classes_seen_so_far = torch.as_tensor(
            tmp_scenario.train_stream[j].classes_seen_so_far, dtype=torch.int32).unique().tolist()

        tmp_scenario.test_stream[j].classes_in_this_experience = torch.as_tensor(
            tmp_scenario.test_stream[j].classes_in_this_experience, dtype=torch.int32).unique().tolist()
        tmp_scenario.test_stream[j].classes_seen_so_far = torch.as_tensor(
            tmp_scenario.test_stream[j].classes_seen_so_far, dtype=torch.int32).unique().tolist()

    tmp_scenario.n_classes = 10
    return tmp_scenario


def _get_mnist_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location('mnist')

    train_set = MNIST(root=dataset_root,
                      train=True, download=True)

    test_set = MNIST(root=dataset_root,
                     train=False, download=True)

    return train_set, test_set


__all__ = [
    'SplitMNIST',
    'PermutedMNIST',
    'RotatedMNIST',
    'RotatedMNIST_di'
]


if __name__ == "__main__":
    import sys

    print('Split MNIST')
    benchmark_instance = SplitMNIST(
        5, train_transform=None, eval_transform=None)
    check_vision_benchmark(benchmark_instance)

    print('Permuted MNIST')
    benchmark_instance = PermutedMNIST(
        5, train_transform=None, eval_transform=None)
    check_vision_benchmark(benchmark_instance)

    print('Rotated MNIST')
    benchmark_instance = RotatedMNIST(
        5, train_transform=None, eval_transform=None)
    check_vision_benchmark(benchmark_instance)
    print('Rotated MNIST')
    benchmark_instance = RotatedMNIST_di(4,
                                         rain_transform=None, eval_transform=None,
                                         seed=None, rotations_list=(0, 90, 180, -90))

    sys.exit(0)
