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

import torch
# import tqdm
from torch import Tensor



class AutoEncorder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8)
        )

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 512),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




class InstanceBuffer:
    def __init__(self, mem_size: int = 200):
        self.mem_size = mem_size
        self.ext_mem: Dict[Any, Tuple[List[Tensor], List[Tensor]]] = {}
        # count occurrences for each class
        self.counter: Dict[Any, Dict[Any, int]] = {}

    def add_items(self, x, y, t):
        # for each pattern, add it to the memory or not
        for i in range(len(x)):
            pattern = x[i]
            task_id = t
            target = y[i]
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
                            current_mem[1][j] = target
                            break
                    current_counter[to_remove] -= 1
                else:
                    # memory not full: add new pattern
                    current_mem[0].append(pattern)
                    current_mem[1].append(target)

                # Indicate that we've changed the number of stored instances of
                # this class.
                current_counter[target_value] += 1


    def get_union_buffer(self, task_id, x, y):
        xx = x.detach()
        yy = y.detach()

        x_size = len(x)
        if self.ext_mem.get(task_id) is not None:
            patterns, targets = self.samples_per_each_task_at_train[task_id]
            patterns = torch.tensor(patterns)
            targets = torch.tensor(targets)

            buffer_size = len(patterns)
            buffer_copy_size = min(x_size/2, buffer_size)

            indices_to_copy = sample(range(buffer_size), buffer_copy_size)
            indices_to_copy = torch.tensor(indices_to_copy)

            replace_indices = sample(range(len(buffer_copy_size)), x_size-buffer_copy_size)
            replace_indices = torch.tensor(replace_indices)

            xx.index_copy_(0, replace_indices, torch.index_select(patterns, 0, indices_to_copy))
            yy.index_copy_(0, replace_indices, torch.index_select(targets, 0, indices_to_copy))

        return xx, yy
