from avalanche.benchmarks import *
from avalanche.benchmarks.classic.ccifar10 import RotatedCIFAR10_di

from avalanche.training.strategies import *
from avalanche.models import *
from avalanche.models.MultiMLP import SimpleCNN
from avalanche.models.MultiMLP import PREDICT_METHOD_ONE_CLASS, PREDICT_METHOD_MAJORITY_VOTE, PREDICT_METHOD_RANDOM, \
    PREDICT_METHOD_TASK_ID_KNOWN, PREDICT_METHOD_NW_CONFIDENCE, PREDICT_METHOD_NAIVE_BAYES
from avalanche.evaluation.metrics import *
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import subprocess
import argparse

PROGRAM_NAME = "train_pool_app"


def scenario_from_csv_with_drift(csv_file, n_classes, n_input_features, minibatch_size=10, drift_width=250000):
    class MyDataset(Dataset):
        def __init__(self, file_path, number_of_features):
            self.df = pd.read_csv(file_path)
            self.data = self.df.to_numpy()
            self.x, self.y = (torch.from_numpy(self.data[:, :number_of_features]),
                              torch.from_numpy(self.data[:, number_of_features:]))

        def __getitem__(self, idx):
            return self.x[idx, :], self.y[idx, :]

        def __len__(self):
            return len(self.data)

    my_data = MyDataset(csv_file, n_input_features)
    data_loader = DataLoader(my_data, batch_size=minibatch_size, shuffle=False)
    train_experiences = []
    test_experiences = []
    task_labels = []
    task_id = -1
    test_tensor_for_this_drift_x = None
    test_tensor_for_this_drift_y = None
    train_tensor_for_this_drift_x = None
    train_tensor_for_this_drift_y = None
    batch_count = 0
    for x, y in data_loader:
        xx = torch.as_tensor(x, dtype=torch.float)
        yy = torch.as_tensor(y, dtype=torch.long).view(-1, )
        if batch_count % (drift_width / minibatch_size) == 0:
            if batch_count != 0:
                task_id += 1
                task_labels.append(task_id)
                test_experiences.append((test_tensor_for_this_drift_x, test_tensor_for_this_drift_y))
                train_experiences.append((train_tensor_for_this_drift_x, train_tensor_for_this_drift_y))
                test_tensor_for_this_drift_x = None
                test_tensor_for_this_drift_y = None
                train_tensor_for_this_drift_x = None
                train_tensor_for_this_drift_y = None
        if batch_count % 2 == 0:
            if test_tensor_for_this_drift_x is None:
                test_tensor_for_this_drift_x = xx
                test_tensor_for_this_drift_y = yy
            else:
                test_tensor_for_this_drift_x = torch.cat((test_tensor_for_this_drift_x, xx)).detach()
                test_tensor_for_this_drift_y = torch.cat((test_tensor_for_this_drift_y, yy)).detach()
        else:
            if train_tensor_for_this_drift_x is None:
                train_tensor_for_this_drift_x = xx
                train_tensor_for_this_drift_y = yy
            else:
                train_tensor_for_this_drift_x = torch.cat((train_tensor_for_this_drift_x, xx)).detach()
                train_tensor_for_this_drift_y = torch.cat((train_tensor_for_this_drift_y, yy)).detach()
        batch_count += 1
    test_experiences.append((test_tensor_for_this_drift_x, test_tensor_for_this_drift_y))
    train_experiences.append((train_tensor_for_this_drift_x, train_tensor_for_this_drift_y))

    scenario = tensors_benchmark(
        train_tensors=train_experiences,
        test_tensors=train_experiences,
        # task_labels=[0, 0, 0, 0],  # Task label of each train exp
        task_labels=task_labels,
        complete_test_set_only=False
    )
    scenario.n_classes = n_classes
    return scenario


def main(args):
    # check if selected GPU is available or use CPU
    assert args.cuda == -1 or args.cuda >= 0, "cuda must be -1 or >= 0."
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                             args.cuda >= 0 else "cpu")
    print(f'Using device: {device}')

    # create streams
    if args.dataset == 'RotatedMNIST':
        scenario = RotatedMNIST_di(4, seed=None, rotations_list=(0, 90, 180, -90))
        input_size = 28 * 28
        num_channels = 1
    elif args.dataset == 'RotatedCIFAR10':
        scenario = RotatedCIFAR10_di(4, seed=None, rotations_list=(0, 90, 180, -90))
        input_size = 3 * 32 * 32
        num_channels = 3
    elif args.dataset == 'LED_a' or args.dataset == 'LED_a_ex':
        input_size = 24
        num_channels = 0
        scenario = scenario_from_csv_with_drift(
            csv_file='/Users/ng98/Desktop/datasets/NEW/unzipped/' + args.dataset + '.csv',
            n_input_features=input_size,
            n_classes=10,
            minibatch_size=args.minibatch_size, drift_width=250000)
    elif args.dataset == 'CORe50':
        scenario = CORe50(scenario='ni', run=0, object_lvl=False, mini=True)
        input_size = 3 * 32 * 32
        num_channels = 3
        scenario.n_classes = 10

    # for step in scenario.train_stream:
    #     data = step.dataset
    #     dl = DataLoader(data, batch_size=args.minibatch_size)
    #     for x, y, t in dl:
    #         # print(y)
    #         print(t)
    #         break
    # print('hi')
    # exit(0)

    if args.module == 'SimpleMLP' or args.module == 'SimpleCNN':
        if args.module == 'SimpleMLP':
            model = SimpleMLP(hidden_size=args.hs, num_classes=scenario.n_classes, input_size=input_size)
        else:
            model = SimpleCNN(num_classes=10, num_channels=num_channels)
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = None
    elif args.module == 'MultiMLP':
        if args.predict_method == 'ONE_CLASS':
            predict_method = PREDICT_METHOD_ONE_CLASS
        elif args.predict_method == 'MAJORITY_VOTE':
            predict_method = PREDICT_METHOD_MAJORITY_VOTE
        elif args.predict_method == 'RANDOM':
            predict_method = PREDICT_METHOD_RANDOM
        elif args.predict_method == 'TASK_ID_KNOWN':
            predict_method = PREDICT_METHOD_TASK_ID_KNOWN
        elif args.predict_method == 'NW_CONFIDENCE':
            predict_method = PREDICT_METHOD_NW_CONFIDENCE
        elif args.predict_method == 'NAIVE_BAYES':
            predict_method = PREDICT_METHOD_NAIVE_BAYES

        model = MultiMLP(
            num_classes=scenario.n_classes,
            use_threads=False,
            use_adwin=True,
            number_of_mlps_to_train=args.number_of_mpls_to_train,
            predict_method=predict_method,
            nn_pool_type=args.pool_type,
            back_prop_skip_loss_threshold=args.skip_back_prop_threshold,
            device=device)
        optimizer = None

    criterion = torch.nn.CrossEntropyLoss()

    command = subprocess.Popen("realpath " + args.base_dir, shell=True, stdout=subprocess.PIPE)
    for line in command.stdout.readlines():
        print("Base directory: ", line.decode("utf-8").replace('\n', ''))

    # set loggers
    # log to Tensorboard
    tb_logger = TensorboardLogger(tb_log_dir=args.base_dir + '/logs/tb_data/' + args.log_file_name)
    # log to text file
    text_logger = TextLogger(open(args.base_dir + '/logs/txt_logs/' + args.log_file_name + '.txt', 'a'))
    # csv logger
    csv_logger = CSVLogger(log_folder=args.base_dir + '/logs/csv_data/' + args.log_file_name)
    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(save_image=True, normalize='all', stream=True),
        loggers=[interactive_logger, text_logger, tb_logger, csv_logger])

    # create strategy
    if args.strategy == 'TrainPool':
        strategy = TrainPool(model, optimizer, criterion,
                             train_epochs=args.epochs, device=device, train_mb_size=args.minibatch_size,
                             evaluator=eval_plugin)
    elif args.strategy == 'LwF':
        assert len(args.lwf_alpha) == 1 or len(args.lwf_alpha) == 5, \
            'Alpha must be a non-empty list.'
        lwf_alpha = args.lwf_alpha[0] if len(args.lwf_alpha) == 1 \
            else args.lwf_alpha
        strategy = LwF(model, optimizer, criterion,
                       alpha=lwf_alpha, temperature=args.softmax_temperature,
                       train_epochs=args.epochs, device=device, train_mb_size=args.minibatch_size,
                       evaluator=eval_plugin)
    elif args.strategy == 'EWC':
        strategy = EWC(model, optimizer, criterion,
                       ewc_lambda=0.4, mode='online', decay_factor=0.1,
                       train_epochs=args.epochs, device=device, train_mb_size=args.minibatch_size,
                       evaluator=eval_plugin)
    elif args.strategy == 'GDumb':
        strategy = GDumb(model, optimizer, criterion,
                         mem_size=2000,
                         train_epochs=args.epochs, device=device, train_mb_size=args.minibatch_size,
                         evaluator=eval_plugin)

    # for j in range(len(scenario.train_stream)):
    #     print('Task: {} size: {}'.format(j, len(scenario.train_stream[j].dataset)))
    #     for i, (img, label, unknown) in enumerate(scenario.train_stream[j].dataset):
    #         if label == 7:
    #             print(torch.as_tensor(img, dtype=torch.long))
    #             break
    # exit(0)

    # train on the selected scenario with the chosen strategy
    print('Starting experiment...')
    results = []
    for train_task in scenario.train_stream:
        print("Start training on experience ", train_task.current_experience)
        strategy.train(train_task, num_workers=0)
        print("End training on experience ", train_task.current_experience)

        print('Computing accuracy on the test set')
        # results.append(strategy.eval(scenario.test_stream[:]))
        results.append(strategy.eval(scenario.test_stream[0:train_task.current_experience + 1]))

    # print("Test results ", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lwf_alpha', nargs='+', type=float,
                        default=[0, 0.5, 1.333, 2.25, 3.2],
                        help='Penalty hyperparameter for LwF. It can be either'
                             'a list with multiple elements (one alpha per '
                             'experience) or a list of one element (same alpha '
                             'for all experiences).')
    parser.add_argument('--softmax_temperature', type=float, default=1,
                        help='Temperature for softmax used in distillation')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--hs', type=int, default=1024, help='MLP hidden size.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs.')
    parser.add_argument('--minibatch_size', type=int, default=100,
                        help='Minibatch size.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Specify GPU id to use. Use CPU if -1.')
    parser.add_argument('--dataset', type=str, default='RotatedMNIST',
                        help='dataset to train and test')
    parser.add_argument('--module', type=str, default='MultiMLP',
                        help='Module type')
    parser.add_argument('--strategy', type=str, default='TrainPool',
                        help='Strategy type')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer type')
    parser.add_argument('--base_dir', type=str, default='/Users/ng98/Desktop/avalanche_test',
                        help='Base Directory')
    parser.add_argument('--number_of_mpls_to_train', type=int, default=6,
                        help='Number of MPLs to train for MultiMLP.')
    parser.add_argument('--pool_type', type=str, default='6CNN',
                        help='Pool type for MultiMLP.')
    parser.add_argument('--predict_method', type=str, default='ONE_CLASS',
                        choices=['ONE_CLASS', 'MAJORITY_VOTE', 'RANDOM', 'TASK_ID_KNOWN', 'NW_CONFIDENCE',
                                 'NAIVE_BAYES'],
                        help='Prediction method for MultiMLP: '
                             'ONE_CLASS, MAJORITY_VOTE, RANDOM, TASK_ID_KNOWN, NW_CONFIDENCE or NAIVE_BAYES.')
    parser.add_argument('--skip_back_prop_threshold', type=float, default=0.0,
                        help='back_prop_skip_loss_threshold for MultiMLP')
    parser.add_argument('--log_file_name', type=str, default='',
                        help='Log file name')

    args = parser.parse_args()

    print(args)

    main(args)
    print('Experiment completed')
