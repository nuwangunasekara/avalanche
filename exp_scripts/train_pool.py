import os.path
import shutil

from avalanche.benchmarks import *
from avalanche.benchmarks.classic.ccifar10 import RotatedCIFAR10_di
from avalanche.benchmarks.classic.stream51 import CLStream51

from avalanche.training.strategies import *
from avalanche.models import *
from avalanche.models.MultiMLP import SimpleCNN, CNN4
from avalanche.models.MultiMLP import PREDICT_METHOD_ONE_CLASS, PREDICT_METHOD_MAJORITY_VOTE, PREDICT_METHOD_RANDOM, \
    PREDICT_METHOD_TASK_ID_KNOWN, PREDICT_METHOD_NW_CONFIDENCE, PREDICT_METHOD_NAIVE_BAYES, PREDICT_METHOD_HT
from avalanche.models.MultiMLP import DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END, WITH_ACCUMULATED_INSTANCES, \
    WITH_ACCUMULATED_LEARNED_FEATURES, WITH_ACCUMULATED_STATIC_FEATURES
from avalanche.evaluation.metrics import *
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins import ReplayPlugin

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
    print("======\n")
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                             args.cuda >= 0 else "cpu")
    print("CUDA version: {}".format(torch.version.cuda))
    print(f'Using device: {device}')
    print("======\n")

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
        scenario = CORe50(scenario='ni_di_task_id_by_session', run=0, object_lvl=False, mini=True)
        input_size = 3 * 32 * 32
        num_channels = 3
        scenario.n_classes = 10
    elif args.dataset == 'CLStream51':
        scenario = CLStream51(scenario='instance', seed=10, eval_num=None,
                              dataset_root='/Scratch/ng98/CL/avalanche_data/',
                              no_novelity_detection=True
                              )
        input_size = 3 * 224 * 224
        num_channels = 3
    # exit(0)

    if args.module == 'SimpleMLP' or args.module == 'SimpleCNN' or args.module == 'CNN4':
        if args.module == 'SimpleMLP':
            model = SimpleMLP(hidden_size=args.hs, num_classes=scenario.n_classes, input_size=input_size)
        if args.module == 'CNN4':
            model = CNN4(num_classes=scenario.n_classes, num_channels=num_channels)
        else:
            model = SimpleCNN(num_classes=scenario.n_classes, num_channels=num_channels)
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = None
    elif args.module == 'MultiMLP':
        if args.task_detector_type == 'ONE_CLASS':
            predict_method = PREDICT_METHOD_ONE_CLASS
        elif args.task_detector_type == 'MAJORITY_VOTE':
            predict_method = PREDICT_METHOD_MAJORITY_VOTE
        elif args.task_detector_type == 'RANDOM':
            predict_method = PREDICT_METHOD_RANDOM
        elif args.task_detector_type == 'TASK_ID_KNOWN':
            predict_method = PREDICT_METHOD_TASK_ID_KNOWN
        elif args.task_detector_type == 'NW_CONFIDENCE':
            predict_method = PREDICT_METHOD_NW_CONFIDENCE
        elif args.task_detector_type == 'NAIVE_BAYES':
            predict_method = PREDICT_METHOD_NAIVE_BAYES
        elif args.task_detector_type == 'HT':
            predict_method = PREDICT_METHOD_HT

        model_dump_dir = os.path.join(args.base_dir + '/logs/exp_logs/', args.log_file_name + '_f_pool')
        if os.path.isdir(model_dump_dir):
            shutil.rmtree(model_dump_dir)
        os.mkdir(model_dump_dir)

        train_task_predictor_at_the_end = DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END
        if args.train_task_predictor_at_the_end == 'DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END':
            train_task_predictor_at_the_end = DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END
        elif args.train_task_predictor_at_the_end == 'WITH_ACCUMULATED_INSTANCES':
            train_task_predictor_at_the_end = WITH_ACCUMULATED_INSTANCES
        elif args.train_task_predictor_at_the_end == 'WITH_ACCUMULATED_LEARNED_FEATURES':
            train_task_predictor_at_the_end = WITH_ACCUMULATED_LEARNED_FEATURES
        elif args.train_task_predictor_at_the_end == 'WITH_ACCUMULATED_STATIC_FEATURES':
            train_task_predictor_at_the_end = WITH_ACCUMULATED_STATIC_FEATURES
            args.use_static_f_ex = True

        model = MultiMLP(
            num_classes=scenario.n_classes,
            use_threads=False,
            loss_estimator_delta=1e-3,
            number_of_mlps_to_train=args.number_of_mpls_to_train,
            predict_method=predict_method,
            nn_pool_type=args.pool_type,
            back_prop_skip_loss_threshold=args.skip_back_prop_threshold,
            device=device,
            train_task_predictor_at_the_end=train_task_predictor_at_the_end,
            stats_file=args.base_dir + '/logs/exp_logs/' + args.log_file_name + '_Nets.csv',
            model_dump_dir=model_dump_dir,
            reset_training_pool=args.reset_training_pool,
            use_one_class_probas=args.use_one_class_probas,
            use_weights_from_task_detectors=args.use_weights_from_task_detectors,
            auto_detect_tasks=args.auto_detect_tasks,
            n_experiences=scenario.n_experiences,
            use_static_f_ex=args.use_static_f_ex,
            train_nn_using_ex_static_f=args.train_nn_using_ex_static_f,
            train_only_the_best_nn=args.train_only_the_best_nn,
            use_1_channel_pretrained_for_1_channel=args.use_1_channel_pretrained_for_1_channel,
            use_quantized=args.use_quantized)
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
        # loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        # confusion_matrix_metrics(save_image=True, normalize='all', stream=True),
        loggers=[interactive_logger, text_logger, tb_logger, csv_logger])

    # create strategy
    if args.strategy == 'TrainPool':
        strategy = TrainPool(model, optimizer, criterion,
                             train_epochs=args.epochs, device=device, train_mb_size=args.minibatch_size,
                             # plugins=[ReplayPlugin(mem_size=1000)],
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
                         mem_size=args.mem_buff_size,
                         train_epochs=args.epochs, device=device, train_mb_size=args.minibatch_size,
                         evaluator=eval_plugin)
    elif args.strategy == 'ER':
        strategy = Replay(model, optimizer, criterion,
                          mem_size=args.mem_buff_size,
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
        results.append(strategy.eval(scenario.test_stream[0: 1 if len(scenario.test_stream) == 1 else train_task.current_experience + 1]))

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
    parser.add_argument('--mem_buff_size', type=int, default=1000,
                        help='Memory buffer size for replay methods(GDumb,ER)')
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
    parser.add_argument('--task_detector_type', type=str, default='ONE_CLASS',
                        choices=['ONE_CLASS', 'MAJORITY_VOTE', 'RANDOM', 'TASK_ID_KNOWN', 'NW_CONFIDENCE',
                                 'NAIVE_BAYES', 'HT'],
                        help='Prediction method for MultiMLP: '
                             'ONE_CLASS, MAJORITY_VOTE, RANDOM, TASK_ID_KNOWN, NW_CONFIDENCE or NAIVE_BAYES.')
    parser.add_argument('--skip_back_prop_threshold', type=float, default=0.0,
                        help='back_prop_skip_loss_threshold for MultiMLP')
    parser.add_argument('--log_file_name', type=str, default='',
                        help='Log file name')

    parser.add_argument('--train_task_predictor_at_the_end', type=str, default='ONE_CLASS',
                        choices=['DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END',
                                 'WITH_ACCUMULATED_INSTANCES',
                                 'WITH_ACCUMULATED_LEARNED_FEATURES',
                                 'WITH_ACCUMULATED_STATIC_FEATURES'],
                        help='Train task predictor at the end: '
                             'DO_NOT_NOT_TRAIN_TASK_PREDICTOR_AT_THE_END, '
                             'WITH_ACCUMULATED_INSTANCES, '
                             'WITH_ACCUMULATED_LEARNED_FEATURES, '
                             'WITH_ACCUMULATED_STATIC_FEATURES.')

    # reset_training_pool
    parser.add_argument('--reset_training_pool', dest='reset_training_pool',
                        action='store_true')
    parser.add_argument('--no-reset_training_pool', dest='reset_training_pool',
                        action='store_false')
    parser.set_defaults(reset_training_pool=False)

    # use_one_class_probas
    parser.add_argument('--use_one_class_probas', dest='use_one_class_probas',
                        action='store_true')
    parser.add_argument('--no-use_one_class_probas', dest='use_one_class_probas',
                        action='store_false')
    parser.set_defaults(use_one_class_probas=False)

    # use_weights_from_task_detectors
    parser.add_argument('--use_weights_from_task_detectors', dest='use_weights_from_task_detectors',
                        action='store_true')
    parser.add_argument('--no-use_weights_from_task_detectors', dest='use_weights_from_task_detectors',
                        action='store_false')
    parser.set_defaults(use_weights_from_task_detectors=False)

    # auto_detect_tasks
    parser.add_argument('--auto_detect_tasks', dest='auto_detect_tasks',
                        action='store_true')
    parser.add_argument('--no-auto_detect_tasks', dest='auto_detect_tasks',
                        action='store_false')
    parser.set_defaults(auto_detect_tasks=False)

    # use_static_f_ex
    parser.add_argument('--use_static_f_ex', dest='use_static_f_ex',
                        action='store_true')
    parser.add_argument('--no-use_static_f_ex', dest='use_static_f_ex',
                        action='store_false')
    parser.set_defaults(use_static_f_ex=False)

    # train_nn_using_ex_static_f
    parser.add_argument('--train_nn_using_ex_static_f', dest='train_nn_using_ex_static_f',
                        action='store_true')
    parser.add_argument('--no-train_nn_using_ex_static_f', dest='train_nn_using_ex_static_f',
                        action='store_false')
    parser.set_defaults(train_nn_using_ex_static_f=False)

    # train_only_the_best_nn
    parser.add_argument('--train_only_the_best_nn', dest='train_only_the_best_nn',
                        action='store_true')
    parser.add_argument('--no-train_only_the_best_nn', dest='train_only_the_best_nn',
                        action='store_false')
    parser.set_defaults(train_only_the_best_nn=False)

    # use_1_channel_pretrained_for_1_channel
    parser.add_argument('--use_1_channel_pretrained_for_1_channel', dest='use_1_channel_pretrained_for_1_channel',
                        action='store_true')
    parser.add_argument('--no-use_1_channel_pretrained_for_1_channel', dest='use_1_channel_pretrained_for_1_channel',
                        action='store_false')
    parser.set_defaults(use_1_channel_pretrained_for_1_channel=False)

    # use_quantized
    parser.add_argument('--use_quantized', dest='use_quantized',
                        action='store_true')
    parser.add_argument('--no-use_quantized', dest='use_quantized',
                        action='store_false')
    parser.set_defaults(use_quantized=True)

    args = parser.parse_args()

    print(args)

    main(args)
    print('Experiment completed')
