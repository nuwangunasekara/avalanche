import os
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultsDir", type=str, help="Results directory", default='/Users/ng98/Desktop/results/results/reset_oneclass_reset_loss_estd_include_best_tr_seed_0_ex2/reset_oneclass_reset_loss_estd_include_best_tr_seed_0_ex2/reset_oneclass_reset_loss_estd_include_best_tr_seed_0_ex2_1')
args = parser.parse_args()

# pd.set_option('max_columns', None)
# pd.set_option('max_rows', None)
# pd.set_option('display.width', 1000)

csv_dir = os.path.join(args.resultsDir,  'logs/exp_logs/')

datasets = ['CORe50', 'RotatedCIFAR10', 'RotatedMNIST']


def get_net_csv_file(d):
    csv_files = []
    if len(csv_files) == 0:
        command = subprocess.Popen("find " + csv_dir + " -iname " + d + "*Nets.csv",
                                   shell=True, stdout=subprocess.PIPE)
        for line in command.stdout.readlines():
            csv_files.append(line.decode("utf-8").replace('\n', ''))
    print('csv_files', csv_files)
    return csv_files[0] if len(csv_files) > 0 else None


def plot_task_detection():
    fig = plt.figure(constrained_layout=False, figsize=(18, 10))
    gs = fig.add_gridspec(len(datasets), 1)
    rows = 0
    col = 0

    for d in datasets:
        ax = fig.add_subplot(gs[rows, col], label=d)
        csv_file = get_net_csv_file(d)
        if csv_file is None:
            continue
        df = pd.read_csv(csv_file)

        ax.set_ylabel('task_id (' + d + ')')

        df_training = df.loc[df['dumped_at'] == 'after_training']
        df_detected = df.loc[df['dumped_at'] == 'task_detect']

        ax.plot(df_training['total_samples_seen_for_train'],  df_training['training_exp'], label='training_task_id', marker=".")
        ax.plot(df_detected['total_samples_seen_for_train'],  df_detected['detected_task_id'], label='detected_task_id', marker=".")

        for x, y in zip(df_training['total_samples_seen_for_train'], df_training['training_exp']):
            ax.annotate(y, (x, y))
        for x, y in zip(df_detected['total_samples_seen_for_train'], df_detected['detected_task_id']):
            ax.annotate(y, (x, y))

        x_ticks = df_training['total_samples_seen_for_train'].unique()
        ax.set_xticks(x_ticks)
        ax.set(xlim=(0, x_ticks.max()), ylim=(0, None))

        rows += 1

    ax.legend()


def plot_network_selection():
    # fig = plt.figure(constrained_layout=False, figsize=(18, 10))
    # gs = fig.add_gridspec(len(datasets), 1)
    rows = 0
    col = 0
    columns = ['training_exp', 'dumped_at', 'list_type', 'this_id', 'this_estimated_loss', 'this_correct_network_selected', 'correct_network_selected', 'this_acc', 'acc']

    with pd.ExcelWriter(args.resultsDir + '/NetworkInfo.xlsx') as writer:
        for d in datasets:
            # ax = fig.add_subplot(gs[rows, col], label=d)
            csv_file = get_net_csv_file(d)
            if csv_file is None:
                continue
            df = pd.read_csv(csv_file)

            # ax.set_ylabel('task_id (' + d + ')')

            # df_eval = df.loc[df['dumped_at'] == 'after_eval']
            #
            # df_eval = df_eval.loc[df_eval['list_type'] == 'frozen_net']

            df_frozen = df.query('dumped_at.str.contains("after_eval") and list_type.str.contains("frozen_net")', engine='python')
            # print(df_frozen.to_string())
            # print(len(df_frozen))

            df_best_train = df.query('dumped_at.str.contains("after_eval") and list_type.str.contains("train_net")', engine='python')
            df_best_train_idx = df_best_train.groupby(['training_exp'])['this_estimated_loss'].idxmin()
            df_best_train = df_best_train.loc[df_best_train_idx]
            # print(df_best_train)
            # print(len(df_best_train))

            pd_selected_for_pred = pd.concat([df_best_train, df_frozen])
            pd_selected_for_pred = pd_selected_for_pred.sort_values(by=['training_exp'])

            # print(pd_selected_for_pred[columns].to_string())

            pd_selected_for_pred.to_excel(writer, sheet_name=d, index=False)

            # ax.plot(df_training['total_samples_seen_for_train'],  df_training['training_exp'], label='training_task_id', marker=".")
            # ax.plot(df_detected['total_samples_seen_for_train'],  df_detected['detected_task_id'], label='detected_task_id', marker=".")
            #
            # for x, y in zip(df_training['total_samples_seen_for_train'], df_training['training_exp']):
            #     ax.annotate(y, (x, y))
            # for x, y in zip(df_detected['total_samples_seen_for_train'], df_detected['detected_task_id']):
            #     ax.annotate(y, (x, y))
            #
            # x_ticks = df_training['total_samples_seen_for_train'].unique()
            # ax.set_xticks(x_ticks)
            # ax.set(xlim=(0, None), ylim=(0, None))

            rows += 1

    # ax.legend()


plot_task_detection()
plot_network_selection()

mplcursors.cursor(hover=True)
plt.savefig(args.resultsDir+'/TaskDetection.png')
plt.show()
