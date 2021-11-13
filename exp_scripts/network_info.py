import os
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import argparse
import ast

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


def plot_task_detection(d, row, col):
    df = pd.read_csv(csv_file)

    ax = fig.add_subplot(gs[row, col], label=d)
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

    ax.legend()


def plot_network_selection(d, row, col):
    columns = ['training_exp', 'dumped_at', 'detected_task_id', 'list_type', 'this_name', 'this_frozen_id', 'this_id', 'this_correctly_predicted_task_ids_test', 'correct_network_selected', 'correct_class_predicted', 'total_samples_seen_for_test']

    df = pd.read_csv(csv_file)

    ax_1 = fig_1.add_subplot(gs_1[row, col], label=d)
    ax_2 = fig_2.add_subplot(gs_2[row, col], label=d)
    ax_1.set_ylabel('accum counts (' + d + ')')

    df_frozen = df.query('dumped_at.str.contains("after_eval") and list_type.str.contains("frozen_net")', engine='python')

    df_best_train = df.query('dumped_at.str.contains("after_eval") and list_type.str.contains("train_net")', engine='python')
    df_best_train_idx = df_best_train.groupby(['training_exp'])['this_estimated_loss'].idxmin()
    df_best_train = df_best_train.loc[df_best_train_idx]

    pd_selected_for_pred = pd.concat([df_best_train, df_frozen])
    pd_selected_for_pred = pd_selected_for_pred.sort_values(by=['training_exp'])
    pd_selected_for_pred = pd_selected_for_pred[columns]

    c_prefix = 'correctly_predicted_task_id_'
    c_prefix = ''

    col_names_1 = ['correct_network_selected', 'correct_class_predicted', 'total_samples_seen_for_test']
    col_names_2 = ['training_exp',  'this_frozen_id']
    col_names_3 = []
    for t in pd_selected_for_pred['training_exp'].unique():
        t_id = c_prefix + str(t)
        pd_selected_for_pred[t_id] = 0
        col_names_2.append(t_id)
        col_names_3.append(t_id)

    for i in pd_selected_for_pred.index.values:
        t_ids = ast.literal_eval(pd_selected_for_pred.at[i, 'this_correctly_predicted_task_ids_test'])
        for t, val in t_ids.items():
            pd_selected_for_pred.at[i, c_prefix + str(t)] = val

    pd_selected_for_pred.to_excel(excel_writer, sheet_name=d, index=False)

    pd_f = pd_selected_for_pred[pd_selected_for_pred['list_type'] == 'frozen_net']
    pd_f = pd_f[col_names_2].groupby(['training_exp',  'this_frozen_id'])[col_names_3].max()
    # print(pd_f.to_string())
    pd_f.plot(kind='bar', stacked=False, ax=ax_2)
    # ax_2.set_xticks(rotation=45)
    for label in ax_2.get_xticklabels():
        label.set_rotation(20)
        label.set_ha('right')
    ax_2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    pd_selected_for_pred = pd_selected_for_pred.groupby(['training_exp'])[col_names_1].max()
    # print(pd_selected_for_pred.to_string())
    pd_selected_for_pred.plot(kind='bar', stacked=False, ax=ax_1)


# Start of main
fig = plt.figure(constrained_layout=False, figsize=(18, 10))
fig_1 = plt.figure(constrained_layout=False, figsize=(18, 10))
fig_2 = plt.figure(constrained_layout=False, figsize=(18, 10))
gs = fig.add_gridspec(len(datasets), 1)
gs_1 = fig_1.add_gridspec(len(datasets), 1)
gs_2 = fig_1.add_gridspec(len(datasets), 1)
rows = 0
cols = 0
with pd.ExcelWriter(args.resultsDir + '/NetworkInfo.xlsx') as excel_writer:
    for d_name in datasets:
        csv_file = get_net_csv_file(d_name)
        if csv_file is None:
            continue
        plot_task_detection(d_name, rows, cols)
        plot_network_selection(d_name, rows, cols)
        rows += 1

mplcursors.cursor(hover=True)
plt.savefig(args.resultsDir+'/TaskDetection.png')
plt.show()
