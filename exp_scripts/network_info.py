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

exp_log_dir = os.path.join(args.resultsDir, 'logs/exp_logs/')
csv_dir = os.path.join(args.resultsDir, 'logs/csv_data/')

datasets = ['CORe50', 'RotatedCIFAR10', 'RotatedMNIST']


def get_net_csv_file(d):
    csv_files = []
    if len(csv_files) == 0:
        command = subprocess.Popen("find " + exp_log_dir + " -iname " + d + "*Nets.csv",
                                   shell=True, stdout=subprocess.PIPE)
        for line in command.stdout.readlines():
            csv_files.append(line.decode("utf-8").replace('\n', ''))
    print('Net csv_files', csv_files)
    return csv_files[0] if len(csv_files) > 0 else None


def get_eval_csv_file(d):
    csv_files = []
    if len(csv_files) == 0:
        command = subprocess.Popen("find " + csv_dir + " -iname eval_results.csv |grep " + d,
                                   shell=True, stdout=subprocess.PIPE)
        for line in command.stdout.readlines():
            csv_files.append(line.decode("utf-8").replace('\n', ''))
    print('eval csv_files', csv_files)
    return csv_files[0] if len(csv_files) > 0 else None


def get_experiment_dir():
    output = []
    if len(output) == 0:
        command = subprocess.Popen("realpath " + args.resultsDir,
                                   shell=True, stdout=subprocess.PIPE)
        for line in command.stdout.readlines():
            output.append(line.decode("utf-8").replace('\n', ''))
    print('Experiment directory ', output)
    return output[0] if len(output) > 0 else None


def plot_task_detection(csv_file, d, ax):
    df = pd.read_csv(csv_file)

    ax.set_ylabel('task_id (' + d + ')')

    df_training = df.loc[df['dumped_at'] == 'after_training']
    df_detected = df.loc[df['dumped_at'] == 'task_detect']

    ax.plot(df_training['total_samples_seen_for_train'], df_training['training_exp'], label='training_task_id', marker=".")
    ax.plot(df_detected['total_samples_seen_for_train'], df_detected['detected_task_id'], label='detected_task_id', marker=".")

    for x, y in zip(df_training['total_samples_seen_for_train'], df_training['training_exp']):
        ax.annotate(y, (x, y))
    for x, y in zip(df_detected['total_samples_seen_for_train'], df_detected['detected_task_id']):
        ax.annotate(y, (x, y))

    x_ticks = df_training['total_samples_seen_for_train'].unique()
    ax.set_xticks(x_ticks)
    ax.set(xlim=(0, x_ticks.max()), ylim=(0, None))

    ax.legend()


def plot_network_selection(csv_file, excel_writer, d, ax):
    columns = [
        'training_exp', 'dumped_at', 'detected_task_id', 'list_type', 'this_name', 'this_frozen_id', 'this_id',
        'samples_per_each_task_at_train', 'this_seen_task_ids_train',
        'this_correctly_predicted_task_ids_test', 'correct_network_selected', 'correct_class_predicted', 'total_samples_seen_for_test',
        'this_correctly_predicted_task_ids_test_at_last', 'this_correctly_predicted_task_ids_probas_test_at_last', 'correct_network_selected_count_at_last', 'instances_per_task_at_last']

    df = pd.read_csv(csv_file)
    ax.set_ylabel(d)

    df_frozen = df.query('dumped_at.str.contains("after_eval") and list_type.str.contains("frozen_net")', engine='python')

    df_best_train = df.query('dumped_at.str.contains("after_eval") and list_type.str.contains("train_net")', engine='python')
    df_best_train_idx = df_best_train.groupby(['training_exp'])['this_estimated_loss'].idxmin()
    df_best_train = df_best_train.loc[df_best_train_idx]

    pd_selected_for_pred = pd.concat([df_best_train, df_frozen])
    pd_selected_for_pred = pd_selected_for_pred.sort_values(by=['training_exp'])
    pd_selected_for_pred = pd_selected_for_pred[columns]

    col_names_1 = ['correct_network_selected', 'correct_class_predicted', 'total_samples_seen_for_test']

    pd_selected_for_pred.to_excel(excel_writer, sheet_name=d, index=False)

    pd_selected_for_pred = pd_selected_for_pred.groupby(['training_exp'])[col_names_1].max()
    pd_selected_for_pred['correct_network_selected_%'] = pd_selected_for_pred['correct_network_selected'] / \
                                                       pd_selected_for_pred['total_samples_seen_for_test'] * 100
    pd_selected_for_pred['correct_class_predicted_%'] = pd_selected_for_pred['correct_class_predicted'] / \
                                                      pd_selected_for_pred['total_samples_seen_for_test'] * 100
    pd_selected_for_pred = pd_selected_for_pred.groupby(['training_exp'])[['correct_network_selected_%', 'correct_class_predicted_%']].max()
    # print(pd_selected_for_pred.to_string())
    pd_selected_for_pred.plot(kind='bar', stacked=False, ax=ax)
    ax.legend()


def plot_fozen_nw_stats(csv_file, eval_csv_file, excel_writer, d, ax):
    columns = ['training_exp', 'dumped_at', 'detected_task_id', 'list_type', 'this_name', 'this_frozen_id', 'this_id', 'this_correctly_predicted_task_ids_test', 'correct_network_selected', 'correct_class_predicted', 'total_samples_seen_for_test']

    df = pd.read_csv(csv_file)
    df_frozen = df.query('dumped_at.str.contains("after_eval") and list_type.str.contains("frozen_net")',
                         engine='python')

    col_names_2 = ['training_exp',  'this_frozen_id']
    col_names_3 = []

    df_frozen = df_frozen.sort_values(by=['training_exp'])
    df_frozen = df_frozen[columns]

    for t in df['training_exp'].unique():
        t_id = str(t)
        df_frozen[t_id] = 0
        col_names_2.append(t_id)
        col_names_3.append(t_id)
    no_of_instances_per_each_task = df[df['total_samples_seen_for_test'] != 0]['total_samples_seen_for_test'].min()
    # print('min:', c)
    for i in df_frozen.index.values:
        e = df_frozen.at[i, 'training_exp']
        t_ids = ast.literal_eval(df_frozen.at[i, 'this_correctly_predicted_task_ids_test'])
        for t, val in t_ids.items():
            # print(e, t, val)
            if t <= e:
                total_instances_seen_for_task_t_at_x = ((e - t) + 1) * no_of_instances_per_each_task
                df_frozen.at[i, str(t)] = val/total_instances_seen_for_task_t_at_x * 100 if total_instances_seen_for_task_t_at_x != 0 else 0.0

    ax.set_ylabel(d)
    pd_f = df_frozen[df_frozen['list_type'] == 'frozen_net']
    pd_f = pd_f[col_names_2].groupby(['training_exp',  'this_frozen_id'])[col_names_3].max()
    # print(pd_f.to_string())
    pd_f.plot(kind='bar', stacked=False, ax=ax)
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha('right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    last_t_id = df['training_exp'].max()
    df_frozen_nw_at_end = df.query(
        'dumped_at.str.contains("after_eval") and list_type.str.contains("frozen_net") and training_exp==' + str(
            last_t_id),
        engine='python')
    frozen_nw_count_at_end = df_frozen_nw_at_end['training_exp'].count()
    df_frozen_nw_at_end = df_frozen_nw_at_end[[
        'training_exp', 'dumped_at', 'detected_task_id', 'list_type', 'this_frozen_id',
        'samples_per_each_task_at_train', 'this_seen_task_ids_train', 'this_correctly_predicted_task_ids_test_at_last', 'this_correctly_predicted_task_ids_probas_test_at_last',
        'correct_network_selected_count_at_last', 'instances_per_task_at_last']]
    df_frozen_nw_at_end.to_excel(excel_writer, sheet_name=d+'_frozen_last', index=False)

    y_max = np.max(pd_f.max())

    df_eval = pd.read_csv(eval_csv_file)
    avg_acc_after_last = round(df_eval[df_eval['training_exp'] == last_t_id]['eval_accuracy'].mean() * 100, 2)
    df_eval = df_eval[df_eval['training_exp'] == last_t_id ]
    df_eval[df_eval['forgetting'] != 0]
    avg_forgetting_after_last = round(df_eval['forgetting'].mean(), 2)
    ax.annotate('avg acc after last ' + str( avg_acc_after_last) + '%, avg forgetting after last ' + str( avg_forgetting_after_last) + ', frozen nets at the end=' + str(frozen_nw_count_at_end), (0, y_max))


def plot_fozen_nw_stats_at_the_end(csv_file, excel_writer, d, ax):
    df = pd.read_csv(csv_file)
    df_frozen = df.query('dumped_at.str.contains("after_eval") and list_type.str.contains("frozen_net")',
                         engine='python')
    last_t_id = df['training_exp'].max()
    df_frozen_nw_at_end = df.query(
        'dumped_at.str.contains("after_eval") and list_type.str.contains("frozen_net") and training_exp==' + str(
            last_t_id),
        engine='python')

    df_frozen_nw_at_end = df_frozen_nw_at_end[[
        'training_exp', 'dumped_at', 'detected_task_id', 'list_type', 'this_frozen_id',
        'samples_per_each_task_at_train', 'this_seen_task_ids_train', 'this_correctly_predicted_task_ids_test_at_last', 'this_correctly_predicted_task_ids_probas_test_at_last',
        'correct_network_selected_count_at_last', 'instances_per_task_at_last']]

    df_empty = pd.DataFrame({'this_frozen_id': [], 'type': []})
    task_ids = []
    for t in df['training_exp'].unique():
        task_ids.append(t)

    frozen_idx = {}
    f_idx = 0
    for f in df_frozen_nw_at_end['this_frozen_id'].unique():
        frozen_idx[f] = str(f_idx)
        f_idx += 1

    group_cols = {
        # 'samples_per_each_task_at_train': 'at_tr',
        'this_seen_task_ids_train': '_t',
        'this_correctly_predicted_task_ids_test_at_last': 'y^',
        'this_correctly_predicted_task_ids_probas_test_at_last': 'p'}

    for i in df_frozen_nw_at_end.index.values:
        for k, c in group_cols.items():
            t_ids = ast.literal_eval(df_frozen.at[i, k])
            t_ids['type'] = c
            t_ids['this_frozen_id'] = frozen_idx[df_frozen.at[i, 'this_frozen_id']]
            tmp_df = pd.DataFrame.from_records([t_ids])
            df_empty = df_empty.append(tmp_df, ignore_index=True)

    df_empty.to_excel(excel_writer, sheet_name=d+'_frozen_last', index=False)
    df_empty = df_empty.groupby(['this_frozen_id', 'type'])[task_ids].max()
    ax.set_ylabel(d)
    df_empty.plot(kind='bar', stacked=False, ax=ax)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# Start of the main
def main():
    exp = get_experiment_dir().split('_')[-1].split('/')[0]
    fig_1 = plt.figure(constrained_layout=False, figsize=(18, 10))
    fig_1.suptitle('task detection (at train) experiment# ' + exp)
    # fig_2 = plt.figure(constrained_layout=False, figsize=(18, 10))
    # fig_2.suptitle('correct nw selected Vs. correct class detected (at eval) experiment# ' + exp)
    # fig_3 = plt.figure(constrained_layout=False, figsize=(18, 10))
    # fig_3.suptitle('correct task_id predicted % by each frozen nw, after training on each task (at eval) experiment# '
    #                + exp + '\n (trained task, frozen at task id_ detected id_nw id)')
    fig_4 = plt.figure(constrained_layout=False, figsize=(18, 10))
    fig_4.suptitle('For each task, accumulated in-class (1s) predictions (y^) and probabilities (p) by each frozen network(at test),'
                   '\nalso the number of instances seen by each frozen network during training.'
                   # '\nexperiment# ' + exp
                   )

    gs_1 = fig_1.add_gridspec(len(datasets), 1)
    # gs_2 = fig_2.add_gridspec(len(datasets), 1)
    # gs_3 = fig_3.add_gridspec(len(datasets), 1)
    gs_4 = fig_4.add_gridspec(len(datasets), 1)
    rows = 0
    cols = 0
    with pd.ExcelWriter(args.resultsDir + '/NW_used_for_prediction_'+exp+'.xlsx') as excel_writer:
        for d_name in datasets:
            net_csv_file = get_net_csv_file(d_name)
            if net_csv_file is None:
                continue

            eval_csv_file = get_eval_csv_file(d_name)

            ax_1 = fig_1.add_subplot(gs_1[rows, cols], label=d_name)
            # ax_2 = fig_2.add_subplot(gs_2[rows, cols], label=d_name)
            # ax_3 = fig_3.add_subplot(gs_3[rows, cols], label=d_name)
            ax_4 = fig_4.add_subplot(gs_4[rows, cols], label=d_name)
            plot_task_detection(net_csv_file, d_name, ax_1)
            # plot_network_selection(net_csv_file, excel_writer, d_name, ax_2)
            # plot_fozen_nw_stats(net_csv_file, eval_csv_file, excel_writer, d_name, ax_3)
            plot_fozen_nw_stats_at_the_end(net_csv_file, excel_writer, d_name, ax_4)
            rows += 1

    ax_4.set_xlabel('(frozen_nn_id, type)' +
                    '\nFor given task, types: y^ = Accumulated in-class prediction(1)s at test,'
                    ' p = Accumulated in-class probabilities at test,'
                    ' _t = number of instances seen at train')
    mplcursors.cursor(hover=True)
    fig_1.savefig(args.resultsDir+'/TaskDetection_' + exp + '.png')
    # fig_2.savefig(args.resultsDir+'/CorrectNWSelected_' + exp + '.png')
    # fig_3.savefig(args.resultsDir+'/CorrectTaskIDPredicted_' + exp + '.png')
    fig_4.savefig(args.resultsDir + '/CorrectTaskIDPredicted_at_the_end_' + exp + '.png')
    plt.show()


main()
