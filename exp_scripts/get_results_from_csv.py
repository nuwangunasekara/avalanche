import os

import mplcursors
import pandas as pd
import numpy as np
import re
import subprocess
import argparse
from get_frozen_nets import get_net_info
import matplotlib
from matplotlib.pylab import plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultsDir", type=str, help="Results directory", default='/Users/ng98/Desktop/results/results/test/1CNN')

parser.add_argument('--plot_eval_after_training_each_task', dest='plot_eval_after_training_each_task',
                    action='store_true')
parser.add_argument('--no-plot_eval_after_training_each_task', dest='plot_eval_after_training_each_task',
                    action='store_false')
parser.set_defaults(plot_eval_after_training_each_task=True)

parser.add_argument('--plot_avg', dest='plot_avg',
                    action='store_true')
parser.add_argument('--no-plot_avg', dest='plot_avg',
                    action='store_false')
parser.set_defaults(plot_avg=False)

args = parser.parse_args()

args.resultsDir = os.path.abspath(args.resultsDir)
top_dirs = os.listdir(args.resultsDir)
print (top_dirs)

fig_title = None
csv_files = []
if len(csv_files) == 0:
    command = subprocess.Popen("find " + args.resultsDir + " -iname '*eval_results.csv'",
                               shell=True, stdout=subprocess.PIPE)
    for line in command.stdout.readlines():
        csv_files.append(line.decode("utf-8").replace('\n', ''))
print(csv_files)

df_final_results = pd.DataFrame(columns=['dataset', 'strategy', 'sub_strategy', 'eval_accuracy', 'forgetting', 'correct_net_percentage'])
df_all = pd.DataFrame(columns=['dataset', 'strategy', 'sub_strategy', 'eval_accuracy', 'forgetting', 'correct_net_percentage'])

for csv_file in csv_files:
    print(csv_file)
    log_file = csv_file.replace('/csv_data/', '/exp_logs/').replace('/eval_results.csv', '')
    df_frozen, df_correct = get_net_info(log_file)
    df = pd.read_csv(csv_file)
    last_exp_id = df['training_exp'].max()
    last_exp_id_str = str(last_exp_id)
    exp_info = csv_file.split('/')[-2]
    dataset = exp_info.split('_')[0]
    strategy = exp_info.split('_')[1]
    sub_strategy = exp_info.split('_')[-1]
    correct_net = 0.0

    if exp_info.find('NAIVE_BAYES_end') >= 0 or exp_info.find('ONE_CLASS_end') >= 0:
        initial_pattern = 'ONE_CLASS_end|NAIVE_BAYES_end'
    else:
        initial_pattern = 'ONE_CLASS|NAIVE_BAYES|HT'
    match = re.search('(' + initial_pattern + '|MAJORITY_VOTE|RANDOM|TASK_ID_KNOWN|SimpleCNN|CNN4)(.*)', exp_info)
    if match:
        sub_strategy = match.group(1)
        if len(match.group(2)) > 0 and (sub_strategy != match.group(2)):
            if fig_title is None:
                fig_title = match.group(2)
        if strategy == 'TrainPool':
            # correct_net = df_correct.query("training_exp == " + last_exp_id_str).loc[last_exp_id, 'correct_net_percentage']
            # print(df_frozen)
            # print(df_correct)
            pass
    else:
        sub_strategy = 'NA'

    top_dir_name = ''
    top_dir_index = len(os.path.abspath(args.resultsDir).split('/'))
    for top_dir in top_dirs:
        # for dir_name in csv_file.split('/'):
        if top_dir == csv_file.split('/')[top_dir_index]:
            top_dir_name = top_dir
            break

    tmp_df_all = df.copy(deep=True)
    print(dataset)
    tmp_df_all.loc[:, 'dataset'] = dataset
    tmp_df_all.loc[:, 'strategy'] = strategy
    tmp_df_all.loc[:, 'sub_strategy'] = sub_strategy
    tmp_df_all.loc[:, 'top_dir'] = top_dir
    tmp_df_all.loc[:, 'correct_net_percentage'] = correct_net
    df_all = df_all.append(tmp_df_all, ignore_index=True)

    tmp_pd = df.query("eval_exp == 0 and training_exp == " + last_exp_id_str).copy(deep=True)
    tmp_pd.loc[:, 'dataset'] = dataset
    tmp_pd.loc[:, 'strategy'] = strategy
    tmp_pd.loc[:, 'sub_strategy'] = sub_strategy
    tmp_pd.loc[:, 'correct_net_percentage'] = correct_net
    tmp_pd.loc[:, 'avg_eval_accuracy_after_last'] = df.query("training_exp == " + last_exp_id_str)['eval_accuracy'].mean()
    tmp_pd.loc[:, 'avg_eval_forgetting_after_last'] = df.query("eval_exp != " + last_exp_id_str + " and training_exp == " + last_exp_id_str)['forgetting'].mean()

    df_final_results = df_final_results.append(tmp_pd, ignore_index=True)


df_final_results.to_csv(args.resultsDir+'/Results.csv', header=True, index=False)

pd_avg_forgetting = pd.pivot_table(
    df_final_results, values='avg_eval_forgetting_after_last', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)
pd_avg_eval_accuracy = pd.pivot_table(
    df_final_results, values='avg_eval_accuracy_after_last', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)
pd_correct_net = pd.pivot_table(
    df_final_results, values='correct_net_percentage', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)

pd_avg_forgetting_std = pd.pivot_table(
    df_final_results, values='avg_eval_forgetting_after_last', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)
pd_avg_eval_accuracy_std = pd.pivot_table(
    df_final_results, values='avg_eval_accuracy_after_last', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)
pd_correct_net_std = pd.pivot_table(
    df_final_results, values='correct_net_percentage', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)

with pd.ExcelWriter(args.resultsDir+'/Results.xlsx') as writer:
    pd_avg_eval_accuracy.to_excel(writer, sheet_name='AvgAccAfterLast', index=True)
    pd_avg_eval_accuracy_std.to_excel(writer, sheet_name='AvgAccAfterLastS', index=True)
    pd_avg_forgetting.to_excel(writer, sheet_name='AvgForgetting', index=True)
    pd_avg_forgetting_std.to_excel(writer, sheet_name='AvgForgettingS', index=True)

    pd_correct_net.to_excel(writer, sheet_name='NetSelect', index=True)
    pd_correct_net_std.to_excel(writer, sheet_name='NetSelectS', index=True)

    df_final_results.to_excel(writer, sheet_name='RawResults', index=True)

datasets = df_all["dataset"].unique()
datasets.sort()
strategies = df_all["strategy"].unique()
sub_strategies = df_all["sub_strategy"].unique()
top_dirs = df_all["top_dir"].unique()

line = ['solid', 'dashdot', 'dashed', 'dotted']
colors = {'EWC': 'black',
          'LwF': 'darkgrey',
          'ER': 'darkviolet',
          'GDumb': 'violet',
          'MAJORITY_VOTE': 'darkorange',
          'NAIVE_BAYES': 'greenyellow',
          'NAIVE_BAYES_end': 'forestgreen',
          'HT': 'fuchsia',
          'ONE_CLASS': 'dodgerblue',
          'ONE_CLASS_end': 'blue',
          'RANDOM': 'gold',
          'TASK_ID_KNOWN': 'red'}
colors2 = {'black',
          'darkgrey',
          'darkviolet',
          'violet',
          'darkorange',
          'greenyellow',
          'forestgreen',
          'fuchsia',
          'dodgerblue',
          'blue',
          'gold',
          'red'}
used_colors = []
already_ploted = {}
already_ploted_count = {}

figs = []
last_d = None
for d in datasets:
    fig = plt.figure(constrained_layout=False, figsize=(18, 10))
    figs.append(fig)
    axes = []
    experiences = df_all[df_all['dataset'].isin([d])]
    experiences = experiences['eval_exp'].unique()
    experiences.sort()
    gs = fig.add_gridspec(1, len(experiences))
    rows = 0
    col = 0
    # RotatedMNIST RotatedCIFAR10 CORe50
    if d == 'CORe50':
        y_max=0.80
        y_min=0.17
    elif d == 'RotatedCIFAR10':
        y_max=0.55
        y_min=0.30
    elif d == 'RotatedMNIST':
        y_max=1.00
        y_min=0.25
    else:
        y_max=1.00
        y_min=0.00

    for e in experiences:
        ax = fig.add_subplot(gs[rows, col], label=d)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('gray')
        ax.spines['left'].set_color('gray')
        if d == datasets[0]:
            ax.set_title('task ' + str(int(e)))
        if d == datasets[len(datasets)-1]:
            ax.set_xlabel('after training task')
        if last_d != d:
            ax.set_ylabel('acc (' + d + ')')
        last_d = d
        ax.set_xticks(
            np.arange(len(df_all.query('dataset .str.contains("' + d + '")', engine='python')['eval_exp'].unique())))
        ax.set_ylim(y_min, y_max)
        for s in strategies:
            for sub_s in sub_strategies:
                # ONE_CLASS. * | MAJORITY_VOTE | RANDOM | NAIVE_BAYES. * | TASK_ID_KNOWN | SimpleCNN | CNN4
                if (s == 'TrainPool' and (sub_s == 'SimpleCNN' or sub_s == 'CNN4')) or \
                        (s != 'TrainPool' and not (sub_s == 'SimpleCNN' or sub_s == 'CNN4')):
                    # we have already plot above s, skipp "SimpleCNN | CNN4" for TrainPool, and "ONE_CLASS. * | MAJORITY_VOTE | RANDOM | NAIVE_BAYES. * | TASK_ID_KNOWN" for others
                    continue
                for top_dir in top_dirs:
                    if s == 'TrainPool' and top_dir == 'ER_SimpleCNN':
                        continue
                    p_df = df_all.query(
                        'dataset .str.contains("' + d + '") and strategy.str.contains("' + s + '") and sub_strategy.str.contains("' + sub_s +'") and top_dir.str.contains("'+ top_dir +'") and eval_exp == ' + str(e), engine='python')
                    if p_df.empty:
                        continue
                    label = s + sub_s
                    line_type = line[0]
                    if s == 'TrainPool':
                        label = 'ODIN_' + top_dir
                    else:
                        label = s
                        if sub_s == 'SimpleCNN':
                            line_type = line[1]


                    combination = s + (top_dir if s == 'TrainPool' else '')
                    # print(combination, already_ploted)
                    if already_ploted.get(combination, None) is None:
                        color = list(colors2 - set(already_ploted.values()))[0]
                        already_ploted[combination] = color
                        already_ploted_count[combination] = 1
                    else:
                        already_ploted_count[combination] += 1
                        color = already_ploted[combination]
                        # if s == 'TrainPool':
                        #     color = already_ploted[combination]
                        # else:
                        #     # if already_ploted_count[combination] > 2:
                        #     #     continue
                        #     # else:
                        #     #     print(combination)
                        #     pass




                    # color = colors[sub_s] if s == 'TrainPool' else colors[s]

                    #
                    # exps = p_df['training_exp'].unique()
                    exps = p_df['training_exp'].unique()
                    # p_df_avg_eval_acc_for_exp = p_df.groupby(['training_exp'])['eval_accuracy'].mean()
                    p_df_avg_eval_acc_for_exp = p_df.groupby(['training_exp'])['eval_accuracy'].mean()
                    if args.plot_eval_after_training_each_task:
                        ax.plot(exps, p_df_avg_eval_acc_for_exp, label=label, color=color, linestyle=line_type, marker=".")
                    if args.plot_avg:
                        ax.plot(exps, np.ones(len(exps)) * p_df_avg_eval_acc_for_exp.mean(), label=label+'_avg', color=color, linestyle='dotted')
        axes.append(ax)
        col += 1
    rows += 1
    axes[0].legend(ncol=10, bbox_to_anchor=(0.0, -0.05), loc="upper left")
    mplcursors.cursor(hover=True)
    plt.subplots_adjust(left=0.04, right=0.99)
    # fig.suptitle(fig_title)
    plt.savefig(args.resultsDir + '/Per-Task_' + d + '.pdf')
    plt.savefig(args.resultsDir + '/Per-Task_' + d + '.png')
    # plt.savefig(args.resultsDir + '/Per-Task_' + d + '.svg')
    # plt.show()
# input()
