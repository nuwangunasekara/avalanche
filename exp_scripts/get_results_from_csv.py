import pandas as pd
import numpy as np
import re
import subprocess
import argparse
from get_frozen_nets import get_net_info
from matplotlib.pylab import plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultsDir", type=str, help="Results directory", default='/Users/ng98/Desktop/avalanche_test/results/')
args = parser.parse_args()

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
    log_file = csv_file.replace('/csv_data/', '/exp_logs/').replace('/eval_results.csv', '')
    print(log_file)
    df_frozen, df_correct = get_net_info(log_file)
    df = pd.read_csv(csv_file)
    last_exp_id = df['training_exp'].max()
    last_exp_id_str = str(last_exp_id)
    exp_info = csv_file.split('/')[-2]
    dataset = exp_info.split('_')[0]
    strategy = exp_info.split('_')[1]
    sub_strategy = exp_info.split('_')[-1]
    match = re.search('ONE_CLASS.*|MAJORITY_VOTE|RANDOM|NAIVE_BAYES.*|TASK_ID_KNOWN', exp_info)
    correct_net = 0
    if match:
        sub_strategy = match.group(0)
        correct_net = df_correct.query("training_exp == " + last_exp_id_str).loc[last_exp_id, 'correct_net_percentage']
        print(df_frozen)
        print(df_correct)
    else:
        sub_strategy = 'NA'

    tmp_df_all = df.copy(deep=True)
    tmp_df_all.loc[:, 'dataset'] = dataset
    tmp_df_all.loc[:, 'strategy'] = strategy
    tmp_df_all.loc[:, 'sub_strategy'] = sub_strategy
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

print(df_final_results)

df_final_results.to_csv(args.resultsDir+'/Final_results.csv', header=True, index=False)

pd_forgetting = pd.pivot_table(
    df_final_results, values='forgetting', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)
pd_eval_accuracy = pd.pivot_table(
    df_final_results, values='eval_accuracy', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)
pd_avg_forgetting = pd.pivot_table(
    df_final_results, values='avg_eval_forgetting_after_last', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)
pd_avg_eval_accuracy = pd.pivot_table(
    df_final_results, values='avg_eval_accuracy_after_last', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)
pd_correct_net = pd.pivot_table(
    df_final_results, values='correct_net_percentage', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)

pd_forgetting_std = pd.pivot_table(
    df_final_results, values='forgetting', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)
pd_eval_accuracy_std = pd.pivot_table(
    df_final_results, values='eval_accuracy', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)
pd_avg_forgetting_std = pd.pivot_table(
    df_final_results, values='avg_eval_forgetting_after_last', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)
pd_avg_eval_accuracy_std = pd.pivot_table(
    df_final_results, values='avg_eval_accuracy_after_last', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)
pd_correct_net_std = pd.pivot_table(
    df_final_results, values='correct_net_percentage', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)

with pd.ExcelWriter(args.resultsDir+'/Final_results.xlsx') as writer:
    df_final_results.to_excel(writer, sheet_name='RawResults', index=True)

    pd_eval_accuracy.to_excel(writer, sheet_name='AccAfterLast', index=True)
    pd_eval_accuracy_std.to_excel(writer, sheet_name='AccAfterLastS', index=True)
    pd_forgetting.to_excel(writer, sheet_name='Forgetting', index=True)
    pd_forgetting_std.to_excel(writer, sheet_name='ForgettingS', index=True)

    pd_avg_eval_accuracy.to_excel(writer, sheet_name='AvgAccAfterLast', index=True)
    pd_avg_eval_accuracy_std.to_excel(writer, sheet_name='AvgAccAfterLastS', index=True)
    pd_avg_forgetting.to_excel(writer, sheet_name='AvgForgetting', index=True)
    pd_avg_forgetting_std.to_excel(writer, sheet_name='AvgForgettingS', index=True)

    pd_correct_net.to_excel(writer, sheet_name='NetSelect', index=True)
    pd_correct_net_std.to_excel(writer, sheet_name='NetSelectS', index=True)

fig = plt.figure(constrained_layout=False)
datasets = df_all["dataset"].unique()
experiences = df_all['eval_exp'].unique()
strategies = df_all["strategy"].unique()
sub_strategies = df_all["sub_strategy"].unique()
line = ['dotted', 'dashed', 'dashdot', 'solid']
line = ['solid', 'dashdot', 'dashed', 'dotted']
colors = {'EWC': 'black', 'GDumb': 'dimgrey', 'LwF': 'darkgrey',
          'MAJORITY_VOTE': 'darkorange',
          'NAIVE_BAYES': 'greenyellow', 'NAIVE_BAYES_end': 'forestgreen',
          'ONE_CLASS': 'dodgerblue', 'ONE_CLASS_end': 'blue',
          'RANDOM': 'gold',
          'TASK_ID_KNOWN': 'red'}

gs = fig.add_gridspec(len(datasets), len(experiences))
rows = 0
axes = []
for d in datasets:
    col = 0
    for e in experiences:
        ax = fig.add_subplot(gs[rows, col], label=d)
        ax.set_title(d)
        axes.append(ax)
        for s in strategies:
            for sub_s in sub_strategies:
                if (s == 'TrainPool' and sub_s == 'NA') or ((s == 'EWC' or s == 'LwF' or s == 'GDumb') and sub_s != 'NA'):
                    continue

                p_df = df_all.query(
                    'dataset .str.contains("' + d + '") and strategy.str.contains("' + s + '") and sub_strategy.str.contains("' + sub_s +'") and eval_exp == ' + str(e), engine='python')
                # label = s + "_" + sub_s + "_" + str(e)
                label = s
                if s == 'TrainPool':
                    label += sub_s
                color = colors[sub_s] if s == 'TrainPool' else colors[s]
                # line_type = line[ll]
                line_type = line[0]
                exps = p_df['training_exp'].unique()
                p_df_avg_eval_acc_for_exp = p_df.groupby(['training_exp'])['eval_accuracy'].mean()

                print(d, s, sub_s, e, color, line_type)
                print(exps)
                print(p_df_avg_eval_acc_for_exp)

                ax.plot(exps, p_df_avg_eval_acc_for_exp, label=label, color=color, linestyle=line_type, marker="o")
        col += 1
    rows += 1
axes[-4].legend(ncol=5, bbox_to_anchor=(0.0, -0.1), loc="upper left")
plt.show()