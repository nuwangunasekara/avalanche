import pandas as pd
import numpy as np
import re
import subprocess
import argparse
from get_frozen_nets import get_net_info

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

    tmp_pd = df.query("eval_exp == 0 and training_exp == " + last_exp_id_str).copy(deep=True)
    tmp_pd.loc[:, 'dataset'] = dataset
    tmp_pd.loc[:, 'strategy'] = strategy
    tmp_pd.loc[:, 'sub_strategy'] = sub_strategy
    tmp_pd.loc[:, 'correct_net_percentage'] = correct_net

    df_final_results = df_final_results.append(tmp_pd, ignore_index=True)

print(df_final_results)

df_final_results.to_csv(args.resultsDir+'/Final_results.csv', header=True, index=False)

pd_forgetting = pd.pivot_table(
    df_final_results, values='forgetting', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)
pd_eval_accuracy = pd.pivot_table(
    df_final_results, values='eval_accuracy', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)
pd_correct_net = pd.pivot_table(
    df_final_results, values='correct_net_percentage', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)

pd_forgetting_std = pd.pivot_table(
    df_final_results, values='forgetting', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)
pd_eval_accuracy_std = pd.pivot_table(
    df_final_results, values='eval_accuracy', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)
pd_correct_net_std = pd.pivot_table(
    df_final_results, values='correct_net_percentage', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)

with pd.ExcelWriter(args.resultsDir+'/Final_results.xlsx') as writer:
    df_final_results.to_excel(writer, sheet_name='RawResults', index=True)
    pd_eval_accuracy.to_excel(writer, sheet_name='AccAfterLast', index=True)
    pd_eval_accuracy_std.to_excel(writer, sheet_name='AccAfterLastS', index=True)
    pd_forgetting.to_excel(writer, sheet_name='Forgetting', index=True)
    pd_forgetting_std.to_excel(writer, sheet_name='ForgettingS', index=True)
    pd_correct_net.to_excel(writer, sheet_name='NetSelect', index=True)
    pd_correct_net_std.to_excel(writer, sheet_name='NetSelectS', index=True)

# df_frozen, df_correct = get_net_info()
# print(df_frozen)
# print(df_correct)

