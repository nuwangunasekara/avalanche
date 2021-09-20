import pandas as pd
import numpy as np
import re
import subprocess
import argparse

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

df_final_results = pd.DataFrame(columns=['dataset', 'strategy', 'sub_strategy', 'eval_accuracy', 'forgetting'])

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    exp_info = csv_file.split('/')[-2]
    dataset = exp_info.split('_')[0]
    strategy = exp_info.split('_')[1]
    sub_strategy = exp_info.split('_')[-1]
    match = re.search('ONE_CLASS.*|MAJORITY_VOTE|RANDOM|NAIVE_BAYES.*|TASK_ID_KNOWN', exp_info)
    if match:
        sub_strategy = match.group(0)
    else:
        sub_strategy = 'NA'

    last_exp_id = str(df['training_exp'].max())
    tmp_pd = df.query("eval_exp == 0 and training_exp == " + last_exp_id).copy(deep=True)
    tmp_pd.loc[:, 'dataset'] = dataset
    tmp_pd.loc[:, 'strategy'] = strategy
    tmp_pd.loc[:, 'sub_strategy'] = sub_strategy

    df_final_results = df_final_results.append(tmp_pd, ignore_index=True)

print(df_final_results)

df_final_results.to_csv(args.resultsDir+'/Final_results.csv', header=True, index=False)

pd_forgetting = pd.pivot_table(
    df_final_results, values='forgetting', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)
pd_eval_accuracy = pd.pivot_table(
    df_final_results, values='eval_accuracy', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.mean, fill_value=0)

pd_forgetting_std = pd.pivot_table(
    df_final_results, values='forgetting', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)
pd_eval_accuracy_std = pd.pivot_table(
    df_final_results, values='eval_accuracy', index=['dataset'], columns=['strategy', 'sub_strategy'], aggfunc=np.std, fill_value=0)

with pd.ExcelWriter(args.resultsDir+'/Final_results.xlsx') as writer:
    pd_eval_accuracy.to_excel(writer, sheet_name='AccAfterLast', index=True)
    pd_eval_accuracy_std.to_excel(writer, sheet_name='AccAfterLastS', index=True)
    pd_forgetting.to_excel(writer, sheet_name='Forgetting', index=True)
    pd_forgetting_std.to_excel(writer, sheet_name='ForgettingS', index=True)

