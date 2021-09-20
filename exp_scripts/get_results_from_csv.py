import pandas as pd
import re
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultsDir", type=str, help="Results directory", default='/Users/ng98/Desktop/avalanche_test/exp/logs')
args = parser.parse_args()

csv_files = []
if len(csv_files) == 0:
    command = subprocess.Popen("find " + args.resultsDir + "/csv_data/ -iname 'eval_results.csv'",
                               shell=True, stdout=subprocess.PIPE)
    for line in command.stdout.readlines():
        csv_files.append(line.decode("utf-8").replace('\n', ''))
print(csv_files)

print('dataset,strategy,sub_strategy,eval_accuracy,forgetting')
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
    eval_after_training_last = df.query("eval_exp == 0 & training_exp == " + str(df['training_exp'].max()))
    eval_accuracy = eval_after_training_last.iloc[0, eval_after_training_last.columns.get_loc('eval_accuracy')]
    forgetting = eval_after_training_last.iloc[0, eval_after_training_last.columns.get_loc('forgetting')]

    new_row = {'dataset': dataset, 'strategy': strategy, 'sub_strategy': sub_strategy, 'eval_accuracy': sub_strategy, 'forgetting': forgetting}

    df_final_results = df_final_results.append(new_row, ignore_index=True)
    print('{},{},{},{},{}'.format(
        dataset,
        strategy,
        sub_strategy,
        eval_accuracy,
        forgetting))

df_final_results.to_csv(args.resultsDir+'/Final_results.csv', header=True, index=False)
