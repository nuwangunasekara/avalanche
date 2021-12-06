import subprocess

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultsDir", type=str, help="Results directory", default='/Users/ng98/Desktop/avalanche_nuwan_fork/exp_scripts/logs/exp_logs/')
args = parser.parse_args()


def plot_roc_cur(ax, fper, tper, auc, title_prefix):
    ax.plot(fper, tper, color='orange', label='ROC')
    ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title_prefix + ': ROC Curve. AUC = ' + str(round(auc, 3)))
    ax.legend()


def read_file_plot_roc_cur_auc(file_name, ax, title_prefix):
    df = pd.read_csv(file_name)

    y = df['is_nw_trained_on_task_id']
    decision_function = df['one_class_df']
    roc_auc = roc_auc_score(y, decision_function)
    print(roc_auc)

    # p = 1/(1 + np.exp(-decision_function))
    #
    # fper, tper, thresholds = roc_curve(y, p)
    # plot_roc_cur(fper, tper)

    fper, tper, thresholds = roc_curve(y, decision_function)
    plot_roc_cur(ax, fper, tper, roc_auc, title_prefix)


datasets = ('CORe50', 'RotatedMNIST', 'RotatedCIFAR10')
fig = plt.figure(constrained_layout=True, figsize=(5, 10))
command = subprocess.Popen('pwd | xargs basename',
                           shell=True, stdout=subprocess.PIPE)
for line in command.stdout.readlines():
    exp = line.decode("utf-8").replace('\n', '')
    print(exp)
fig.suptitle(exp)
gs = fig.add_gridspec(len(datasets), 1)
rows = 0
col = 0
for d in datasets:
    ax = fig.add_subplot(gs[rows, col], label=d)
    command = subprocess.Popen("find " + args.resultsDir + " -iname '*Nets_TD.csv' | grep " + d,
                               shell=True, stdout=subprocess.PIPE)
    for line in command.stdout.readlines():
        f = line.decode("utf-8").replace('\n', '')
        read_file_plot_roc_cur_auc(f, ax, d)
    rows += 1

plt.show()
