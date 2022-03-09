import subprocess
from itertools import cycle
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from numpy import interp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultsDir", type=str, help="Results directory", default='/Users/ng98/Desktop/results/results/reset_oneclass_reset_loss_estd_include_best_tr_seed_0_no_task_detect_WITH_ACCUMULATED_STATIC_FEATURES_NB/logs/exp_logs')

parser.add_argument('--task_predictor', type=str, default='NB',
                    choices=['NB', 'HT', 'OC'],
                    help='Task predictor to plot AUC for: '
                         'NB, HT, OC')
args = parser.parse_args()


def plot_roc_cur(ax, fper, tper, auc, title_prefix):
    ax.plot(fper, tper, color='deeppink', label='ROC curve (area = {0:0.2f})'.format(auc))
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title_prefix)
    ax.legend()


def auc_for_multi_class(y_test, y_prob):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(
        y_test, y_prob, multi_class="ovo", average="weighted"
    )
    macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    weighted_roc_auc_ovr = roc_auc_score(
        y_test, y_prob, multi_class="ovr", average="weighted"
    )
    print(
        "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
        "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
    )
    print(
        "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
        "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
    )
    return macro_roc_auc_ovo, weighted_roc_auc_ovo, macro_roc_auc_ovr, weighted_roc_auc_ovr


def read_file_plot_roc_cur_auc(file_name, ax, title_prefix, numpy_file=False):
    print(title_prefix)

    if numpy_file:
        r = np.load(file_name)
        y = r[:, -1]
        p = r[:, 0:-1]

        n_classes = p.shape[1]
        # data clean up
        idxs = np.where(p.sum(axis=1) != 1.0)
        for i in idxs[0]:
            delta = 1.0 - p[i].sum()
            for j in range(n_classes):
                p[i, j] += delta/n_classes

        macro_roc_auc_ovo, weighted_roc_auc_ovo, macro_roc_auc_ovr, weighted_roc_auc_ovr = auc_for_multi_class(y, p)

        # Compute ROC curve and ROC area for each class
        y = pd.get_dummies(y).to_numpy()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], p[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), p.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        lw = 2
        # ax.figure()
        ax.plot(
            fpr["micro"],
            tpr["micro"],
            label="ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            # linestyle=":",
            # linewidth=4,
        )

        # ax.plot(
        #     fpr["macro"],
        #     tpr["macro"],
        #     label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        #     color="navy",
        #     linestyle=":",
        #     linewidth=4,
        # )
        # print ROC for each one class task detector
        # colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "blue", "pink", "yellow", "grey", "darkgreen", "brown"])
        # for i, color in zip(range(n_classes), colors):
        #     ax.plot(
        #         fpr[i],
        #         tpr[i],
        #         color=color,
        #         lw=lw,
        #         label="ROC curve of task {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        #     )

        ax.plot([0, 1], [0, 1], "k--", lw=lw)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title('{}'.format(title_prefix))
        # ax.set_title("Some extension of Receiver operating characteristic to multiclass")
        ax.legend(loc="lower right")
    else:
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
# fig.suptitle(exp)
gs = fig.add_gridspec(len(datasets), 1)
rows = 0
col = 0
for d in datasets:
    ax = fig.add_subplot(gs[rows, col], label=d)
    f = None

    file_pattern = "'*_Nets_" + args.task_predictor + (".csv'" if args.task_predictor == "OC" else ".npy'")
    numpy_file = False if args.task_predictor == "OC" else True
    command = subprocess.Popen("find " + args.resultsDir + " -iname " + file_pattern + " | grep " + d,
                               shell=True, stdout=subprocess.PIPE)

    for line in command.stdout.readlines():
        f = line.decode("utf-8").replace('\n', '')
        print(f)
        read_file_plot_roc_cur_auc(f, ax, d, numpy_file=numpy_file)

    rows += 1

plt.show()
