import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc_cur(title_prefix, fper, tper, auc):
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title_prefix + ': ROC Curve. AUC = ' + str(round(auc, 3)))
    plt.legend()
    plt.show()


d = 'CORe50'
file_name = '/Users/ng98/Desktop/avalanche_nuwan_fork/exp_scripts/logs/exp_logs/' + d +\
            '_TrainPool_TP_6CNN_6_ONE_CLASS_Nets_TD.csv'
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
plot_roc_cur(d, fper, tper, roc_auc)
