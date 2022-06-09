import os
import subprocess
import argparse
import pandas as pd
from io import StringIO
import ast
import matplotlib.pylab as plt
import mplcursors
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultsDir", type=str, help="Results directory",
                    # default='~/Desktop/tmp')
                    default='/Users/ng98/Desktop/results/results/1CNN/buffer_5_MV_RR_NWstats/buffer_5_MV_RR_NWstats_1/logs/exp_logs')
args = parser.parse_args()

fields_to_plot = {
    'correct_nn_with_lowest_loss_predicted_acc': {
        'column_splitter': 'e',
        'title': 'task',
        'ylabel': 'Acc of the TP predicting the frozen NW with lowest loss for given eval instance for each task',
        'fig_prefix': 'TP',
        'data':{}
    },
    'correct_class_predicted_at_test_acc_after_training_each_task':{
        'column_splitter': 'f',
        'title': 'frozenNW',
        'ylabel': 'Acc of the frozen NWs after training on each task',
        'fig_prefix': 'FrozenNW',
        'data':{}
    }
}

datasets = ('CORe50', 'RotatedCIFAR10', 'RotatedMNIST')

args.resultsDir = os.path.abspath(args.resultsDir)
top_dirs = os.listdir(args.resultsDir)

for d in datasets:
    file_pattern = "'*_Nets.csv'"
    command = subprocess.Popen("find " + args.resultsDir + " -iname " + file_pattern + " | grep " + d,
                               shell=True, stdout=subprocess.PIPE)
    iteration = 0
    for line in command.stdout.readlines():
        f = line.decode("utf-8").replace('\n', '')
        if f.split('/')[-1].find(d) > -1:
            print(f)

            top_dir_index = len(os.path.abspath(args.resultsDir).split('/'))
            for top_dir in top_dirs:
                if top_dir == f.split('/')[top_dir_index]:
                    break

            head_c = subprocess.Popen("head -n 1 " + f, shell=True, stdout=subprocess.PIPE)
            head_lines = head_c.stdout.readlines()
            csv_string = head_lines[0].decode("utf-8").replace('\n', '')
            csv_string += '\n'
            tail_c = subprocess.Popen("tail -n 1 " + f, shell=True, stdout=subprocess.PIPE)
            tail_lines = tail_c.stdout.readlines()
            tail_line = tail_lines[0].decode("utf-8").replace('\n', '')
            # tail_line = tail_line.replace('defaultdict', '"defaultdict')
            # tail_line = tail_line.replace('}),', '})",')
            csv_string += tail_line
            buff = StringIO(csv_string)
            df = pd.read_csv(buff)

            for col_name in fields_to_plot.keys():
                s = df[col_name].to_string()
                s = s.replace("0    defaultdict(<class 'int'>, ", '').replace(')', '')
                buffer_used_for_train_count_dic = ast.literal_eval(s)
                print(buffer_used_for_train_count_dic)

                experiences = sorted(
                    list(set([int(k.split('_')[1].split(fields_to_plot[col_name]['column_splitter'])[1]) for k in buffer_used_for_train_count_dic.keys()])))

                for col in experiences:
                    training_tasks = [t for t in experiences if t >= col]
                    x_y = [(t, buffer_used_for_train_count_dic['t{}_{}{}'.format(t, fields_to_plot[col_name]['column_splitter'], col)]) for t in training_tasks]
                    x, y = zip(*x_y)
                    if d not in fields_to_plot[col_name]['data']:
                        fields_to_plot[col_name]['data'][d] = {}
                    if '{}'.format(col) not in fields_to_plot[col_name]['data'][d]:
                        fields_to_plot[col_name]['data'][d]['{}'.format(col)] = {}
                    if top_dir not in fields_to_plot[col_name]['data'][d]['{}'.format(col)]:
                        fields_to_plot[col_name]['data'][d]['{}'.format(col)][top_dir] = {}
                    if 'x' not in fields_to_plot[col_name]['data'][d]['{}'.format(col)][top_dir]:
                        fields_to_plot[col_name]['data'][d]['{}'.format(col)][top_dir]['x'] = []
                        fields_to_plot[col_name]['data'][d]['{}'.format(col)][top_dir]['y'] = []
                    fields_to_plot[col_name]['data'][d]['{}'.format(col)][top_dir]['x'].append(x)
                    fields_to_plot[col_name]['data'][d]['{}'.format(col)][top_dir]['y'].append(y)

for field in  fields_to_plot.keys():
    for d in fields_to_plot[field]['data'].keys():
        figs = []
        axes = []
        last_d = None
        fig = plt.figure(constrained_layout=False, figsize=(18, 10))
        figs.append(fig)
        gs = fig.add_gridspec(1, len(fields_to_plot[field]['data'][d].keys()))
        for col1 in fields_to_plot[field]['data'][d].keys():
            ax = fig.add_subplot(gs[0, int(col1)], label=d)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('gray')
            ax.spines['left'].set_color('gray')
            ax.set_xlabel('after training task')
            if last_d != d:
                ax.set_ylabel(fields_to_plot[field]['ylabel'] + ' \n(' + d + ')')
            last_d = d
            y_max = 1.00
            y_min = 0.00
            ax.set_ylim(y_min, y_max)
            for dir1 in fields_to_plot[field]['data'][d][col1].keys():
                x = fields_to_plot[field]['data'][d][col1][dir1]['x'][0]
                y = np.array(fields_to_plot[field]['data'][d][col1][dir1]['y']).mean(axis=0)
                ax.plot(x, y,label='{}'.format(dir1), marker=".")
            ax.set_xticks(x)
            ax.set_title('{} {}'.format(fields_to_plot[field]['title'],col1))
            axes.append(ax)
        axes[0].legend(ncol=10, bbox_to_anchor=(0.0, -0.05), loc="upper left")
        mplcursors.cursor(hover=True)
        plt.subplots_adjust(left=0.04, right=0.99)
        plt.savefig(args.resultsDir + '/{}_Per-Task_{}.pdf'.format(fields_to_plot[field]['fig_prefix'], d))
        plt.show()