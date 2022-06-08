import subprocess
import argparse
import pandas as pd
from io import StringIO
import ast
import matplotlib.pylab as plt
import mplcursors

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resultsDir", type=str, help="Results directory",
                    # default='~/Desktop/tmp')
                    default='/Users/ng98/Desktop/results/results/1CNN/buffer_5_MV_RR_NWstats/buffer_5_MV_RR_NWstats_1/logs/exp_logs')
args = parser.parse_args()

datasets = ('CORe50', 'RotatedCIFAR10', 'RotatedMNIST')
figs = []
last_d = None
for d in datasets:
    fig = plt.figure(constrained_layout=False, figsize=(18, 10))
    figs.append(fig)
    axes = []
    file_pattern = "'*_Nets.csv'"
    command = subprocess.Popen("find " + args.resultsDir + " -iname " + file_pattern + " | grep " + d,
                               shell=True, stdout=subprocess.PIPE)
    iteration = 0
    for line in command.stdout.readlines():
        f = line.decode("utf-8").replace('\n', '')
        if f.split('/')[-1].find(d) > -1:
            print(f)

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

            # fields_to_plot = ('buffer_used_for_train_count', 'correct_nn_with_lowest_loss_predicted_acc')
            fields_to_plot = ['correct_nn_with_lowest_loss_predicted_acc']
            for col_name in fields_to_plot:
                s = df[col_name].to_string()
                s = s.replace("0    defaultdict(<class 'int'>, ", '').replace(')', '')
                buffer_used_for_train_count_dic = ast.literal_eval(s)
                print(buffer_used_for_train_count_dic)

                experiences = sorted(
                    list(set([int(k.split('_')[1].split('e')[1]) for k in buffer_used_for_train_count_dic.keys()])))

                gs = fig.add_gridspec(1, len(experiences))
                rows = 0
                col = 0
                for e in experiences:
                    ax = fig.add_subplot(gs[rows, col], label=d)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_color('gray')
                    ax.spines['left'].set_color('gray')
                    if d == datasets[0]:
                        ax.set_title('task ' + str(int(e)))
                    if d == datasets[len(datasets) - 1]:
                        ax.set_xlabel('after training task')
                    if last_d != d:
                        ax.set_ylabel('Acc of the TP predicting the frozen NW with lowest loss for given eval instance for each task \n(' + d + ')')
                    last_d = d
                    y_max = 1.00
                    y_min = 0.00
                    ax.set_xticks(experiences)
                    ax.set_ylim(y_min, y_max)
                    training_tasks = [t for t in experiences if t >= e]
                    x_y = [(t, buffer_used_for_train_count_dic['t{}_e{}'.format(t, e)]) for t in training_tasks]
                    x, y = zip(*x_y)
                    ax.plot(x, y,label='iteration {}'.format(iteration), marker=".")
                    axes.append(ax)
                    col += 1
            iteration += 1
    # mplcursors.cursor(hover=True)
    plt.savefig(args.resultsDir + '/TP-Per-Task_Acc_' + d + '.pdf')
    # plt.show()
