import pandas as pd
import re
from io import StringIO

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def get_net_info(log_file='/Users/ng98/Desktop/avalanche_test/results/exp1/logs/exp_logs/RotatedCIFAR10_TrainPool_mb_10_TP_6CNN_6_ONE_CLASS_end'):
    file_h = open(log_file, 'r')
    lines = file_h.readlines()
    task_id = 0
    search_for_frozen_nets = False
    read_lines = 0
    print_header = True
    frozen_net_csv_buffer = StringIO('')
    correct_net_csv_buffer = StringIO('')
    correct_net_csv_buffer.write('training_exp,correct_net_percentage\n')
    for line in lines:
        if print_header:
            match = re.search('total_samples_seen_for_train,', line)
            if match:
                header = line.replace('\n', '')
                frozen_net_csv_buffer.write('{},after_training_task\n'.format(header))
                print_header = False

        match = re.search('Eval on experience (' + str(task_id) + ') (.*) (from test stream ended)', line)
        if match:
            # print(match.group(1), match.group(2), match.group(3))
            search_for_frozen_nets = True
            task_id += 1
        match = re.search('---frozen_nets---', line)
        if search_for_frozen_nets and match:
            read_lines = task_id + 1
        if read_lines > 0:
            read_lines -= 1
            if not match:
                frozen_net_csv_buffer.write('{},{}\n'.format(line.replace('\n', ''), task_id - 1))
        match = re.search('correct_network_selected\(%\)= (.*)', line)
        if match:
            correct_net_percentage = match.group(1)
            correct_net_csv_buffer.write('{},{}\n'.format(task_id - 1, correct_net_percentage))
            search_for_frozen_nets = False
    frozen_net_csv_buffer_size = frozen_net_csv_buffer.tell()
    correct_net_csv_buffer_size = correct_net_csv_buffer.tell()
    frozen_net_csv_buffer.seek(0)
    correct_net_csv_buffer.seek(0)
    df_frozen = pd.read_csv(frozen_net_csv_buffer) if frozen_net_csv_buffer_size > 0 else None
    df_correct = pd.read_csv(correct_net_csv_buffer) if correct_net_csv_buffer_size > 0 else None
    frozen_net_csv_buffer.close()
    correct_net_csv_buffer.close()
    return df_frozen, df_correct



