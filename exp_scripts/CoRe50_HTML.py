import os
import random
import shutil
import subprocess

base_dir = '/Users/ng98/Desktop/avalanche_test/'
core50_dir = '/Users/ng98/.avalanche/data/core50/'
image_dir = core50_dir + 'core50_128x128'

head = '<!DOCTYPE html><html><head><title>CoRe50</title></head><body>'
tail ='</body></html>'


def print_table_NI_DI_cat_task_id_by_session(f, show_only_fist_object_of_the_class=False):
    fd.write('<h3>{} from each category/class</h3>'.format('Only first object type' if show_only_fist_object_of_the_class else 'All object types'))

    f.write('<table border="1">\n')
    f.write('<tr>')
    for o in range (51):
        if o == 0:
            f.write('<td></td>')
        q, mod = divmod(o, 5)
        if mod == 1:
            f.write('<td>{}</td>'.format('class ' + str(q)))
        else:
            continue

    f.write('</tr>')
    for s in range(1, 12):
        f.write('<tr>\n<td>task {}</td>'.format(s))
        for o in range(1, 51):
            q, mod = divmod(o, 5)

            if mod == 1:
                f.write('<td>')

            if (show_only_fist_object_of_the_class and mod == 1) or (not show_only_fist_object_of_the_class):
                ss = str(s)
                sss = str(s).zfill(2)
                img_number = str(random.randrange(1, 300)).zfill(3)
                file_name = 'C_' + sss + '_' + str(o).zfill(2) + '_' + img_number + '.png'
                src = image_dir + '/s' + ss + '/o' + str(o) + '/' + file_name
                dest = os.path.join(html_dir_path, file_name)
                if os.path.exists(src):
                    shutil.copyfile(src, dest)
                    f.write('<img src="' + file_name + '">')
                else:
                    # fd.write('{}, {}, {}'.format(o, mod, q))
                    pass

            if mod == 0:
                f.write('</td>')
        f.write('</tr>')

    f.write('<table>\n')


def add_task_run_dic_key(task_info, task_id=None, run=None, key=None):
    if task_id in task_info:
        if run in task_info[task_id]:
            if key in task_info[task_id][run]:
                pass
            else:
                task_info[task_id][run].update({key: []})
        else:
            task_info[task_id].update({run: {key: []}})
    else:
        task_info.update({task_id: {run: {key: []}}})


def join_values_using_key(task_info, key=None):
    for task_id in sorted(task_info.keys()):
        for run in sorted(task_info[task_id].keys()):
            old_key_idx_items = ''
            for key_idx in task_info[task_id][run][key]:
                key_idx_items = ''.join([str(x)+',' for x in key_idx])
                if old_key_idx_items != key_idx_items:
                    old_key_idx_items += key_idx_items
            task_info[task_id][run][key] = old_key_idx_items


def get_range(s):
    ll = s.split(',')
    ll = ll[0: len(ll): len(ll) - 2]
    return ll[0] + ' - ' + ll[1]

def original_NI(df, scenario=None, print_objects=True):
    scenario_dir = os.path.join(core50_dir, 'batches_filelists', scenario)
    task_info = {}
    for run in os.scandir(path=scenario_dir):
        if run.is_dir():
            # print(run.name)
            r = run.name.replace('run', '')
            for f in os.scandir(path=os.path.join(scenario_dir, run.name)):
                if f.is_file():
                    # train_batch_03_filelist.txt | test_filelist.txt
                    if f.name == 'test_filelist.txt':
                        t_id = '-1'
                    else:
                        t_id = f.name.replace('train_batch_', '').replace('_filelist.txt', '')
                    # print('==', f.name)
                    # s11/o1/C_11_01_000.png 0
                    sessions = []
                    command = subprocess.Popen(
                        "awk -F '/' '{print $1}' " + os.path.join(scenario_dir, run.name,
                                                                  f.name) + " | sed 's/s//g' | sort | uniq",
                        shell=True, stdout=subprocess.PIPE)
                    for line in command.stdout.readlines():
                        sessions.append(int(line.decode("utf-8").replace('\n', '')))

                    objects = []
                    command = subprocess.Popen(
                        "awk -F '/' '{print $2}' " + os.path.join(scenario_dir, run.name,
                                                                  f.name) + " | sed 's/o//g' | sort | uniq",
                        shell=True, stdout=subprocess.PIPE)
                    for line in command.stdout.readlines():
                        objects.append(int(line.decode("utf-8").replace('\n', '')))

                    classes = []
                    command = subprocess.Popen(
                        "awk -F ' ' '{print $2}' " + os.path.join(scenario_dir, run.name,
                                                                  f.name) + " | sed 's/o//g' | sort | uniq",
                        shell=True, stdout=subprocess.PIPE)
                    for line in command.stdout.readlines():
                        classes.append(int(line.decode("utf-8").replace('\n', '')))

                    sessions.sort()
                    objects.sort()
                    classes.sort()

                    add_task_run_dic_key(task_info, task_id=t_id, run=r, key='sessions')
                    add_task_run_dic_key(task_info, task_id=t_id, run=r, key='objects')
                    add_task_run_dic_key(task_info, task_id=t_id, run=r, key='classes')

                    task_info[t_id][r]['sessions'].append(sessions)
                    task_info[t_id][r]['objects'].append(objects)
                    task_info[t_id][r]['classes'].append(classes)
                    # print('S: ', sessions)
                    # print('O: ', objects)
    # print(task_info['-1'])

    join_values_using_key(task_info, key='sessions')
    join_values_using_key(task_info, key='objects')
    join_values_using_key(task_info, key='classes')

    fd.write('<h3>{}</h3>'.format(scenario))
    fd.write('<table border="1">\n')
    fd.write('<tr><td>task</td><td>run</td><td>session</td>{}<td>classes</td></tr>\n'.
             format('<td>objects</td>' if print_objects else ''))
    for t in sorted(task_info.keys()):
        for r in sorted(task_info[t].keys()):
            fd.write('<tr>')
            fd.write('<td>{}</td>'.format(t))
            fd.write('<td>{}</td>'.format(r))
            fd.write('<td>{}</td>'.format(task_info[t][r]['sessions']))
            if print_objects:
                fd.write('<td>{}</td>'.format(get_range(task_info[t][r]['objects'])))
            fd.write('<td>{}</td>'.format(get_range(task_info[t][r]['classes'])))
            fd.write('</tr>\n')
    fd.write('<table>\n')



html_dir_path = os.path.join(base_dir, 'Core50')
if os.path.isdir(html_dir_path):
    shutil.rmtree(html_dir_path)
os.mkdir(html_dir_path)

fd = open(os.path.join(html_dir_path, 'corRe50.html'), 'w')

fd.write(head)

print_table_NI_DI_cat_task_id_by_session(fd, show_only_fist_object_of_the_class=True)
print_table_NI_DI_cat_task_id_by_session(fd, show_only_fist_object_of_the_class=False)
original_NI(fd, scenario='NI_cum')
original_NI(fd, scenario='NI_inc')
original_NI(fd, scenario='NI_inc_cat')

fd.write(tail)
fd.close()
