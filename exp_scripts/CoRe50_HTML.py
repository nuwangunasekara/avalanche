import os
import random
import shutil

base_dir = '/Users/ng98/Desktop/avalanche_test/'
image_dir = '/Users/ng98/.avalanche/data/core50/core50_32x32'

head = '<!DOCTYPE html><html><head><title>CoRe50</title></head><body>'
tail ='</body></html>'


def print_table(f, show_only_fist_object_of_the_class=False):
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


html_dir_path = os.path.join(base_dir, 'Core50')
if os.path.isdir(html_dir_path):
    shutil.rmtree(html_dir_path)
os.mkdir(html_dir_path)

fd = open(os.path.join(html_dir_path, 'corRe50.html'), 'w')

fd.write(head)
fd.write('<h3>Only first object type from each category/class</h3>')
print_table(fd, show_only_fist_object_of_the_class=True)
fd.write('<h3>All object types for each category/class</h3>')
print_table(fd, show_only_fist_object_of_the_class=False)

fd.write(tail)
fd.close()
