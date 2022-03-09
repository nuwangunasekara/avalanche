import os
import random
import shutil
import subprocess
import numpy as np

from avalanche.benchmarks.classic.stream51 import CLStream51

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

base_dir = '/Users/ng98/Desktop/avalanche_nuwan_fork/exp_scripts/data/'

_mu = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]

head = '<!DOCTYPE html><html><head><title>CLStream51</title></head><body>'
tail ='</body></html>'

html_dir_path = os.path.join(base_dir, 'CLStream51')
if os.path.isdir(html_dir_path):
    shutil.rmtree(html_dir_path)
os.mkdir(html_dir_path)

fd = open(os.path.join(html_dir_path, 'CLStream.html'), 'w')
fd.write(head)

scenario = CLStream51(scenario='instance', seed=10, eval_num=None,
                      # dataset_root='/Scratch/ng98/CL/avalanche_data/',
                      no_novelity_detection=True
                      )

train_imgs_count = 0

# fd.write('<table border="1">\n')
fd.write('<table>\n')
for t, stream in enumerate(scenario.train_stream):
    # print(t, stream)
    dataset, task_label = stream.dataset, stream.task_label
    train_imgs_count += len(dataset)

    fd.write('<tr>')
    print(t, end=' ')
    dl = DataLoader(dataset, batch_size=1)
    previous_y = None
    y_count = 0

    for j, mb in enumerate(dl):
        x, y, *_ = mb

        if y.item() == previous_y:
            if y_count < 2:
                print(y.item(), end=' ')
                # show a few un-normalized images from data stream
                # this code is for debugging purposes
                x_np = x[0, :, :, :].numpy().transpose(1, 2, 0)
                x_np = x_np * _std + _mu
                plt.imshow((x_np * 255).astype(np.uint8))
                # plt.show()
                file_name = 'IMG_' + str(t) +'_' + str(j) +'.png'
                plt.savefig(html_dir_path + '/' + file_name)
                fd.write('<td>{}</td><td><img src="{}" style="object-fit:contain; width:100px; height:150px;"></td>'.format(y.item(),file_name))

                # print(x.shape)
                # print(y.shape)
            y_count += 1
        else:
            y_count = 0

        previous_y = y.item()
    print(" ")
    fd.write('</tr>\n')
fd.write('\n<table>\n')
fd.write(tail)
fd.close()
# make sure all of the training data was loaded properly
assert train_imgs_count == 150736


