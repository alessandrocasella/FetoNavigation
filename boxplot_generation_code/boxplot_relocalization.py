import csv
from itertools import product

import matplotlib.pyplot as plt
import matplotlib
import pandas
from matplotlib import rc
#matplotlib.rcParams['text.usetex'] = True
import numpy as np
import os
import glob
import matplotlib.cm as cm
import pandas


#videos = sorted(os.listdir('final_dataset'))
videos_1 = ['anon001', 'anon002','anon003','anon005']
videos_1_names = ['video1', 'video2', 'video3', 'video4']
videos_2 = ['anon010','anon012','anon016','anon016_CLIP18' ]
videos_2_names = ['video5', 'video6', 'video7', 'video8']
list_figures = [videos_1, videos_2]
list_figures_names = [videos_1_names, videos_2_names]
list_tries = ['VGG_FINAL_ALL_VIDEOS', 'ResNet_FINAL_ALL_VIDEOS', 'ResNetVLAD_FINAL_ALL_VIDEOS', 'SLAM_SIFT', 'SLAM_ORB','SLAM_LoFTR_FINAL']
WIDTH = 0.25
colorlist = ['#F9EFAE', '#CAEDF7', '#FADCDA', '#DAF2D9','#0096bd', '#00b4bb', '#00d0a3', '#00e976', '#44ff29']



for index,figure in enumerate(list_figures):

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    # plt.subplots_adjust(left=0.2, right=0.9, top=1.0, bottom=0.6)
    fig.tight_layout()
    color = plt.get_cmap("Set1")
    top = 1.0
    bottom = -0.1
    c = 0

    for i, video_name in enumerate(figure):

        for n,name_try in enumerate(list_tries):
            path_data = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try, 'ssim_relocalization_' + video_name + '.npy')
            if os.path.exists(path_data):
                data = np.load(path_data)
                print(name_try)
                print(data)
            else:
                data = np.zeros(10)

            plt.boxplot(data, positions=[i+n*WIDTH+c], widths=0.2, patch_artist=True,
                            boxprops=dict(facecolor=colorlist[n], color='k'), medianprops=dict(color='k'),
                        flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[n]}
                        )
        c = c+0.75

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_axisbelow(True)

    ax.set_ylim([bottom, top])

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    #ax.set_xticks([2,6,10,14,18,22,26,30])
    #ax.set_xticks([i+0.5+0.75*i for i in range(0,4,1)])
    #list_names = list_figures_names[index]
    #print(list_names)
    #ax.set_xticklabels([x for x in list_names])
    #plt.setp(ax.get_xticklabels(), ha="center")
    ax.yaxis.set_ticks_position('none')

    fig.savefig('metric_relocalization_'+str(index)+'.png', bbox_inches='tight')
    plt.close(fig)