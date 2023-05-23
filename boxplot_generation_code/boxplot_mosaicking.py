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



# videos_1 = ['anon001', 'anon002','anon003','anon005', 'anon010']
# videos_1_names = ['Video 1', 'Video 2', 'Video 3', 'Video 4', 'Video 5']
# videos_2 = ['anon012','anon016','anon016_CLIP18', 'anon024_CLIP00', 'anon024_CLIP01']
# videos_2_names = ['Video 6', 'Video 7', 'Video 8', 'Video 9', 'Video 10']
# list_figures = [videos_1, videos_2]
# list_figures_names = [videos_1_names, videos_2_names]

videos = ['anon024_CLIP01']#['anon001', 'anon012','anon024_CLIP00']
#videos_names = ['Video 9']#['Video 1', 'Video 6', 'Video 9']
list_figures = [videos]
# #list_figures_names = [videos_names]
list_tries = ['Bano_mosaicking', 'SLAM_SIFT', 'SLAM_ORB', 'SLAM_LoFTR_FINAL']
WIDTH = 0.25
colorlist = [ '#DFF2F8','#91E2FB','#479DB7','#186279','#00b4bb', '#00d0a3', '#00e976', '#44ff29']



for index,figure in enumerate(list_figures):

    fig, ax = plt.subplots(figsize=(2.25, 2), dpi=150)
    # plt.subplots_adjust(left=0.2, right=0.9, top=1.0, bottom=0.6)
    fig.tight_layout()
    color = plt.get_cmap("Set1")
    top = 1.0
    bottom = -0.05
    c = 0

    for i, video_name in enumerate(figure):

        for n,name_try in enumerate(list_tries):
            path_data = os.path.join(os.getcwd(), '../final_dataset_files/boxplots_npy', name_try, 'ssim_mosaicking_' + video_name + '.npy')
            if os.path.exists(path_data):
                data = np.load(path_data)
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
    # ax.set_xticks([i+0.5+0.75*i for i in range(0,5,1)])
    # list_names = list_figures_names[index]
    # ax.set_xticklabels([x for x in list_names])
    # plt.setp(ax.get_xticklabels(), ha="center")
    # ax.yaxis.set_ticks_position('none')
    # ax.set_yticks([1.75])
    # ax.set_yticklabels(['s'])
    plt.ylabel("5-frame SSIM (s)", rotation='vertical')
    plt.xlabel(" ", rotation='vertical')
    ax.set_xticks([])


    fig.savefig('metric_mosaicking_'+str(index)+'.png', bbox_inches='tight')
    plt.close(fig)




