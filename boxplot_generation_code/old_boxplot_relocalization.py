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

matplotlib.use('agg')
metrics = ['Performance metric']

colorlist = ['#34405c', '#005a87', '#0077aa', 'lightcoral', '#0096bd', '#00b4bb', '#00d0a3', '#00e976', '#44ff29']
net = ['VGG', 'ResNet50', 'ResNet50 + VLAD', 'SIFT', 'ORB','proposed']

nice_net = ['VGG', 'ResNet50', 'ResNet50 + VLAD', 'SIFT', 'ORB','Proposed']
for i, a in enumerate(net):
    print (a)

for metric in metrics:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    #plt.subplots_adjust(left=0.2, right=0.9, top=1.0, bottom=0.6)
    fig.tight_layout()
    color = plt.get_cmap("Set1")
    top = 1.15
    bottom = -0.1

    vgg_001 = np.load('final_dataset_files/boxplots_npy/VGG_FINAL/ssim_relocalization_anon001.npy')
    resnet_001 = np.load('final_dataset_files/boxplots_npy/ResNet50_FINAL/ssim_relocalization_anon001.npy')
    resnet_vlad_001 = np.load('final_dataset_files/boxplots_npy/ResNet50_VLAD_FINAL/ssim_relocalization_anon001.npy')
    sift_001 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_relocalization_anon001.npy')
    orb_001 = np.load('../final_dataset_files/boxplots_npy/SLAM_ORB/ssim_relocalization_anon001.npy')
    proposed_001 = np.load('../final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_relocalization_anon001.npy')

    data_001 = [vgg_001, resnet_001, resnet_vlad_001, sift_001, orb_001, proposed_001]

    plt.boxplot(data_001, positions=[0], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=colorlist[0], color='k'), medianprops=dict(color='k'),
                flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[0]}
                )

    ax.text(0, top - (top * 0.15), '{:.2f}±{:.2f}'.format(np.round(np.mean(data_001), 2), np.round(np.std(data_001), 2)),
            horizontalalignment='center', rotation=45, size='x-small', weight='semibold',
            color=colorlist[0])



    vgg_002 = np.load('final_dataset_files/boxplots_npy/VGG_FINAL/ssim_relocalization_anon002.npy')
    resnet_002 = np.load('final_dataset_files/boxplots_npy/ResNet50_FINAL/ssim_relocalization_anon002.npy')
    resnet_vlad_002 = np.load('final_dataset_files/boxplots_npy/ResNet50_VLAD_FINAL/ssim_relocalization_anon002.npy')
    sift_002 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_relocalization_anon002.npy')
    orb_002 = np.load('../final_dataset_files/boxplots_npy/SLAM_ORB/ssim_relocalization_anon002.npy')
    proposed_002 = np.load('../final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_relocalization_anon002.npy')

    data_002 = [vgg_002, resnet_002, resnet_vlad_002, sift_002, orb_002, proposed_002]


    plt.boxplot(data_002, positions=[1], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=colorlist[1], color='k'), medianprops=dict(color='k'),
                flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[1]}
                )

    ax.text(0, top - (top * 0.15), '{:.2f}±{:.2f}'.format(np.round(np.mean(data_002), 2), np.round(np.std(data_002), 2)),
            horizontalalignment='center', rotation=45, size='x-small', weight='semibold',
            color=colorlist[1])



    vgg_003 = np.load('final_dataset_files/boxplots_npy/VGG_FINAL/ssim_relocalization_anon003.npy')
    resnet_003 = np.load('final_dataset_files/boxplots_npy/ResNet50_FINAL/ssim_relocalization_anon003.npy')
    resnet_vlad_003 = np.load('final_dataset_files/boxplots_npy/ResNet50_VLAD_FINAL/ssim_relocalization_anon003.npy')
    sift_003 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_relocalization_anon003.npy')
    orb_003 = np.load('../final_dataset_files/boxplots_npy/SLAM_ORB/ssim_relocalization_anon003.npy')
    proposed_003 = np.load('../final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_relocalization_anon003.npy')

    data_003 = [vgg_003, resnet_003, resnet_vlad_003, sift_003, orb_003, proposed_003]

    plt.boxplot(data_003, positions=[2], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=colorlist[2], color='k'), medianprops=dict(color='k'),
                flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[2]}
                )

    ax.text(0, top - (top * 0.15), '{:.2f}±{:.2f}'.format(np.round(np.mean(data_003), 2), np.round(np.std(data_003), 2)),
            horizontalalignment='center', rotation=45, size='x-small', weight='semibold',
            color=colorlist[2])

    bano_005 = np.load('../final_dataset_files/boxplots_npy/Bano_mosaicking/ssim_mosaicking_anon005.npy')
    proposed_005 = np.load('../final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_mosaicking_anon005.npy')
    sift_005 = np.zeros(10)
    orb_005 = np.load('../final_dataset_files/boxplots_npy/SLAM_ORB/ssim_mosaicking_anon005.npy')

    data_005 = [bano_005, sift_005, orb_005, proposed_005]
    plt.boxplot(data_005, positions=[3], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=colorlist[3], color='k'), medianprops=dict(color='k'),
                flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[3]}
                )

    ax.text(0, top - (top * 0.15), '{:.2f}±{:.2f}'.format(np.round(np.mean(data_005), 2), np.round(np.std(data_005), 2)),
            horizontalalignment='center', rotation=45, size='x-small', weight='semibold',
            color=colorlist[3])


    vgg_010 = np.load('final_dataset_files/boxplots_npy/VGG_FINAL/ssim_relocalization_anon010.npy')
    resnet_010 = np.load('final_dataset_files/boxplots_npy/ResNet50_FINAL/ssim_relocalization_anon010.npy')
    resnet_vlad_010 = np.load('final_dataset_files/boxplots_npy/ResNet50_VLAD_FINAL/ssim_relocalization_anon010.npy')
    sift_010 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_relocalization_anon010.npy')
    orb_010 = np.load('../final_dataset_files/boxplots_npy/SLAM_ORB/ssim_relocalization_anon010.npy')
    proposed_010 = np.load('../final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_relocalization_anon010.npy')

    data_010 = [vgg_010, resnet_010, resnet_vlad_010, sift_010, orb_010, proposed_010]

    plt.boxplot(data_010, positions=[4], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=colorlist[4], color='k'), medianprops=dict(color='k'),
                flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[4]}
                )

    ax.text(0, top - (top * 0.15), '{:.2f}±{:.2f}'.format(np.round(np.mean(data_010), 2), np.round(np.std(data_010), 2)),
            horizontalalignment='center', rotation=45, size='x-small', weight='semibold',
            color=colorlist[4])


    vgg_012 = np.load('final_dataset_files/boxplots_npy/VGG_FINAL/ssim_relocalization_anon012.npy')
    resnet_012 = np.load('final_dataset_files/boxplots_npy/ResNet50_FINAL/ssim_relocalization_anon012.npy')
    resnet_vlad_012 = np.load('final_dataset_files/boxplots_npy/ResNet50_VLAD_FINAL/ssim_relocalization_anon012.npy')
    sift_012 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_relocalization_anon012.npy')
    orb_012 = np.load('../final_dataset_files/boxplots_npy/SLAM_ORB/ssim_relocalization_anon012.npy')
    proposed_012 = np.load('../final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_relocalization_anon012.npy')

    data_012 = [vgg_012, resnet_012, resnet_vlad_012, sift_012, orb_012, proposed_012]

    plt.boxplot(data_012, positions=[5], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=colorlist[5], color='k'), medianprops=dict(color='k'),
                flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[5]}
                )

    ax.text(0, top - (top * 0.15), '{:.2f}±{:.2f}'.format(np.round(np.mean(data_012), 2), np.round(np.std(data_012), 2)),
            horizontalalignment='center', rotation=45, size='x-small', weight='semibold',
            color=colorlist[5])


    vgg_016 = np.load('final_dataset_files/boxplots_npy/VGG_FINAL/ssim_relocalization_anon016.npy')
    resnet_016 = np.load('final_dataset_files/boxplots_npy/ResNet50_FINAL/ssim_relocalization_anon016.npy')
    resnet_vlad_016 = np.load('final_dataset_files/boxplots_npy/ResNet50_VLAD_FINAL/ssim_relocalization_anon016.npy')
    sift_016 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_relocalization_anon016.npy')
    orb_016 = np.load('../final_dataset_files/boxplots_npy/SLAM_ORB/ssim_relocalization_anon016.npy')
    proposed_016 = np.load('../final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_relocalization_anon016.npy')

    data_016 = [vgg_016, resnet_016, resnet_vlad_016, sift_016, orb_016, proposed_016]

    plt.boxplot(data_016, positions=[6], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=colorlist[6], color='k'), medianprops=dict(color='k'),
                flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[6]}
                )

    ax.text(0, top - (top * 0.15), '{:.2f}±{:.2f}'.format(np.round(np.mean(data_016), 2), np.round(np.std(data_016), 2)),
            horizontalalignment='center', rotation=45, size='x-small', weight='semibold',
            color=colorlist[6])


    vgg_016_CLIP18 = np.load('final_dataset_files/boxplots_npy/VGG_FINAL/ssim_relocalization_anon016_CLIP18.npy')
    resnet_016_CLIP18 = np.load('final_dataset_files/boxplots_npy/ResNet50_FINAL/ssim_relocalization_anon016_CLIP18.npy')
    resnet_vlad_016_CLIP18 = np.load('final_dataset_files/boxplots_npy/ResNet50_VLAD_FINAL/ssim_relocalization_anon016_CLIP18.npy')
    sift_016_CLIP18 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_relocalization_anon016_CLIP18.npy')
    orb_016_CLIP18 = np.load('../final_dataset_files/boxplots_npy/SLAM_ORB/ssim_relocalization_anon016_CLIP18.npy')
    proposed_016_CLIP18 = np.load(
        '../final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_relocalization_anon016_CLIP18.npy')

    data_016_CLIP18 = [vgg_016_CLIP18, resnet_016_CLIP18, resnet_vlad_016_CLIP18, sift_016_CLIP18, orb_016_CLIP18, proposed_016_CLIP18]

    plt.boxplot(data_016_CLIP18, positions=[7], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=colorlist[7], color='k'), medianprops=dict(color='k'),
                flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[7]}
                )

    ax.text(0, top - (top * 0.15), '{:.2f}±{:.2f}'.format(np.round(np.mean(data_016_CLIP18), 2), np.round(np.std(data_016_CLIP18), 2)),
            horizontalalignment='center', rotation=45, size='x-small', weight='semibold',
            color=colorlist[7])



    vgg_024_CLIP00 = np.load('final_dataset_files/boxplots_npy/VGG_FINAL/ssim_relocalization_anon024_CLIP00.npy')
    resnet_024_CLIP00  = np.load('final_dataset_files/boxplots_npy/ResNet50_FINAL/ssim_relocalization_anon024_CLIP00.npy')
    resnet_vlad_024_CLIP00  = np.load('final_dataset_files/boxplots_npy/ResNet50_VLAD_FINAL/ssim_relocalization_anon024_CLIP00.npy')
    sift_024_CLIP00  = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_relocalization_anon024_CLIP00.npy')
    orb_024_CLIP00  = np.load('../final_dataset_files/boxplots_npy/SLAM_ORB/ssim_relocalization_anon024_CLIP00.npy')
    proposed_024_CLIP00  = np.load(
        '../final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_relocalization_anon024_CLIP00.npy')

    data_024_CLIP00  = [vgg_024_CLIP00 , resnet_024_CLIP00 , resnet_vlad_024_CLIP00 , sift_024_CLIP00 , orb_024_CLIP00 , proposed_024_CLIP00]


    plt.boxplot(data_024_CLIP00, positions=[8], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=colorlist[8], color='k'), medianprops=dict(color='k'),
                flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[8]}
                )

    ax.text(0, top - (top * 0.15), '{:.2f}±{:.2f}'.format(np.round(np.mean(data_024_CLIP00), 2), np.round(np.std(data_024_CLIP00), 2)),
            horizontalalignment='center', rotation=45, size='x-small', weight='semibold',
            color=colorlist[8])



    vgg_024_CLIP01 = np.load('final_dataset_files/boxplots_npy/VGG_FINAL/ssim_relocalization_anon024_CLIP01.npy')
    resnet_024_CLIP01  = np.load('final_dataset_files/boxplots_npy/ResNet50_FINAL/ssim_relocalization_anon024_CLIP01.npy')
    resnet_vlad_024_CLIP01  = np.load('final_dataset_files/boxplots_npy/ResNet50_VLAD_FINAL/ssim_relocalization_anon024_CLIP01.npy')
    sift_024_CLIP01  = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_relocalization_anon024_CLIP01.npy')
    orb_024_CLIP01  = np.load('../final_dataset_files/boxplots_npy/SLAM_ORB/ssim_relocalization_anon024_CLIP01.npy')
    proposed_024_CLIP01  = np.load(
        '../final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_relocalization_anon024_CLIP01.npy')

    data_024_CLIP01  = [vgg_024_CLIP01 , resnet_024_CLIP01 , resnet_vlad_024_CLIP01 , sift_024_CLIP01 , orb_024_CLIP01 , proposed_024_CLIP01]

    plt.boxplot(data_024_CLIP01, positions=[9], widths=0.4, patch_artist=True,
                boxprops=dict(facecolor=colorlist[9], color='k'), medianprops=dict(color='k'),
                flierprops={'marker': 'o', 'markersize': 2, 'markerfacecolor': colorlist[9]}
                )

    ax.text(0, top - (top * 0.15),
            '{:.2f}±{:.2f}'.format(np.round(np.mean(data_024_CLIP01), 2), np.round(np.std(data_024_CLIP01), 2)),
            horizontalalignment='center', rotation=45, size='x-small', weight='semibold',
            color=colorlist[9])


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
    ax.set_xticklabels([str(' '.join(x)) for i, x in enumerate(nice_net)])
    plt.setp(ax.get_xticklabels(), ha="right", rotation=45)
    ax.yaxis.set_ticks_position('none')

    fig.savefig('boxplot_relocalization.png'.format(metric), bbox_inches='tight')
    plt.close(fig)