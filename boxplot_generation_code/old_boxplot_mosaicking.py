## numpy is used for creating fake data
import numpy as np
import matplotlib as mpl

## agg backend is used to create plot as a .png file
mpl.use('agg')

import matplotlib.pyplot as plt



#
# bano_001 = np.load('final_dataset_files/boxplots_npy/Bano_mosaicking/ssim_mosaicking_anon001.npy')
# proposed_001 = np.load('final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_mosaicking_anon001.npy')
# sift_001 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_mosaicking_anon001.npy')
# orb_001 = np.load('final_dataset_files/boxplots_npy/SLAM_ORB/ssim_mosaicking_anon001.npy')
#
# data_001 = [bano_001, sift_001, orb_001, proposed_001]
#
# # Create a figure instance
# fig_001 = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax_001 = fig_001.add_subplot(111)
#
# ax_001.get_xaxis().tick_bottom()
# ax_001.get_yaxis().tick_left()
# ax_001.set_xticklabels(['Bano et al.', 'SIFT', 'ORB', 'Proposed'])
#
# bp_001 = ax_001.boxplot_generation_code(data_001, patch_artist=True)
#
# ## change outline color, fill color and linewidth of the boxes
# for box in bp_001['boxes']:
#     # change outline color
#     box.set( color='#000000', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#97DDFC' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp_001['whiskers']:
#     whisker.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp_001['caps']:
#     cap.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp_001['medians']:
#     median.set(color='#023F69', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp_001['fliers']:
#     flier.set(marker='o', color='#F3953C', alpha=0.7)
#
# # Save the figure
# fig_001.savefig('final_dataset_files/boxplot_images/bp_mosaic_anon001.png', bbox_inches='tight')
#
#
#
# bano_002 = np.load('final_dataset_files/boxplots_npy/Bano_mosaicking/ssim_mosaicking_anon002.npy')
# proposed_002 = np.load('final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_mosaicking_anon002.npy')
# sift_002 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_mosaicking_anon002.npy')
# orb_002 = np.load('final_dataset_files/boxplots_npy/SLAM_ORB/ssim_mosaicking_anon002.npy')
#
# data_002 = [bano_002, sift_002, orb_002, proposed_002]
#
# # Create a figure instance
# fig_002 = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax_002 = fig_002.add_subplot(111)
#
# ax_002.get_xaxis().tick_bottom()
# ax_002.get_yaxis().tick_left()
# ax_002.set_xticklabels(['Bano et al.', 'SIFT', 'ORB', 'Proposed'])
#
# bp_002 = ax_002.boxplot_generation_code(data_002, patch_artist=True)
#
# ## change outline color, fill color and linewidth of the boxes
# for box in bp_002['boxes']:
#     # change outline color
#     box.set( color='#000000', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#97DDFC' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp_002['whiskers']:
#     whisker.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp_002['caps']:
#     cap.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp_002['medians']:
#     median.set(color='#023F69', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp_002['fliers']:
#     flier.set(marker='o', color='#F3953C', alpha=0.7)
#
# # Save the figure
# fig_002.savefig('final_dataset_files/boxplot_images/bp_mosaic_anon002.png', bbox_inches='tight')

#
#
#
#
#
# bano_003 = np.load('final_dataset_files/boxplots_npy/Bano_mosaicking/ssim_mosaicking_anon003.npy')
# proposed_003 = np.load('final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_mosaicking_anon003.npy')
# sift_003 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_mosaicking_anon003.npy')
# orb_003 = np.load('final_dataset_files/boxplots_npy/SLAM_ORB/ssim_mosaicking_anon003.npy')
#
# data_003 = [bano_003, sift_003, orb_003, proposed_003]
#
# # Create a figure instance
# fig_003 = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax_003 = fig_003.add_subplot(111)
#
# ax_003.get_xaxis().tick_bottom()
# ax_003.get_yaxis().tick_left()
# ax_003.set_xticklabels(['Bano et al.', 'SIFT', 'ORB', 'Proposed'])
#
# bp_003 = ax_003.boxplot_generation_code(data_003, patch_artist=True)
#
# ## change outline color, fill color and linewidth of the boxes
# for box in bp_003['boxes']:
#     # change outline color
#     box.set( color='#000000', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#97DDFC' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp_003['whiskers']:
#     whisker.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp_003['caps']:
#     cap.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp_003['medians']:
#     median.set(color='#023F69', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp_003['fliers']:
#     flier.set(marker='o', color='#F3953C', alpha=0.7)
#
# # Save the figure
# fig_003.savefig('final_dataset_files/boxplot_images/bp_mosaic_anon003.png', bbox_inches='tight')



# bano_005 = np.load('final_dataset_files/boxplots_npy/Bano_mosaicking/ssim_mosaicking_anon005.npy')
# proposed_005 = np.load('final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_mosaicking_anon005.npy')
# orb_005 = np.load('final_dataset_files/boxplots_npy/SLAM_ORB/ssim_mosaicking_anon005.npy')
#
# data_005 = [bano_005, orb_005, proposed_005]
#
# # Create a figure instance
# fig_005 = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax_005 = fig_005.add_subplot(111)
#
# ax_005.get_xaxis().tick_bottom()
# ax_005.get_yaxis().tick_left()
# ax_005.set_xticklabels(['Bano et al.', 'ORB', 'Proposed'])
#
# bp_005 = ax_005.boxplot_generation_code(data_005, patch_artist=True)
#
# ## change outline color, fill color and linewidth of the boxes
# for box in bp_005['boxes']:
#     # change outline color
#     box.set( color='#000000', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#97DDFC' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp_005['whiskers']:
#     whisker.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp_005['caps']:
#     cap.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp_005['medians']:
#     median.set(color='#023F69', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp_005['fliers']:
#     flier.set(marker='o', color='#F3953C', alpha=0.7)
#
# # Save the figure
# fig_005.savefig('final_dataset_files/boxplot_images/bp_mosaic_anon005.png', bbox_inches='tight')


# bano_010 = np.load('final_dataset_files/boxplots_npy/Bano_mosaicking/ssim_mosaicking_anon010.npy')
# proposed_010 = np.load('final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_mosaicking_anon010.npy')
# sift_010 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_mosaicking_anon010.npy')
# orb_010 = np.load('final_dataset_files/boxplots_npy/SLAM_ORB/ssim_mosaicking_anon010.npy')
#
# data_010 = [bano_010, sift_010, orb_010, proposed_010]
#
# # Create a figure instance
# fig_010 = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax_010 = fig_010.add_subplot(111)
#
# ax_010.get_xaxis().tick_bottom()
# ax_010.get_yaxis().tick_left()
# ax_010.set_xticklabels(['Bano et al.', 'SIFT', 'ORB', 'Proposed'])
#
# bp_010 = ax_010.boxplot_generation_code(data_010, patch_artist=True)
#
# ## change outline color, fill color and linewidth of the boxes
# for box in bp_010['boxes']:
#     # change outline color
#     box.set( color='#000000', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#97DDFC' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp_010['whiskers']:
#     whisker.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp_010['caps']:
#     cap.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp_010['medians']:
#     median.set(color='#023F69', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp_010['fliers']:
#     flier.set(marker='o', color='#F3953C', alpha=0.7)
#
# # Save the figure
# fig_010.savefig('final_dataset_files/boxplot_images/bp_mosaic_anon010.png', bbox_inches='tight')




# bano_012 = np.load('final_dataset_files/boxplots_npy/Bano_mosaicking/ssim_mosaicking_anon012.npy')
# proposed_012 = np.load('final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_mosaicking_anon012.npy')
# orb_012 = np.load('final_dataset_files/boxplots_npy/SLAM_ORB/ssim_mosaicking_anon012.npy')
#
# data_012 = [bano_012, orb_012, proposed_012]
#
# # Create a figure instance
# fig_012 = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax_012 = fig_012.add_subplot(111)
#
# ax_012.get_xaxis().tick_bottom()
# ax_012.get_yaxis().tick_left()
# ax_012.set_xticklabels(['Bano et al.', 'ORB', 'Proposed'])
#
# bp_012 = ax_012.boxplot_generation_code(data_012, patch_artist=True)
#
# ## change outline color, fill color and linewidth of the boxes
# for box in bp_012['boxes']:
#     # change outline color
#     box.set( color='#000000', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#97DDFC' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp_012['whiskers']:
#     whisker.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp_012['caps']:
#     cap.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp_012['medians']:
#     median.set(color='#023F69', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp_012['fliers']:
#     flier.set(marker='o', color='#F3953C', alpha=0.7)
#
# # Save the figure
# fig_012.savefig('final_dataset_files/boxplot_images/bp_mosaic_anon012.png', bbox_inches='tight')




# proposed_016 = np.load('final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_mosaicking_anon016.npy')
# sift_016 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_mosaicking_anon016.npy')
# orb_016 = np.load('final_dataset_files/boxplots_npy/SLAM_ORB/ssim_mosaicking_anon016.npy')
#
# data_016 = [sift_016, orb_016, proposed_016]
#
# # Create a figure instance
# fig_016 = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax_016 = fig_016.add_subplot(111)
#
# ax_016.get_xaxis().tick_bottom()
# ax_016.get_yaxis().tick_left()
# ax_016.set_xticklabels(['SIFT', 'ORB', 'Proposed'])
#
# bp_016 = ax_016.boxplot_generation_code(data_016, patch_artist=True)
#
# ## change outline color, fill color and linewidth of the boxes
# for box in bp_016['boxes']:
#     # change outline color
#     box.set( color='#000000', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#97DDFC' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp_016['whiskers']:
#     whisker.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp_016['caps']:
#     cap.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp_016['medians']:
#     median.set(color='#023F69', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp_016['fliers']:
#     flier.set(marker='o', color='#F3953C', alpha=0.7)
#
# # Save the figure
# fig_016.savefig('final_dataset_files/boxplot_images/bp_mosaic_anon016.png', bbox_inches='tight')



# proposed_016_CLIP18 = np.load('final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_mosaicking_anon016_CLIP18.npy')
# sift_016_CLIP18 = np.load('final_dataset_files/boxplots_npy/SLAM_SIFT/ssim_mosaicking_anon016_CLIP18.npy')
# orb_016_CLIP18 = np.load('final_dataset_files/boxplots_npy/SLAM_ORB/ssim_mosaicking_anon016_CLIP18.npy')
#
# data_016_CLIP18 = [sift_016_CLIP18, orb_016_CLIP18, proposed_016_CLIP18]
#
# # Create a figure instance
# fig_016_CLIP18 = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax_016_CLIP18 = fig_016_CLIP18.add_subplot(111)
#
# ax_016_CLIP18.get_xaxis().tick_bottom()
# ax_016_CLIP18.get_yaxis().tick_left()
# ax_016_CLIP18.set_xticklabels(['SIFT', 'ORB', 'Proposed'])
#
# bp_016_CLIP18 = ax_016_CLIP18.boxplot_generation_code(data_016_CLIP18, patch_artist=True)
#
# ## change outline color, fill color and linewidth of the boxes
# for box in bp_016_CLIP18['boxes']:
#     # change outline color
#     box.set( color='#000000', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#97DDFC' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp_016_CLIP18['whiskers']:
#     whisker.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp_016_CLIP18['caps']:
#     cap.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp_016_CLIP18['medians']:
#     median.set(color='#023F69', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp_016_CLIP18['fliers']:
#     flier.set(marker='o', color='#F3953C', alpha=0.7)
#
# # Save the figure
# fig_016_CLIP18.savefig('final_dataset_files/boxplot_images/bp_mosaic_anon016_CLIP18.png', bbox_inches='tight')
#


# proposed_024_CLIP00 = np.load('final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_mosaicking_anon024_CLIP00.npy')
# orb_024_CLIP00 = np.load('final_dataset_files/boxplots_npy/SLAM_ORB/ssim_mosaicking_anon024_CLIP00.npy')
#
# data_024_CLIP00 = [orb_024_CLIP00, proposed_024_CLIP00]
#
# # Create a figure instance
# fig_024_CLIP00 = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax_024_CLIP00 = fig_024_CLIP00.add_subplot(111)
#
# ax_024_CLIP00.get_xaxis().tick_bottom()
# ax_024_CLIP00.get_yaxis().tick_left()
# ax_024_CLIP00.set_xticklabels([ 'ORB', 'Proposed'])
#
# bp_024_CLIP00 = ax_024_CLIP00.boxplot_generation_code(data_024_CLIP00, patch_artist=True)
#
# ## change outline color, fill color and linewidth of the boxes
# for box in bp_024_CLIP00['boxes']:
#     # change outline color
#     box.set( color='#000000', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#97DDFC' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp_024_CLIP00['whiskers']:
#     whisker.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp_024_CLIP00['caps']:
#     cap.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp_024_CLIP00['medians']:
#     median.set(color='#023F69', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp_024_CLIP00['fliers']:
#     flier.set(marker='o', color='#F3953C', alpha=0.7)
#
# # Save the figure
# fig_024_CLIP00.savefig('final_dataset_files/boxplot_images/bp_mosaic_anon024_CLIP00.png', bbox_inches='tight')



# proposed_024_CLIP01 = np.load('final_dataset_files/boxplots_npy/SLAM_LoFTR_FINAL/ssim_mosaicking_anon024_CLIP01.npy')
# orb_024_CLIP01 = np.load('final_dataset_files/boxplots_npy/SLAM_ORB/ssim_mosaicking_anon024_CLIP01.npy')
#
# data_024_CLIP01 = [orb_024_CLIP01, proposed_024_CLIP01]
#
# # Create a figure instance
# fig_024_CLIP01 = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax_024_CLIP01 = fig_024_CLIP01.add_subplot(111)
#
# ax_024_CLIP01.get_xaxis().tick_bottom()
# ax_024_CLIP01.get_yaxis().tick_left()
# ax_024_CLIP01.set_xticklabels(['ORB', 'Proposed'])
#
# bp_024_CLIP01 = ax_024_CLIP01.boxplot_generation_code(data_024_CLIP01, patch_artist=True)
#
# ## change outline color, fill color and linewidth of the boxes
# for box in bp_024_CLIP01['boxes']:
#     # change outline color
#     box.set( color='#000000', linewidth=2)
#     # change fill color
#     box.set( facecolor = '#97DDFC' )
#
# ## change color and linewidth of the whiskers
# for whisker in bp_024_CLIP01['whiskers']:
#     whisker.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the caps
# for cap in bp_024_CLIP01['caps']:
#     cap.set(color='#000000', linewidth=2)
#
# ## change color and linewidth of the medians
# for median in bp_024_CLIP01['medians']:
#     median.set(color='#023F69', linewidth=2)
#
# ## change the style of fliers and their fill
# for flier in bp_024_CLIP01['fliers']:
#     flier.set(marker='o', color='#F3953C', alpha=0.7)
#
# # Save the figure
# fig_024_CLIP01.savefig('final_dataset_files/boxplot_images/bp_mosaic_anon024_CLIP01.png', bbox_inches='tight')






