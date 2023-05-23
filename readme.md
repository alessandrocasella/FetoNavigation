# Mosaicking and Occlusion Recovery for Fetoscopy


## DIRECTORY GUIDE:

'boxplot_generation_code' contains scripts for the generation of boxplots


'code_examples_tries' (GITIGNORE) contains scripts used as examples or first tries of other codes
- 'blitting_tries' contains all tries to implement blitting on scatterplot 3D to speed up the kmeans clustering. at the moment the code is not functioning
- 'code_imported' contains scripts imported from other projects (not via GitHub)
- 'template_matching_tries' contains tries for template matching
- 'tries-LoFTR_images' contains pictures of LoFTR demo


'code_Git_imported' contains directories imported from other GitHub repositories
-'exposure_fusion' contains code from Mertens implementation in MATLAB
-'LoFTR' contains code from LoFTR repository
-'SuperPointPretrainedNetwork' contains scripts and weights to run Superpoint demo
-'VGG' contains code for VGG network in pytorch and keras
-'VLAD_master' contains the repository of VLAD


'dataset_MICCAI_2020' contains the dataset from MICCAI 2020
- 'dataset' contains all the frames divided in subfolders
- 'dataset_videos' contains the reconstructed videos


'dataset_MICCAI_2020_files' contains the files obtained from dataset MICCAI 2020
- 'dictionary_file_npy' contains the numpy files that contain the features extracted with VGG from the new dataset and used as descriptors of the images of the new dataset (MICCAI 2020)
- 'experiment_files' contains the excel files of the experiments performed using MICCAI 2020 dataset
- 'output_panorama_images' contains the final panoramic images divided in subfolders depending on the experiment
- 'output_panorama_video' contains the final video of panorama reconstruction divided in subfolders depending on the experiment
- 'similarity_matrices' contains the entire similarity matrices
- 'similarity_matrices_rt_gb_9' contains the similarity matrices real time with gaussian blur with kernel 9
- 'visual_dictionary' contains the dictionaries built with VLAD method divided in subfolders (all files are .pickle)


'final_dataset' contains the dataset used for the thesis: the frames are divided in subfolders


'final_dataset_files' contains the files obtained from final dataset
- 'Boxplot_images' contains boxplot images obtained from data
- 'boxplots_npy' contains all the .npy files used for boxplot generation
- 'output_panorama_images' contains the final panoramic images divided in subfolders depending on the experiment
- 'output_panorama_video' contains the final video of panorama reconstruction divided in subfolders depending on the experiment
- 'sanity_check_panorama' contains all the test images used to assess the relocalization task


'reconstrunction_algorithm_old_code' contains the old code used to perform the thesis tasks


'similarity_matrix' contains all the scripts for similarity matrices generation


'SLAM_old_code' contains the old SLAM code used to perform the thesis tasks


'statistical_tests' contains all the scripts containing the statistical tests implemented


'utilities' contains scripts that were run just once at the beginning of the project to reorder the dataset, cut the images, reconstruct dataset videos.
-'code_Bano' contains scripts to reproduce results from Bano work (2020)
-'examples_generation_code' contains the scripts used to generate images or videos used as examples or in the presentations
-'mask_cut' contains all the masks from MICCAI 2021 dataset


'VGG_old_code' contains the old VGG code used to perform the thesis tasks


## FILES OUTSIDE THE DIRECTORY GUIDE
-'project_robotics.py' is the script from the mosaicking robotic project
-'resnetfinal.pth' contains the weights from the ResNet50 from FetReg Challenge 2021

All the other files contain the code of the final thesis:
-'reconstruction_algorithm_RESNET.py'
-'reconstruction_algorithm_RESNET_VLAD.py'
-'reconstruction_algorithm_VGG.py'
-'SLAM_LoFTR_new_dataset.py'
-'SLAM_LoFTR_simulation.py'
-'SLAM_ORB.py'
-'SLAM_SIFT.py'