# Mosaicking and Occlusion Recovery for Fetoscopy

> **⚠️ Warning**: Code should be refactored in a next release

## Overview

This repository contains the implementation of mosaicking and occlusion recovery algorithms for fetoscopy, developed as part of a thesis project. The project focuses on panoramic image reconstruction and SLAM techniques for medical imaging applications.

## 📁 Directory Structure

### Core Algorithm Files

- `reconstruction_algorithm_RESNET.py` - Main reconstruction algorithm using ResNet features
- `reconstruction_algorithm_RESNET_VLAD.py` - Reconstruction with ResNet and VLAD descriptors
- `reconstruction_algorithm_VGG.py` - Reconstruction algorithm using VGG features
- `SLAM_LoFTR_new_dataset.py` - SLAM implementation with LoFTR on new dataset
- `SLAM_LoFTR_simulation.py` - SLAM simulation using LoFTR
- `SLAM_ORB.py` - SLAM implementation with ORB features
- `SLAM_SIFT.py` - SLAM implementation with SIFT features
- `project_robotics.py` - Mosaicking robotic project script
- `resnetfinal.pth` - Pre-trained ResNet50 weights from FetReg Challenge 2021

### 📊 Analysis and Visualization

#### `boxplot_generation_code/`
Scripts for generating statistical boxplots from experimental data.

#### `statistical_tests/`
Implementation of statistical tests for validating experimental results.

### 🧪 Experimental Code and Prototypes

#### `code_examples_tries/` *(gitignored)*
- `blitting_tries/` - Attempts to implement blitting optimization for 3D scatterplot k-means clustering
- `code_imported/` - Scripts imported from external projects
- `template_matching_tries/` - Template matching algorithm experiments
- `tries-LoFTR_images/` - LoFTR demonstration images

### 📚 External Dependencies

#### `code_Git_imported/`
- `exposure_fusion/` - Mertens exposure fusion implementation (MATLAB)
- `LoFTR/` - [LoFTR repository](https://github.com/zju3dv/LoFTR)
- `SuperPointPretrainedNetwork/` - [SuperPoint implementation](https://github.com/magicleap/SuperPointPretrainedNetwork)
- `VGG/` - VGG network implementations (PyTorch and Keras)
- `VLAD_master/` - VLAD descriptor repository

### 🗄️ Datasets

#### `dataset_MICCAI_2020/`
- `dataset/` - Frame sequences organized in subfolders
- `dataset_videos/` - Reconstructed video sequences

#### `final_dataset/`
Primary dataset used for thesis experiments, with frames organized in subfolders.

### 📈 Experimental Results

#### `dataset_MICCAI_2020_files/`
- `dictionary_file_npy/` - VGG feature descriptors (NumPy format)
- `experiment_files/` - Experimental results (Excel format)
- `output_panorama_images/` - Generated panoramic images by experiment
- `output_panorama_video/` - Panoramic reconstruction videos
- `similarity_matrices/` - Complete similarity matrices
- `similarity_matrices_rt_gb_9/` - Real-time similarity matrices with Gaussian blur (kernel size 9)
- `visual_dictionary/` - VLAD method dictionaries (Pickle format)

#### `final_dataset_files/`
- `Boxplot_images/` - Statistical visualization outputs
- `boxplots_npy/` - Data files for boxplot generation
- `output_panorama_images/` - Final panoramic reconstructions
- `output_panorama_video/` - Final panoramic videos
- `sanity_check_panorama/` - Relocalization assessment images

### 🔧 Utilities and Tools

#### `utilities/`
- `code_Bano/` - Implementation of Bano et al. (2020) methods
- `examples_generation_code/` - Scripts for generating presentation materials
- `mask_cut/` - MICCAI 2021 dataset masks

#### `similarity_matrix/`
Scripts for generating and processing similarity matrices.

### 📜 Legacy Code

#### `reconstrunction_algorithm_old_code/`
Previous versions of reconstruction algorithms.

#### `SLAM_old_code/`
Earlier SLAM implementations.

#### `VGG_old_code/`
Previous VGG-based feature extraction code.

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Additional dependencies as specified in individual script files

### Usage
Each main algorithm file can be run independently. Refer to individual scripts for specific usage instructions and parameter configurations.

## 📖 Research Context

This work focuses on:
- **Mosaicking**: Creating panoramic views from fetoscopic video sequences
- **Occlusion Recovery**: Handling visual obstructions in medical imaging
- **SLAM**: Simultaneous Localization and Mapping for surgical navigation
- **Feature Matching**: Using various descriptors (LoFTR, SIFT, ORB, VGG, ResNet)

## 📚 References

- MICCAI 2020 Dataset
- MICCAI 2021 Dataset  
- FetReg Challenge 2021
- Bano et al. (2020)

## ⚠️ Development Status

This repository represents research code that requires refactoring for production use. The codebase includes experimental implementations and legacy code that should be cleaned up in future releases. This work was comtribute by @ChiaraLena
