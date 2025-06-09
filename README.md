# Block-wise-GLM-Analysis-of-fMRI-Data
This repository contains a Python script for performing block-wise first-level GLM analysis of fMRI data using the Nilearn library. It extracts trial-by-trial beta maps for later multivariate pattern analysis (MVPA), and is compatible with BIDS-formatted datasets.



## Overview

The goal of this script is to:

Load BIDS-style fMRI data for multiple subjects
Fit a GLM to each run of each subject's data
Split each trial into individual blocks
Compute and save beta maps (effect size images) for each block
This enables fine-grained MVPA using trial-specific beta estimates, compatible with tools like CosmoMVPA or machine learning pipelines. 

## Folder Structure

Assumes a BIDS-style folder layout, e.g.:

project_root/
├── sub-001/
│   └── func/
│       ├── wasub-001_task-music_run-1_bold.nii.gz
│       ├── sub-001_task-music_run-1_events.tsv
│       ├── rp_asub-001_task-music_run-1_bold.txt
│       └── task-music_bold.json
├── sub-002/
└── ...


##Install the required Python packages:

pip install numpy pandas nibabel nilearn


## The output beta maps will be saved to:
/your/root/path/Betas_all_blocks/last_beta_maps_true/sub-XXX/

Each beta map is named like:

sub-001_task-music_run-1_positive_music_block1_beta.nii.gz


###  Design Matrix Plotting Script

This script generates visualizations of the design matrices used in the block-wise GLM analysis for each subject, task, and run. Design matrices are a key component of the GLM and show how different conditions and confounds are modeled over time.

 What It Does
Iterates through all subjects and their design matrix .tsv files (saved during GLM fitting)
Uses nilearn.plotting.plot_design_matrix() to create a visual plot of each matrix
Saves the plots as .png images for easy review and reporting



