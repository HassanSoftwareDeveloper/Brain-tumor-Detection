# Project Documentation: Brain Glioma Segmentation with nnU-Net and MONAI
## 1. Project Overview

This project focuses on brain glioma segmentation using deep learning. The main libraries used are:

# MONAI: For medical imaging data handling, transformations, and dataset preparation.

PyTorch: Deep learning framework.

nnU-Net: Automated deep learning framework for medical image segmentation.

The workflow includes:

Preparing and verifying dataset.

Preprocessing and normalization.

Creating datasets and dataloaders with MONAI.

Calculating mean and standard deviation per channel.

Applying data augmentations.

Splitting train/test datasets.

Using nnU-Net for planning, preprocessing, and training.
# 1. Project Overview

This project focuses on brain glioma segmentation using deep learning. The main libraries used are:

MONAI: For medical imaging data handling, transformations, and dataset preparation.

PyTorch: Deep learning framework.

nnU-Net: Automated deep learning framework for medical image segmentation.

# The workflow includes:

Preparing and verifying dataset.

Preprocessing and normalization.

Creating datasets and dataloaders with MONAI.

Calculating mean and standard deviation per channel.

Applying data augmentations.

Splitting train/test datasets.

Using nnU-Net for planning, preprocessing, and training.






Dataset Structure

nnU-Net expects the following folder structure:

nnUNet_raw_data/
└── Task001_Glioma/
    ├── imagesTr/  # Training images (.nii.gz)
    ├── labelsTr/  # Training labels (.nii.gz)
    └── imagesTs/  # Test images (.nii.gz)


Images must follow nnU-Net naming conventions (e.g., patient001_0000.nii.gz).

Installation
# MONAI & dependencies
pip install monai torch torchvision nibabel tqdm

# nnU-Net
pip install nnunet

Usage
1. Verify Dataset
nnUNet_plan_and_preprocess -t 1 --verify_dataset_integrity

2. Preprocessing & Data Loading (MONAI)

Load images and labels.

Apply transforms: channel first, normalization, augmentation.

Split dataset into train/test.

Example:

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std, channel_wise=True),
    RandRotated(keys=["image", "label"], range_x=0.3, prob=0.5),
    RandFlipd(keys=["image", "label"], prob=0.5),
    RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ToTensord(keys=["image", "label"])
])

3. Compute Dataset Statistics
mean, std = get_mean_std(train_loader, nonzero=True)


Use nonzero=True to ignore background voxels.

4. Split Dataset
from torch.utils.data import random_split

length_train = int(len(dataset) * 0.8)
length_remaining = len(dataset) - length_train
train_subset, val_subset = random_split(dataset, [length_train, length_remaining])

5. Training nnU-Net
nnUNet_train 3d_fullres nnUNetTrainerV2 Task001_Glioma FOLD


Replace FOLD with 0–4 for cross-validation.

6. Inference
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t 1 -m 3d_fullres

Tips

Always verify dataset structure.

Use SafeDataset and collate_skip_none to handle corrupted images.

Normalize using non-zero voxels for better results.

Maintain consistent file/folder naming for nnU-Net.
