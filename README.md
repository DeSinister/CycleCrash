# CycleCrash


## Overview
Recent developments in computer vision have notably featured extensive labelled datasets of urban environments, primarily focused on advancing self-driving cars. In contrast, datasets do not specifically consider cyclist collisions and safety. To address this, we introduce CycleCrash, a novel dataset comprising 3,000 dashcam video clips, totaling 436,347 frames that capture cyclists in diverse real-world scenarios, including both accident and non-accident clips. This dataset targets potentially hazardous conditions for cyclists and is annotated with collision-related, cyclist-related, and scene-related labels. Furthermore, we present VidNeXt, a novel method that adapts a non-stationary transformer structure originally proposed for forecasting multivariate time series data, toward problems of classification and regression of video within our dataset. We’ve also applied and compared 7 additional baseline models and detailed ablation for our method on 9 cyclist collision prediction and classification tasks.

## Files
- **Final.csv**: CSV file containing the CycleCrash dataset.
- **splitter.py**: Python script to split the dataset in training and validation split.
- **pre_processing.py**: Python script to Pre-process videos for background cropping and uniform temporal and spatial dimensions.
- **data_loader.py**: Python script for implementing PyTorch-based Data loader for CycleCrash dataset
- **get_model.py**: Python script to load the baseline models, the proposed VidNeXt and its ablation variants.

## Directions
- Prepare the videos, and save them as used in the {File Name}_{Counter}.mp4 as shown in Final.csv.
- Run the `pre_processing.py` file, to preprocess the videos.
- Run the `splitter.py` file, to provide the necessary split CSV files for the training and testing set.
- Run the `main.py` file, with the required parameters in arguments for training. 


Please refer to the `Final.csv` file for detailed annotations and additional information about each video in the dataset.



This repository is submitted for review in a double-blinded conference. Specific details, including licensing information, have been intentionally omitted to preserve anonymity.
