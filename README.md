</h1>CycleCrash: A Dataset of Bicycle Collision Videos for Collision
Prediction and Analysis
</h1>

<h3 align="center">
This paper has been accepted to WACV 2025
</h3>

<h3 align="center">
<a href="https://www.pritamsarkar.com">Nishq Poorav Desai</a>
&nbsp; <a href="">Ali Etemad</a>
&nbsp; <a href="">Michael Greenspan</a>
</h3>

<h3 align="center"> 
<a href="#">[Paper]</a>   <!-- change with aaai link -->
<a href="#"> [Appendix]</a> 
<a href="#"> [ArXiv]</a> 
<!-- <a href="https://github.com/pritamqu/AVCAffe"> [Code]</a>   -->
<a href="[https://github.com/DeSinister/CycleCrash/]"> [Website]</a>
</h3>


## Overview
Self-driving research often underrepresents cyclist collisions and safety. To address this, we present CycleCrash, a novel dataset consisting of 3,000 dashcam videos with 436,347 frames that capture cyclists in a range of critical situations, from collisions to safe interactions. This dataset enables 9 different cyclist collision prediction and classification tasks focusing on potentially hazardous conditions for cyclists and is annotated with collision-related, cyclist-related, and scene-related labels. Next, we present
VidNeXt, a novel method that uses a non-stationary transformer on the defined tasks within our dataset. To demonstrate the effectiveness of our method and create additional baselines on CycleCrash, we apply and compare 7 models along with a detailed ablation. 

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


Please refer to the `dataset.csv` file for detailed annotations and additional information about each video in the dataset.


### Citation
If you find this repository useful, please consider giving a star :star: and citing the paper:
```
@inproceedings{desai2025cyclecrash,
  title={CycleCrash: A Dataset of Bicycle Collision Videos for Collision Prediction and Analysis},
  author={Desai, Nishq Poorav and Etemad, Ali and Greenspan, Michael},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2025}
}
```
