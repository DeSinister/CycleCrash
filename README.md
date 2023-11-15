# CycleCrash


## Overview
CycleCrash is a newly introduced dataset that prioritizes the safety of cyclists. In response to a lack of datasets focusing on cycling in contrast to the abundance for self-driving cars, CycleCrash consists of 3000 annotated video clips. Notably, 1000 of these clips depict cyclists in urban environments facing potentially dangerous situations, such as collisions or near-misses. The dataset's annotations include details like pre-event and post-event labels, cyclist scene descriptors, and camera/motion labels. This dataset provides valuable resources for studying cyclist behavior, implementing collision prevention measures, and advancing our understanding of safety in cycling scenarios.

## Files
- **Final.csv**: CSV file containing the CycleCrash dataset.
- **splitter.py**: Python script to split the dataset in training and validation split.
- **pre_processing.py**: Python script to Pre-processs videos for background cropping and uniform temporal and spatial dimensions.
- **collision_anticipation.py**: Python script to train and evaluate video processing models for Collision Ancticipation task.
- **time_to_collision.py**: Python script to train and evaluate video processing models for Collision Ancticipation task.
- **severity_prediction.py**: Python script to train and evaluate video processing models for Time-to-Collsion prediction task.
- **right_of_way_prediction.py**: Python script to train and evaluate video processing models for Right of Way prediction task.
- **vulnerability_estimation.py**: Python script to train and evaluate video processing models for Vulnerability Estimation task.

## Dataset File Structure
#### Unsafe Folder
The `unsafe` folder contains the first 1000 videos from `Final.csv` that exhibit unsafe cycling interactions. These videos are labeled as potentially hazardous or containing unsafe cycling behavior.
#### Safe Folder
The `safe` folder includes the remaining 2000 videos from `Final.csv` that are labeled as safe cycling interactions. These videos are considered to depict cycling situations that are safe and do not involve any potential hazards.

dataset/

|-- unsafe/

| |-- video1.mp4

| |-- video2.mp4

| |-- ...

| |-- video1000.mp4

|

|-- safe/

| |-- video1001.mp4

| |-- video1002.mp4

| |-- ...

| |-- video3000.mp4


Please refer to the `Final.csv` file for detailed annotations and additional information about each video in the dataset.
