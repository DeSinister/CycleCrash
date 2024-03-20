
import cv2
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import transforms as video_transforms
import time
import plotly.graph_objects as go
import os
from tqdm import tqdm
import random
from utils import parse_timestamp, subtract_timestamps, uniform_frame_sampling, pad_tensor



class FetchData():
    def __init__(self, csv_path, video_dir, task_name, set_name = 'train', length = None, segment_duration=1, segment_interval=0.5, target_size=(224, 224), future_sight = 1, after_acc = True):
        self.df = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.video_dir2 = None
        self.task_name = task_name
        self.set_name = set_name
        self.segment_duration = segment_duration
        self.segment_interval = segment_interval
        self.target_size = target_size
        self.frames_list = []
        self.labels = []
        self.future_sight = future_sight
        self.after_acc = after_acc
        self.task_type = None
        self.num_classes = None
        if not length:
            if self.task in ['Time-to-collision', 'Collision Anticipation', 'Right of Way', 'Severity Score', 'Fault', 'Object Involved in Accident Direction']:
                self.length_videos = 3000
            else:
                self.length_videos = 1000
            if self.set_name == 'train':
                self.length_videos = int(0.7*self.length_videos)
            elif self.set_name == 'test':
                self.length_videos = int(0.3*self.length_videos)
        self.length_videos = length
        if self.task_name == 'Time-to-collision':
            self.task_type = "reg"
            self.prepare_data_ttc()
        elif self.task_name == 'Collision Anticipation':
            self.task_type = "binclass"
            self.num_classes = 1
            self.prepare_data_ca()
        elif self.task_name == 'Right of Way':
            self.task_type = "binclass"
            self.num_classes = 1
            self.df = self.df[self.df['Right of Way'] != -1]
            self.df = self.df.reset_index(drop=True)
            self.prepare_data(task_type='binclass', mappings=None)
        elif self.task_name == 'Vulnerability Score':
            self.task_type = "reg"
            self.prepare_data(task_type='reg', mappings=None)
        elif self.task_name == 'Severity Score':
            self.task_type = "multiclass"
            self.num_classes = 5
            self.prepare_data(task_type='multiclass', mappings={'0' : 0, '1' : 1, '2': 2, '3': 3, '4': 4})
        elif self.task_name == 'Fault':
            self.task_type = "binclass"
            self.num_classes = 1
            self.df = self.df[self.df['Fault'].isin([0, 1])]
            self.df = self.df.reset_index(drop=True)
            self.prepare_data(task_type='binclass', mappings=None)
        elif self.task_name == 'Age':
            self.num_classes = 3
            self.task_type = "multiclass"
            self.df = self.df[self.df['Age'] != -1]
            self.df = self.df.reset_index(drop=True)
            self.prepare_data(task_type='multiclass', mappings={'1' : 0, '2' : 1, '3': 2})
        elif self.task_name == 'Cyclist Direction':
            self.num_classes = 5
            self.task_type = "multiclass"
            self.prepare_data(task_type='multiclass', mappings={'None': 0, 'Forwards': 1, 'Backwards': 2, 'Left': 3, 'Right': 4, 'Forwards (Reverse)': 1, 'Right (Reverse)': 4})
        elif self.task_name == 'Object Involved in Accident Direction':
            self.num_classes = 5
            self.task_type = "multiclass"
            self.df = self.df[self.df['Object Involved'].isin(['Car', 'Cyclist', 'Pedestrian', 'Motorcycle', 'Truck', 'Bus', 'Train', 'Scooter', 'Tow Truck', 'Sheep', 'Tractor', 'Skater', 'Deer', 'Cart', 'Cycle', 'Kangaroo', 'Moped-rider', 'Bus-Door', 'Dog', 'Car-Door'])]
            self.df = self.df.reset_index(drop=True)
            self.prepare_data(task_type='multiclass', mappings={'None': 0, 'Forwards': 1, 'Backwards': 2, 'Left': 3, 'Right': 4, 'Forwards (Reverse)': 1, 'Right (Reverse)': 4})
        else:
            print("INVALID task")
            return 
        print("Number of Video Sequences: ", len(self.labels))
        

    def __len__(self):
        return len(self.frames_list)
    
    def __getitem__(self, id):
        if torch.is_tensor(id):
            id = id.tolist()
        frames = self.frames_list[id]
        label = self.labels[id]
        return frames, label

    def check_overlap(self, t1, t2):
        return 0 if t1<t2 else 1

    def create_time_segments_ttc(self, vid_path, total_time, accident_start_time, duration_threshold=6):
        time_segments = []
        vid_segments = []
        current_time = 0.0
        if accident_start_time > duration_threshold:
            current_time = accident_start_time - duration_threshold + self.segment_duration
        while current_time + self.segment_duration <= total_time:
            if accident_start_time - current_time - self.segment_duration>= 0:
                time_segments.append(accident_start_time - current_time )    
                vid_segments.append([vid_path, current_time, current_time + self.segment_duration, None, None, None])
            current_time += self.segment_interval
        if not time_segments:
            return None
        time_segments = torch.stack([torch.tensor(x, dtype=torch.float32) for x in time_segments])
        return time_segments, vid_segments
    
    def create_time_segments_ca(self, vid_path, total_time, accident_start_time, duration_threshold = 6):
        time_segments = []
        vid_segments = []
        current_time = 0.0
        if accident_start_time > duration_threshold:
            current_time = accident_start_time - duration_threshold + self.segment_duration
        while current_time + self.segment_duration <= total_time:
                time_segments.append(self.check_overlap(current_time + self.future_sight, accident_start_time))    
                vid_segments.append([vid_path, current_time, current_time + self.segment_duration])
                current_time += self.segment_interval
        if not time_segments:
            return None
        time_segments = torch.stack([torch.tensor(x, dtype=torch.float32) for x in time_segments])
        return time_segments, vid_segments
    
    def create_vid_segments(self, vid_path, total_time):
        vid_segments = []
        current_time = 0.0
        while current_time + self.segment_duration <= total_time:
            vid_segments.append([vid_path, current_time, current_time + self.segment_duration])
            current_time += self.segment_interval
        return vid_segments
    
    def prepare_data_ttc(self):
        for i in tqdm(range(self.length_videos)):
            if self.df['Duration'][i]>=self.segment_duration:
                accident_time = subtract_timestamps(self.df['Time of Collision'][i], self.df['Start Time'][i])
                temp = self.create_time_segments_ttc(f"{self.video_dir}/{self.df['File Name'][i]}_{int(self.df['Counter'][i])}.mp4",self.df['Duration'][i] , accident_time)
                if temp != None:
                    self.frames_list.append(temp[1])
                    self.labels.append(temp[0])

        self.labels = torch.cat(self.labels)
        self.frames_list = [item for sublist in self.frames_list for item in sublist]

        self.frames_list = [self.frames_list[i]+ [self.labels[i]] for i in range(len(self.labels))]
        print("NUMBER OF SEQUENCES W/O UPSAMPLING: ", len(self.labels))
        if self.set_name == 'train':
            bin_edges = np.arange(min(self.labels), max(self.labels), 0.5)
            video_groups = {i: [] for i in range(len(bin_edges))}

            for i, label in enumerate(self.frames_list):
                bin_index = np.digitize([label[-1]], bin_edges)[0] - 1
                video_groups[bin_index].append(label)
            print("BEFORE UPSAMPLING: ", [len(video_groups[x]) for x in video_groups.keys()])
            max_occurence = max([len(video_groups[x]) for x in video_groups.keys()])
            for key in video_groups.keys():
                if len(video_groups[key]) < max_occurence:
                    while len(video_groups[key]) < max_occurence:
                        # Duplicate each sublist and append 3 random values to each sublist
                        duplicated_sublist = [sublist[:-4] + [random.uniform(0.5, 1.5), random.uniform(0.5, 1.5), random.uniform(0.8, 1.2), random.uniform(0, 0.2)] + [sublist[-1]] for sublist in video_groups[key]]
                        # Append the duplicated sublist to vid
                        video_groups[key]+= duplicated_sublist
                    video_groups[key] = video_groups[key][:max_occurence]

            video_groups = [value for values in video_groups.values() for value in values]
            random.shuffle(video_groups)
            self.labels = [inner_list[-1] for inner_list in video_groups]
            self.frames_list =  [inner_list[:-1] for inner_list in video_groups]
            self.labels = torch.tensor(self.labels, dtype=torch.float32)

    
    def prepare_data_ca(self):
        for i in tqdm(range(self.length_videos)):
            if self.df['Duration'][i]>=self.segment_duration:
                accident_time = subtract_timestamps(self.df['Time of Collision'][i], self.df['Start Time'][i])
                temp = self.create_time_segments_ca(f"{self.video_dir}/{self.df['File Name'][i]}_{int(self.df['Counter'][i])}.mp4",self.df['Duration'][i] , accident_time)
                if temp != None:
                    self.frames_list.append(temp[1])
                    self.labels.append(temp[0])
        self.labels = torch.cat(self.labels)
        self.frames_list = [item for sublist in self.frames_list for item in sublist]

    def prepare_data(self, task_type, mappings):
        if task_type == 'multiclass':
            target_dtype = torch.long
        else:
            target_dtype = torch.float32
        for i in range(self.length_videos):
            if self.df['Duration'][i]>=self.segment_duration: 
                framings = self.create_vid_segments(f"{self.video_dir}/{self.df['File Name'][i]}_{int(self.df['Counter'][i])}.mp4", self.df['Duration'][i])
                self.frames_list.append(framings)
                if mappings != None:
                    self.labels.append(torch.stack([torch.tensor(mappings[self.df[self.task_name][i]], dtype=target_dtype) for x in range(len(framings))]))
                else:
                    self.labels.append(torch.stack([torch.tensor(float(self.df[self.task_name][i]), dtype=target_dtype) for x in range(len(framings))]))
        self.labels = torch.cat(self.labels)
        self.frames_list = [item for sublist in self.frames_list for item in sublist]






def load_and_augment_video(x, num_frames=30, target_frames = 30):
    video_path = x[0]
    start = x[1]
    end = x[2]
    segment_frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    start_frame = int(start * frame_rate)
    end_frame = int(end * frame_rate)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frame = transforms.ToTensor()(frame)
        if len(x)>3 and x[3]:
            frame = transforms.ColorJitter(brightness=x[3], contrast=x[4], saturation = x[5], hue=x[-1])(frame)
        segment_frames.append(frame)   
    if num_frames != target_frames:
        video = torch.stack(uniform_frame_sampling(segment_frames, target_frames)) 
    else:
        video = torch.stack(segment_frames)
    video = pad_tensor(video, target_frames)
    return video


class TrainCollator():
    def __init__(self, model_type, target_size):
        self.model_type = model_type
        self.target_size = target_size
    def __call__(self, batch):
        if self.model_type in ['VidNeXt', 'ConvNeXtVanillaTransformer', 'ResNetNSTtransformer', 'ViViT']:
            dims_shape = [0, 1, 2, 3, 4] # FOR VidNeXt and its ablation variant, and ViViT
        else:
            dims_shape = [0, 2, 1, 3, 4] # FOR Rest of the models
        train_trans = video_transforms.Compose([
                video_transforms.RandomHorizontalFlip(),
                video_transforms.RandomCrop(700),
                video_transforms.Resize(self.target_size, antialias=True),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        if self.model_type in ['TimeSformer', 'ViViT']:
            temp = [load_and_augment_video(video_path, target_frames=16) for video_path, label in batch]
        else:
            temp = [load_and_augment_video(video_path) for video_path, label in batch]
        transformed_video = torch.stack([train_trans(video) for video in temp])
        return transformed_video.permute(*dims_shape), torch.tensor([label for _, label in batch])


class TestCollator():
    def __init__(self, model_type, target_size):
        self.model_type = model_type
        self.target_size = target_size
    def __call__(self, batch):
        if self.model_type in ['VidNeXt', 'ConvNeXtVanillaTransformer', 'ResNetNSTtransformer', 'ViViT']:
            dims_shape = [0, 1, 2, 3, 4] # FOR VidNeXt and its ablation variant, and ViViT
        else:
            dims_shape = [0, 2, 1, 3, 4] # FOR Rest of the models
        test_trans = video_transforms.Compose([
            video_transforms.Resize(self.target_size, antialias=True),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.model_type in ['TimeSformer', 'ViViT']:
            temp = [load_and_augment_video(video_path, target_frames=16) for video_path, label in batch]
        else:
            temp = [load_and_augment_video(video_path) for video_path, label in batch]
        transformed_video = torch.stack([test_trans(video) for video in temp])
        return transformed_video.permute(*dims_shape), torch.tensor([label for _, label in batch])

