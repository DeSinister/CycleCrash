import cv2
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_timestamp(timestamp_str):
    # Split the timestamp string into components
    components = timestamp_str.split(':')
    # Extract and convert components to integers
    hours = int(components[0])
    minutes = int(components[1])
    seconds = int(components[2])
    centiseconds = int(components[3])
    return hours, minutes, seconds, centiseconds

def subtract_timestamps(timestamp_str1, timestamp_str2):
    # Parse both timestamps
    hours1, minutes1, seconds1, centiseconds1 = parse_timestamp(timestamp_str1)
    hours2, minutes2, seconds2, centiseconds2 = parse_timestamp(timestamp_str2)
    # Calculate the differences for each component
    hours_diff = hours2 - hours1
    minutes_diff = minutes2 - minutes1
    seconds_diff = seconds2 - seconds1
    centiseconds_diff = centiseconds2 - centiseconds1
    
    # Handle negative differences (borrowing)
    if centiseconds_diff < 0:
        centiseconds_diff += 100
        seconds_diff -= 1
    if seconds_diff < 0:
        seconds_diff += 60
        minutes_diff -= 1
    if minutes_diff < 0:
        minutes_diff += 60
        hours_diff -= 1
    result = hours_diff*3600 + minutes_diff*60 + seconds_diff + centiseconds_diff/100
    return abs(result)



def uniform_frame_sampling(frame_list, target_frames):
    # Calculate the step size to achieve uniform sampling
    step_size = len(frame_list) // (target_frames - 1)

    # Use list comprehension to select frames at uniform intervals
    sampled_list = [frame_list[i * step_size] for i in range(target_frames - 1)] + [frame_list[-1]]

    return sampled_list

def pad_tensor(video_tensor, target_frames=30):
    if video_tensor.shape[0] <target_frames:
        return torch.cat([video_tensor, torch.zeros(max(target_frames - video_tensor.size(0), 0), *video_tensor.shape[1:])])
    if video_tensor.shape[0] >target_frames:
        return video_tensor[:target_frames]
    return video_tensor


# Fucntion to Calculate the Duration of an epoch
def epoch_time(start_time, end_time):
    duration = end_time - start_time
    minutes = duration//60
    seconds = duration - (60*minutes)
    return int(minutes), int(seconds)


def train_model_acc_grad(model, loader, optimizer, loss_fn, device, scaler, metric, metric2, opt_step_size=1, task_type='binclass'):
    # Intializing starting Epoch loss as 0
    loss_for_epoch = 0.0 
    score = 0.0   
    score2 = 0.0   
    # Model to be used in Training Mode
    model.train()
    iters = 0
    # For every Input Image, Label Image in a Batch
    for x, y in tqdm(loader):

        # Storing the Images to the Device
        x = x.to(device)
        y = y.to(device)
        # Set Gradient of all parameters to 0
        # Using Unscaled Mixed Precision using half Bit for Faster Processing
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Get Predictions from Model
            y_pred = model(x)

            # Calculate the Loss
            y_pred = torch.squeeze(y_pred, dim=1)
            # y = torch.unsqueeze(y, 1)
            loss = loss_fn(y_pred, y)/opt_step_size
            
            if task_type == 'multiclass':
                y_pred = torch.argmax(y_pred, dim=1)
            score += metric(y_pred, y)
            score2 += metric2(y_pred, y)

        # Scale Loss Backwards
        scaler.scale(loss).mean().backward()

        # Unscale the Gradients in Optimizer
        if iters % opt_step_size == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Add the Loss for every sample in a Batch
        loss_for_epoch += loss.item()
        iters+=1

    
    if iters % opt_step_size == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # Calculating The Average Loss for the Epoch
    loss_for_epoch =  torch.div(loss_for_epoch, len(loader))
    score =  torch.div(score, len(loader))
    score2 =  torch.div(score2, len(loader))
    return loss_for_epoch, score, score2






def train_model(model, loader, optimizer, loss_fn, device, scaler, metric, metric2, opt_step_size=1, task_type='binclass', model_type='vidnext'):
    # Intializing starting Epoch loss as 0
    loss_for_epoch = 0.0 
    score = 0.0   
    score2 = 0.0   
    # Model to be used in Training Mode
    model.train()

    # For every Input Image, Label Image in a Batch
    for x, y in tqdm(loader):

        # Storing the Images to the Device
        x = x.to(device, dtype=torch.float16)
        y = y.to(device)
        # Set Gradient of all parameters to 0
        optimizer.zero_grad()
        
        # Using Unscaled Mixed Precision using half Bit for Faster Processing
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Get Predictions from Model
            y_pred = model(x)

            # Calculate the Loss
            if task_type != "multiclass" or model_type not in ['slow_r50', 'r2plus1d_r50', 'x3d_xs', 'x3d_s', 'x3d_m']:
                y_pred = torch.squeeze(y_pred, dim=1)
            loss = loss_fn(y_pred, y)
            
            if task_type == 'multiclass':
                y_pred = torch.argmax(y_pred, dim=1)
            score += metric(y_pred, y)
            score2 += metric2(y_pred, y)

        # Scale Loss Backwards
        scaler.scale(loss).mean().backward()

        # Unscale the Gradients in Optimizer
        scaler.unscale_(optimizer)

        # Clip the Gradients to they dontreach inf
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        scaler.step(optimizer)
        
        # Update the Scaler
        scaler.update()

        # Add the Loss for every sample in a Batch
        loss_for_epoch += loss.item()
    # Calculating The Average Loss for the Epoch
    loss_for_epoch =  torch.div(loss_for_epoch, len(loader))
    score =  torch.div(score, len(loader))
    score2 =  torch.div(score2, len(loader))
    return loss, score, score2










# Function to Evaluate the Model
def evaluate_model(model, loader, loss_fn, device, metric, metric2, task_type):
    # Intializing starting Epoch loss as 0
    total_loss = 0.0
    score = 0.0
    score2 = 0.0
    # Model to be used in Evaluation Mode
    model.eval()

    # Gradients are not calculated
    with torch.no_grad():
        
        # For every Input Image, Label Image in a Batch
        for x, y in tqdm(loader):

            # Storing the Images to the Device
            x = x.to(device)
            y = y.to(device)
            
            # Get Predictions from Model
            y_pred = model(x)

            y_pred = torch.squeeze(y_pred, 1)

            if task_type == 'binclass':
                y_pred = torch.sigmoid(y_pred)

            # Calculate the Loss
            total_loss += loss_fn(y_pred, y)
            
            if task_type == 'multiclass':
                y_pred = torch.argmax(y_pred, dim=1)
            score += metric(y_pred, y)
            score2 += metric2(y_pred, y)


        # Calculating The Average Loss for the Epoch
        total_loss =  torch.div(total_loss, len(loader))
        score =  torch.div(score, len(loader))
        score2 =  torch.div(score2, len(loader))
        
    return total_loss, score, score2

