
import cv2
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import FetchData, TestCollator, TrainCollator
from get_model import VideoEncoder
from utils import train_model, evaluate_model, epoch_time, train_model_acc_grad
from torchmetrics.classification import Accuracy, MulticlassF1Score, BinaryAccuracy, BinaryF1Score
from torchvision import transforms as video_transforms
import time
import plotly.graph_objects as go
import os
import argparse
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.manual_seed(42)
np.random.seed(42)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch Size")
    parser.add_argument('-e', '--epochs', type=int, default=50, help="Number of Epochs")
    parser.add_argument('-e_p', '--early_stop_patience', type=int, default=15, help="Patience for Early Stopping")
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-6, help="Learning Rate")
    parser.add_argument('-lr_p', '--learning_rate_patience', type=int, default=5, help="Patience for Decaying Learning Rate")
    parser.add_argument('-dev', '--device', type=str, default='cuda', help="Device type: 'cpu', 'cuda'")
    parser.add_argument('-w', '--num_workers', type=int, default=16, help="Number of Worker Threads")
    parser.add_argument('-img_size', '--target_size', type=str, default='224x224', help="Image Dimension required by the model")
    parser.add_argument('-seg_len', '--segment_length', type=float, default=1, help="Video Sequence Segment Length in Seconds")
    parser.add_argument('-seg_over', '--segment_overlap', type=float, default=0.5, help="Video Sequence Overlap Length in Seconds")
    parser.add_argument('-model', '--model_type', type=str, default='VidNeXt', help="Type of the model: ['slow_r50', 'r2plus1d_r50', 'x3d_xs', 'x3d_s', 'x3d_m', 'TimeSformer', 'ViViT', 'ResNetNSTtransformer', 'ConvNeXtVanillaTransformer', 'VidNeXt']")
    parser.add_argument('-task', '--task_type', type=str, default='Collision Anticipation', help="Type of the Task: ['Time-to-collision', 'Collision Anticipation', 'Right of Way', ... ]")
    parser.add_argument('-vid_dir', '--video_dir', type=str, help="Path for the Video Directory")
    parser.add_argument('-csv_file', '--csv_file', type=str, help="CSV File Path for Dataset Labels")

    args = parser.parse_args()

    batch_size = args.batch_size
    num_of_epochs = args.epochs
    learning_rate = args.learning_rate
    learning_rate_patience = args.learning_rate_patience
    early_stop_patience = args.early_stop_patience
    device = torch.device(args.device)
    num_workers = args.num_workers
    target_size = [int(x) for x in args.target_size.lower().split('x')]
    if len(target_size) != 2:
        print("INVALID IMAGE DIMENSION")
    segment_length = args.segment_length
    segment_interval = args.segment_overlap
    model_type= args.model_type
    task = args.task_type
    if args.video_dir:
        video_dir = args.video_dir
    else:
        print("VIDEO DIRECTORY IS MISSING")
    if args.csv_file:
        csv_path = args.csv_file
    else:
        print("FILE PATH FOR DATASET LABELS IS NOT SET")

    # SET UP DATALOADER
    train_dataset = FetchData(csv_path, video_dir, task_name=task, set_name = 'train', segment_duration=segment_length, segment_interval=segment_interval, target_size=target_size, future_sight = 1, after_acc = True)
    train_collate_fn = TrainCollator(model_type=model_type, target_size=target_size)
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last = True,
            pin_memory=False,
            collate_fn=train_collate_fn,
        )   
    task_type = train_dataset.task_type
    num_classes = train_dataset.num_classes
    test_dataset = FetchData(csv_path, video_dir, task_name=task, set_name = 'test', segment_duration= segment_length, segment_interval=segment_interval, target_size=target_size, future_sight = 1, after_acc = True)
    test_collate_fn = TestCollator(model_type=model_type, target_size=target_size)
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last = True,
            pin_memory=False,
            collate_fn=test_collate_fn,
        )
    
    # --- Model Configurations ---
    model = VideoEncoder(model_type = model_type, task_type=task_type, num_classes=num_classes, segment_length=segment_length).to(device)
    model = nn.DataParallel(model)
    
    scaler = torch.cuda.amp.GradScaler()
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=learning_rate_patience, verbose=True)
    # Set up the Loss Functions and Metrics
    if task_type == "multiclass":
        loss_function = nn.CrossEntropyLoss() 
        metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        highest_score = highest_score2 = 0
        metric2 = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    elif task_type == "binclass":
        loss_function = nn.BCEWithLogitsLoss() 
        metric = BinaryAccuracy().to(device)
        metric2 = BinaryF1Score().to(device)
        highest_score = highest_score2 = 0
    elif task_type == "reg":
        loss_function = nn.MSELoss()
        metric = nn.L1Loss().to(device)
        metric2 = nn.L1Loss().to(device)
        highest_score = highest_score2 = np.inf

    lowest_loss = np.inf
    training_loss = []
    training_score = []
    training_score2 = []
    test_loss = []
    test_score = []
    test_score2 = []
    bad_epochs = 0
    early_stop = False
    end_epoch = num_of_epochs
    # Loop for Every epochs
    for epoch in range(num_of_epochs):
        # Start Counting time
        start = time.time()

        # Train the Model for Every epoch
        train_value = train_model(model, train_loader, optimizer, loss_function, device, scaler, metric, metric2, opt_step_size=1, task_type=task_type, model_type=model_type)
        training_loss.append(train_value[0].detach().cpu())
        training_score.append(train_value[1].detach().cpu())
        training_score2.append(train_value[2].detach().cpu())

        # Evaluate the Model using the test Split
        etest_value = evaluate_model(model, test_loader, loss_function, device, metric, metric2, task_type)
        test_loss.append((etest_value[0]).detach().cpu())
        test_score.append((etest_value[1]).detach().cpu())
        test_score2.append((etest_value[2]).detach().cpu())
        
        # Save the Model If the Model is Performing better on test set while Training
        if test_loss[-1] < lowest_loss:
            # Reset Patience for Early Stopping
            bad_epochs = 0
            print(f"test Loss Decreased from {lowest_loss} to {test_loss[-1]}")
            # Changing the Lowest Loss to Current test Loss
            lowest_loss = test_loss[-1]
            # Saving the Model
            torch.save(model.module.state_dict(), rf"{model_type}_{task}_{epoch}_loss{test_loss[-1]}.pt")
        elif (task_type == 'reg' and test_score[-1] < highest_score) or (task_type in ['binclass', 'multiclass'] and test_score[-1] > highest_score):
            # Reset Patience for Early Stopping
            bad_epochs = 0
            print(f"test Score changed from {highest_score} to {test_score[-1]}")
            # Changing the Lowest Loss to Current test Loss
            highest_score = test_score[-1]
            # Saving the Model
            torch.save(model.module.state_dict(), rf"{model_type}_{task}_{epoch}_loss{test_loss[-1]}.pt")
        elif (task_type == 'reg' and test_score2[-1] < highest_score2) or (task_type in ['binclass', 'multiclass'] and test_score2[-1] > highest_score2):
            # Reset Patience for Early Stopping
            bad_epochs = 0
            print(f"test Score 2 changed from {highest_score2} to {test_score2[-1]}")
            # Changing the Lowest Loss to Current test Loss
            highest_score2 = test_score2[-1]
            # Saving the Model
            torch.save(model.module.state_dict(), rf"{model_type}_{task}_{epoch}_loss{test_loss[-1]}.pt")
        else:
            # Model not performning better for this epoch
            bad_epochs+=1
        # Stop the Counting Time
        end = time.time()

        # Estimate the Duration
        minutes, seconds = epoch_time(start, end)

        # Report Training and test Loss
        print(f"Epoch Number: {epoch+1}")
        print(f"Duration: {minutes}m {seconds}s")
        print(f"Training Loss: {training_loss[-1]}")
        print(f"Training Score: {training_score[-1]}")
        print(" Training Score2: ", training_score2[-1])
        print(f"test Loss: {test_loss[-1]}")
        print(f"test Score: {test_score[-1]}")
        print(f"test Score2: {test_score2[-1]}")
        print()

        # If Patience Level reached for Model not Performing better
        if bad_epochs == early_stop_patience:
            print("Stopped Early. The Model is not improving over test loss")
            end_epoch = epoch
            break

    epochs = [x+1 for x in range(len(training_loss))]

    # Create two subplots
    fig = go.Figure()
    # Training Loss and Score subplot
    fig.add_trace(go.Scatter(x=epochs, y=training_loss, mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(x=epochs, y=test_loss, mode='lines', name='test Loss'))

    # Update layout
    fig.update_layout(title='Training Metrics',
                    xaxis_title='Epochs',
                    yaxis_title='Metrics',
                    legend=dict(x=0, y=1, traceorder='normal'))

    # fig.write_image("regression_training.jpeg")
    fig.write_html('regression_loss.html')

    # test Loss and Score subplot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=epochs, y=training_score, mode='lines', name='Training Score'))
    fig1.add_trace(go.Scatter(x=epochs, y=test_score, mode='lines', name='test Score'))


    # Update layout
    fig1.update_layout(title='test Metrics',
                    xaxis_title='Epochs',
                    yaxis_title='Metrics',
                    legend=dict(x=0, y=1, traceorder='normal'))

    fig1.write_html('regression_score.html')
    # Show the plot
    fig.show()
