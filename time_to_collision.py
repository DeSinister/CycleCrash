import cv2
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanSquaredError, R2Score
from torchvision import transforms as video_transforms
import time
import plotly.graph_objects as go
import os
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.manual_seed(42)
np.random.seed(42)

batch_size = 32
num_of_epochs = 50
learning_rate = 2e-5
learning_rate_patience = 5
early_stop_patience = 15
device = torch.device('cuda')
num_workers = 16
target_size = (224, 224)
segment_length = 1
model_type= 'r2plus1d_r50' # Change the following to change the model type 'r2plus1d', 'slow_r50', 'x3d_m', 'x3d_s', and 'x3d_xs'
task = 'TTC'
dataset_dir = 'unsafe_videos_dir' # Directory Containing Videos from Id 0 to 999 in Final.csv file
csv_path = 'train.csv'
val_csv_path = 'val.csv'

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

class FetchData():
    def __init__(self, csv_path, video_dir, length = 300, segment_duration=1, segment_interval=0.5, target_size=(224, 224), future_sight = 1):
        self.df = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.segment_duration = segment_duration
        self.segment_interval = segment_interval
        self.target_size = target_size
        self.frames_list = []
        self.labels = []
        self.future_sight = future_sight
        self.length_videos = length
        self.prepare_data()

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

    def create_time_segments(self, total_time, accident_start_time):
        time_segments = []
        current_time = 0.0
        # print("Total time: ", total_time, "Accident Time: ", accident_start_time, "Duration: ", self.segment_duration)
        while current_time + self.segment_duration <= total_time:
            
            if accident_start_time - current_time - self.segment_duration>= 0:
                time_segments.append(accident_start_time - current_time - self.segment_duration)
            else:
                time_segments.append(0)
            # print(f"From {current_time} to {current_time + self.segment_duration}, accdient happening in {time_segments[-1]}")
            current_time += self.segment_interval
        # print(time_segments)
        time_segments = torch.stack([torch.tensor(x, dtype=torch.float32) for x in time_segments])
        return time_segments
    
      
    def create_vid_segments(self, vid_path, total_time):
        vid_segments = []
        current_time = 0.0

        while current_time + self.segment_duration <= total_time:
            vid_segments.append([vid_path, current_time, current_time + self.segment_duration])
            current_time += self.segment_interval
            # break #<- Single Pass
        return vid_segments
    
    
    def prepare_data(self):
        for i in range(self.length_videos):
            if type(self.df['Time of Collision'][i]) == str and  str(self.df['Time of Collision'][i]).lower()[0] != 'n' and self.df['Duration'][i]>=self.segment_duration:
                self.frames_list.append(
                    self.create_vid_segments(
                        f"{dataset_dir}/{int(self.df['File Name'][i])}_{int(self.df['Counter'][i])}.mp4", 
                        self.df['Duration'][i])
                    )
                self.labels.append(self.create_time_segments(self.df['Duration'][i] , subtract_timestamps(self.df['Start Time'][i], str(self.df['Time of Collision'][i]))))
            elif type(self.df['Time of Collision'][i]) == str and  str(self.df['Time of Collision'][i]).lower()[0] == 'n' and self.df['Duration'][i]>=self.segment_duration:
                self.frames_list.append(
                    self.create_vid_segments(
                        f"{dataset_dir}/{int(self.df['File Name'][i])}_{int(self.df['Counter'][i])}.mp4", 
                        self.df['Duration'][i])
                    )
                self.labels.append(self.create_time_segments(self.df['Duration'][i] , subtract_timestamps(self.df['Start Time'][i], str(self.df['End Time'][i])) + self.segment_duration + self.segment_interval))
        self.labels = torch.cat(self.labels)
        self.frames_list = [item for sublist in self.frames_list for item in sublist]
        # print("Frames List: ", self.frames_list)
        # print("Labels List: ", self.labels)


train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.RandomCrop(700),
        video_transforms.Resize(target_size, antialias=True),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
test_trans = video_transforms.Compose([
        video_transforms.Resize(target_size, antialias=True),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def pad_tensor(video_tensor):
    return torch.cat([video_tensor, torch.zeros(max(round(30*(segment_length)) - video_tensor.size(0), 0), *video_tensor.shape[1:])])


def load_and_augment_video(x, num_frames=30):
    video_path = x[0]
    start = x[1]
    end = x[2]
    segment_frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    start_frame = int(start * frame_rate)
    end_frame = int(end * frame_rate)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    step_size = (end_frame - start_frame) // num_frames
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        segment_frames.append(transforms.ToTensor()(frame))   
    video = torch.stack(segment_frames) 
    if video.shape[0] <30:
        print("-------------------------------------------PROBLEM-----------------------------------------------")
        print(x, video.shape, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))/frame_rate, torch.mean(video), torch.std(video))
        video = pad_tensor(video)
    return video

def train_collate_fn(batch):
    temp = [load_and_augment_video(video_path) for video_path, label in batch]
    transformed_video = torch.stack([train_trans(video) for video in temp])
    return transformed_video.permute(0, 2, 1, 3, 4), torch.tensor([label for _, label in batch])

train_dataset = FetchData(csv_path = csv_path, video_dir=dataset_dir, length = 700, segment_duration= segment_length)
train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last = True,
        pin_memory=False,
        collate_fn=train_collate_fn,
    )   

def test_collate_fn(batch):
    temp = [load_and_augment_video(video_path) for video_path, label in batch]
    transformed_video = torch.stack([test_trans(video) for video in temp])
    return transformed_video.permute(0, 2, 1, 3, 4), torch.tensor([label for _, label in batch])

val_dataset = FetchData(csv_path = val_csv_path, video_dir=dataset_dir, length = 300, segment_duration= segment_length)
val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last = True,
        pin_memory=False,
        collate_fn=test_collate_fn,
    )
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vid_trans = torch.hub.load('facebookresearch/pytorchvideo', model_type, pretrained=True)
        self.drop = nn.Dropout(p=0.25)
        # self.bn1 = nn.BatchNorm1d(400, eps=1e-5)
        self.fc1 = nn.Linear(400, 1)


    def forward(self, x):
        x = self.vid_trans(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = F.relu(x)
        return x


# --- Model Configurations ---
model = Encoder().to(device)
model = nn.DataParallel(model)
scaler = torch.cuda.amp.GradScaler()
model.to(device)
metric = MeanSquaredError().to(device)
metric2 = R2Score().to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate) #, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=learning_rate_patience, verbose=True)
loss_function = nn.MSELoss()


# Fucntion to Calculate the Duration of an epoch
def epoch_time(start_time, end_time):
    duration = end_time - start_time
    minutes = duration//60
    seconds = duration - (60*minutes)
    return int(minutes), int(seconds)

def train_model(model, loader, optimizer, loss_fn, device, scaler):
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
        y = y.to(device, dtype=torch.float16)
        # Set Gradient of all parameters to 0
        optimizer.zero_grad()
        
        # Using Unscaled Mixed Precision using half Bit for Faster Processing
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Get Predictions from Model
            y_pred = model(x)

            # Calculate the Loss
            y = torch.unsqueeze(y, 1)
            loss = loss_fn(y_pred, y)
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
def evaluate_model(model, loader, loss_fn, device):
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
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            # Get Predictions from Model
            y_pred = model(x)

            # Calculate the Loss
            y = torch.unsqueeze(y, 1)
            total_loss += loss_fn(y_pred, y)
            score += metric(y_pred, y)
            score2 += metric2(y_pred, y)


        # Calculating The Average Loss for the Epoch
        total_loss =  torch.div(total_loss, len(loader))
        score =  torch.div(score, len(loader))
        score2 =  torch.div(score2, len(loader))
        
    return total_loss, score, score2


# ------- Training the Model -------
# Training Initialization Setup

# ------- Training the Model -------
# Training Initialization Setup
if __name__ == "__main__":
    
    lowest_loss = 9999
    highest_score = 9999
    highest_score2 = 0
    training_loss = []
    training_score = []
    validation_loss = []
    validation_score = []
    bad_epochs = 0
    early_stop = False
    end_epoch = num_of_epochs
    # Loop for Every epochs
    for epoch in range(num_of_epochs):
        # Start Counting time
        start = time.time()

        # Train the Model for Every epoch
        train_value = train_model(model, train_loader, optimizer, loss_function, device, scaler)
        training_loss.append(train_value[0].detach().cpu())
        training_score.append(train_value[1].detach().cpu())

        # Evaluate the Model using the Validation Split
        eval_value = evaluate_model(model, val_loader, loss_function, device)
        validation_loss.append((eval_value[0]).detach().cpu())
        validation_score.append((eval_value[1]).detach().cpu())
        
        # Save the Model If the Model is Performing better on Validation set while Training
        if validation_loss[-1] < lowest_loss:
            # Reset Patience for Early Stopping
            bad_epochs = 0
            print(f"Validation Loss Decreased from {lowest_loss} to {validation_loss[-1]}")
            # Changing the Lowest Loss to Current Validation Loss
            lowest_loss = validation_loss[-1]
            # Saving the Model
            torch.save(model.module.state_dict(), rf"{model_type}_{task}_{epoch}_loss{validation_loss[-1]}.pt")
        elif validation_score[-1] < highest_score:
            # Reset Patience for Early Stopping
            bad_epochs = 0
            print(f"Validation Score Increased from {highest_score} to {validation_score[-1]}")
            # Changing the Lowest Loss to Current Validation Loss
            highest_score = validation_score[-1]
            # Saving the Model
            torch.save(model.module.state_dict(), rf"{model_type}_{task}_{epoch}_loss{validation_loss[-1]}.pt")
        elif eval_value[2].detach().cpu() > highest_score2:
            # Reset Patience for Early Stopping
            bad_epochs = 0
            print(f"Validation Score 2 Increased from {highest_score2} to {eval_value[2].detach().cpu()}")
            # Changing the Lowest Loss to Current Validation Loss
            highest_score2 = eval_value[2].detach().cpu()
            # Saving the Model
            torch.save(model.module.state_dict(), rf"{model_type}_{task}_{epoch}_loss{validation_loss[-1]}.pt")
        else:
            # Model not performning better for this epoch
            bad_epochs+=1
        # Stop the Counting Time
        end = time.time()

        # Estimate the Duration
        minutes, seconds = epoch_time(start, end)

        # Report Training and Validation Loss
        print(f"Epoch Number: {epoch+1}")
        print(f"Duration: {minutes}m {seconds}s")
        print(f"Training Loss: {training_loss[-1]}")
        print(f"Training Score: {training_score[-1]}")
        print(" TRAIN R2: ", train_value[2])
        print(f"Validation Loss: {validation_loss[-1]}")
        print(f"Validation Score: {validation_score[-1]}")
        print("EVAL R2 Combined: ",(eval_value[2]).detach().cpu())
        # print("EVAL F1: ", (eval_value[2]).detach().cpu())
        print()


        # If Patience Level reached for Model not Performing better
        if bad_epochs == early_stop_patience:
            print("Stopped Early. The Model is not improving over validation loss")
            end_epoch = epoch
            break


    epochs = [x+1 for x in range(len(training_loss))]

    # Create two subplots
    fig = go.Figure()
    # Training Loss and Score subplot
    fig.add_trace(go.Scatter(x=epochs, y=training_loss, mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(x=epochs, y=validation_loss, mode='lines', name='Validation Loss'))

    # Update layout
    fig.update_layout(title='Training Metrics',
                    xaxis_title='Epochs',
                    yaxis_title='Metrics',
                    legend=dict(x=0, y=1, traceorder='normal'))

    # fig.write_image("regression_training.jpeg")
    fig.write_html('regression_loss.html')

    # Validation Loss and Score subplot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=epochs, y=training_score, mode='lines', name='Training Score'))
    fig1.add_trace(go.Scatter(x=epochs, y=validation_score, mode='lines', name='Validation Score'))


    # Update layout
    fig1.update_layout(title='Validation Metrics',
                    xaxis_title='Epochs',
                    yaxis_title='Metrics',
                    legend=dict(x=0, y=1, traceorder='normal'))

    fig1.write_html('regression_score.html')
    # Show the plot
    # fig.show()