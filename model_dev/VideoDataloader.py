#load pretrained vae
import torch
from torch.utils.data import Dataloader, Dataset
#use cv to capture every frame 
import cv2
import os 
import Imagep
class VideoDataloader(Dataset):
    def __init__(self,video_list, transformations):
        self.video_list=video_list
        self.transformations = transformations

    def __len__(self):
        return len(self.video_list)
    


    def __getitem__(self,idx):
        return self.video_list[idx]

def load_video_frames(videoFilename,video_idx):
    #use cv to capture each frame of 
    cap = cv2.VideoCapture(videoFilename)
    if not cap.isOpened():
        exit()
    num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        file="data/frames/video"+str(video_idx)+"/"+str(i)+".jpg"
        cv2.imwrite(file, frame)
    cap.release()
