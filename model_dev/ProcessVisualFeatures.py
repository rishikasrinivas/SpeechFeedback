import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torchvision
from torchvision.transforms import transforms
import os
from VideoDataloader import load_video_frames, VideoDataloader
'''
plan:
use media pipe to get movements in input
track these movements per frame
'''
def mediapipe(image):

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(image)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)
    print(detection_result)

def track_movements(idx, video, xyz_avg_movements_across_all):
    frames= load_video_frames(video,idx) #get each frame
    x_mov_glob =[]
    y_mov_glob =[]
    z_mov_glob =[]      
    for frame in frames:
        x_mov=[]
        y_mov=[]
        z_mov=[]
        xs, ys, zs,_=mediapipe(frame)
        xs_prev=xs[0]
        ys_prev=ys[0]
        zs_prev=zs[0]
        for x in xs:
            x_mov.append(x-xs_prev)
            xs_prev=xs
        x_mov_glob.append(sum(x_mov)/len(x_mov))

        for y in ys:
            y_mov.append(ys-ys_prev)
            ys_prev=ys
        y_mov_glob.append(sum(y_mov)/len(y_mov))

        for z in zs:
            z_mov.append(zs-zs_prev)
            zs_prev=zs
        z_mov_glob.append(sum(z_mov)/len(z_mov))
    x_avg_movement=sum(x_mov_glob)/len(x_mov_glob)
    y_avg_movement=sum(y_mov_glob)/len(y_mov_glob)
    z_avg_movement=sum(z_mov_glob)/len(z_mov_glob)
    xyz_avg_movements_across_all.append( (x_avg_movement, y_avg_movement, z_avg_movement))

    return xyz_avg_movements_across_all

def getAvgMovement(data):
    xyzavg=[]
    for idx,video in enumerate(data):
        xyzavg=track_movements(idx, video, xyzavg)
    return xyzavg


videos_list = [os.path.join("data/videos",fname) for fname in os.listdir("data/videos")]
transformations=transforms.Compose([
    transforms.CenterCrop(10),
    transforms.ToTensor(),
    transforms.Resize((28,28))
    
])
print("Created videos list")
ds = VideoDataloader(videos_list,transformations)

ds=torch.utils.data.DataLoader(ds, num_workers=4)

ds= torch.utils.data.TensorDataset(ds)
print(ds)
xyz_avg=getAvgMovement(ds)
#calculatePercentDiff (avg_from_all-thisone)/thisone x 100 -----> score

