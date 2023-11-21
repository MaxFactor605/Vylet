import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn.decomposition import PCA
from FE_train import TripletsDataSetRandom, TripletsDataSetRandomLocal, FE, FE_class
from tqdm import tqdm
import cv2
import sys
import time
maps = ["round", "round-grass", "round-r", "round-grass-r", "curvy", "curvy-r", "long", "long-r", "big", "big-r",
            "zig-zag", "zig-zag-r", "plus", "plus-r", "H", "H-r"]
TILES_TO_NUM = {"grass" : 0, "tile" : 1, "straight" : 2, "straight_r" : 3, "turn_bl" : 4, "turn_br" : 5, "turn_fl" : 6, "turn_fr" : 7}
NUM_TO_TILES = {}
for (key, value) in TILES_TO_NUM.items():
    NUM_TO_TILES[value] = key
FEATURES_NUM = 32
VIDEO_FILE = "./video_big_v5/ppo_84000000/"
VERSION = "v2.6.4"
ANIM = True
IS_3D = False
MAP_TO_SHOW = "round";
NUM_SAMPLES = 60
DATASET_FOLDER_FIT = "dataset_trp_server/dataset_trp_rot_v2"

frameCountAll = 0
videos = []
video_beg = 0;

cap = cv2.VideoCapture(VIDEO_FILE + "{}.mp4".format(MAP_TO_SHOW))
frameCount =  min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2000)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frameCountAll += frameCount

buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
print(frameCount, frameHeight, frameWidth)
fc = 0
ret = True

while (fc < frameCount  and ret):
    ret, fr = cap.read()
    fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
    buf[fc] = fr
    fc += 1
   



print(buf.shape)
model = FE_class(features_dim=FEATURES_NUM, dropout_rate=0.0)
model.load_state_dict(torch.load("./FE_model_{}/epoch_7000".format(VERSION), map_location="cpu"))
model.eval()

dataset_fit= TripletsDataSetRandomLocal(dataset_dir=DATASET_FOLDER_FIT, group_folder_prefix="group_", sample_name_prefix="sample_", num_groups=1000, num_samples=NUM_SAMPLES)

feature_vectors_fit = np.zeros([3000, FEATURES_NUM])
for i in range(0, 3000, 3):#, [anchor, positive, negative] in enumerate(tqdm(dataset_fit, desc="Making feature vectors matrix for fit...")):
    print(i)
    with torch.no_grad():
        for j in range(3):
            feature_vectors_fit[i+j, :] = model(dataset_fit[NUM_SAMPLES*(i//3) + 27+j][0].unsqueeze(0))[0]
        #feature_vectors_val_groups[i,:] = int(i//NUM_SAMPLES)
    if i == int(len(dataset_fit)) - 1:
        break;


feature_vectors = np.zeros((frameCount, FEATURES_NUM))
tile_pred = np.zeros([frameCount, 8])


for i, obs in tqdm(enumerate(buf), desc="Gathering feature vectors"):
    with torch.no_grad():
        #print(obs)
        feature_vectors[i, :], tile_pred[i, :] = model(torch.from_numpy(obs).unsqueeze(0)/255)

def euclidean_distance(x, y):
    """
    Compute Euclidean distance between two tensors.
    """
    #print(torch.pow((x-y), 2).sum(dim=1))
    return torch.pow(torch.pow((x-y), 2).sum(dim=0) + 1e-6, 0.5)



def find_nearest(sample, dataset_features):
    nearest = None
    nearest_dist = 999999
    for i, data_sample in enumerate(dataset_features):
        dist = euclidean_distance(torch.tensor(sample), torch.tensor(data_sample))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest = dataset_fit[NUM_SAMPLES*(i//3) + 27 + (i%3)][0]

    return nearest

#plt.scatter(X[:, 0], X[:,1], c=colors, cmap=plt.cm.nipy_spectral, edgecolor="k")
if ANIM:
    from matplotlib.animation import FuncAnimation
    
   
    fig, axs = plt.subplots(1,2, figsize=(20,12))
    vid1 = axs[0].imshow(np.zeros((128, 128,3)))
    vid2 = axs[1].imshow(np.zeros((128,128,3)))

    def init():
        
        return vid1,vid2

    def update(frame):
        if (frame % 10 == 0):
     
            vid1.set_data(buf[frame])
            vid2.set_data(find_nearest(feature_vectors[frame], feature_vectors_fit)) 
        #ln.set_data(xdata, ydata, c=colors)
        
            print(NUM_TO_TILES[np.argmax(tile_pred[video_beg + frame, :]).astype(int)])
        #print(np.rint(tile_pred[frame, :].astype(int)))
        return vid1, vid2

    ani = FuncAnimation(fig, update, frames=frameCount,
                    init_func=init, blit=True, interval=0)
    plt.show()
