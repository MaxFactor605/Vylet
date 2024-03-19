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
VIDEO_FILE = "./video_big_v8.3/ppo_25000000/"
VERSION = "v2.6.4"
ANIM = True
IS_3D = False
MAP_TO_SHOW = "plus";
NUM_SAMPLES = 60
DATASET_FOLDER_FIT = "dataset_trp_server/dataset_trp_rot_v2_val"

frameCountAll = 0
videos = []
video_beg = 0;
video_count = 0;
for map in maps:
    cap = cv2.VideoCapture(VIDEO_FILE + "{}.mp4".format(map))
    frameCount =  min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2000)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if map == MAP_TO_SHOW:
        video_beg = frameCountAll
        video_count = frameCount
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
    videos.append(buf)

videos = np.concatenate(videos, axis = 0)

print(buf.shape)
model = FE_class(features_dim=FEATURES_NUM, dropout_rate=0.0)
model.load_state_dict(torch.load("./FE_model_{}/epoch_7000".format(VERSION), map_location="cpu"))
model.eval()
"""
dataset_fit= TripletsDataSetRandomLocal(dataset_dir=DATASET_FOLDER_FIT, group_folder_prefix="group_", sample_name_prefix="sample_", num_groups=100, num_samples=NUM_SAMPLES)

feature_vectors_fit = np.zeros([int(NUM_SAMPLES*1000/3), FEATURES_NUM])
for i, [anchor, positive, negative] in enumerate(tqdm(dataset_fit, desc="Making feature vectors matrix for fit...")):
        
    with torch.no_grad():
        feature_vectors_fit[i, :] = model(anchor.unsqueeze(0))[0]
        #feature_vectors_val_groups[i,:] = int(i//NUM_SAMPLES)
    if i == int(len(dataset_fit)) - 1:
        break;

"""
feature_vectors = np.zeros((frameCountAll, FEATURES_NUM))
tile_pred = np.zeros([frameCountAll, 8])

obs1 = buf[10]
for i, obs in tqdm(enumerate(videos), desc="Gathering feature vectors"):
    with torch.no_grad():
        #print(obs)
        feature_vectors[i, :], tile_pred[i, :] = model(torch.from_numpy(obs).unsqueeze(0)/255)

frag_length = frameCount
pca = PCA(n_components=3 if IS_3D else 2)

pca.fit(feature_vectors)

X = pca.transform(feature_vectors[video_beg:video_beg+video_count, :])


colors = np.array([[i,0,0] for i in np.linspace(0, 255, frag_length)])/255
#plt.scatter(X[:, 0], X[:,1], c=colors, cmap=plt.cm.nipy_spectral, edgecolor="k")
if ANIM:
    from matplotlib.animation import FuncAnimation
    
   
    fig, axs = plt.subplots(1,2, figsize=(20,12))
    if IS_3D:
        xdata, ydata, zdata, colorsdata = [],[],[],[]
        axs[0].remove()
        axs[0] = fig.add_subplot(121, projection="3d")
        ln, = axs[0].plot([], [], [], "ro")
    else:
        xdata, ydata, colorsdata = [], [],[]
        ln = axs[0].scatter([], [], c=[])
    vid = axs[1].imshow(np.zeros((128,128,3)))

    def init():
        axs[0].set_xlim(np.min(X), np.max(X))
        axs[0].set_ylim(np.min(X), np.max(X))
        if IS_3D:
            axs[0].set_zlim(np.min(X), np.max(X))
        global xdata, ydata, colorsdata 
        xdata, ydata, colorsdata = [], [],[]
        return ln,

    def update(frame):
        xdata.append(X[frame, 0])
        ydata.append(X[frame, 1])
        if IS_3D:
            zdata.append(X[frame, 2])
        colorsdata.append([0,1,0])
     
        vid.set_data(videos[video_beg+frame]) 
        #ln.set_data(xdata, ydata, c=colors)
        if IS_3D:
            global ln
            ln.set_data(xdata, ydata)
            ln.set_3d_properties(zdata)
        else:
            ln = axs[0].scatter(xdata, ydata, c=colorsdata)
        colorsdata.pop(-1)
        colorsdata.append(colors[frame])
        print(NUM_TO_TILES[np.argmax(tile_pred[video_beg + frame, :]).astype(int)])
        #print(np.rint(tile_pred[frame, :].astype(int)))
        return ln, vid

    ani = FuncAnimation(fig, update, frames=frag_length,
                    init_func=init, blit=True, interval=50)
    plt.show()
else:
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(X[:, 0], X[:,1], np.array(list(range(video_count))), c =colors, cmap=plt.cm.nipy_spectral, edgecolor="k")
    plt.show()
