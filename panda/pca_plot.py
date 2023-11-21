import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from sklearn.decomposition import PCA
from FE_train import TripletsDataSetRandomLocal, FE_class
from tqdm import tqdm

DATASET_FOLDER_VAL = "dataset_trp_server/dataset_trp_val"
DATASET_FOLDER_FIT = "dataset_trp_server/dataset_trp"
TILES_TO_NUM = {"grass" : 0, "tile" : 1, "straight" : 2, "straight_r" : 3, "turn_bl" : 4, "turn_br" : 5, "turn_fl" : 6, "turn_fr" : 7}
NUM_TO_TILES = {}
for (key, value) in TILES_TO_NUM.items():
    NUM_TO_TILES[value] = key

START_GROUP = 0
VAL = True
GROUPS_TO_PLOT = 30
VERSION = "v2.6.1"
IS_3D = False
NUM_SAMPLES = 40
NUM_FEATURES = 32
dataset_val= TripletsDataSetRandomLocal(dataset_dir=DATASET_FOLDER_VAL, group_folder_prefix="group_", sample_name_prefix="sample_", num_groups=100, num_samples=NUM_SAMPLES)
dataset_fit= TripletsDataSetRandomLocal(dataset_dir=DATASET_FOLDER_FIT, group_folder_prefix="group_", sample_name_prefix="sample_", num_groups=1000, num_samples=NUM_SAMPLES)


model = FE_class(features_dim=NUM_FEATURES, dropout_rate=0.0)
model.load_state_dict(torch.load("./FE_model_{}/epoch_7000".format(VERSION), map_location="cpu"))
model.eval()
feature_vectors_val = np.zeros([NUM_SAMPLES*100, NUM_FEATURES])
feature_vectors_val_groups = np.zeros([NUM_SAMPLES*100, 1])
feature_vectors_fit = np.zeros([int(NUM_SAMPLES*1000/3), NUM_FEATURES])
tiles_prediction = np.zeros([NUM_SAMPLES*100, 8])
def euclidean_distance(x, y):
    """
    Compute Euclidean distance between two tensors.
    """
    #print(torch.pow((x-y), 2).sum(dim=1))
    return torch.pow(torch.pow((x-y), 2).sum(dim=1) + 1e-6, 0.5)
for i, [anchor, positive, negative] in enumerate(tqdm(dataset_fit, desc="Making feature vectors matrix for fit...")):
    #print(anchor)
    break
    with torch.no_grad():
        feature_vectors_fit[i, :] = model(anchor.unsqueeze(0))[0] 
        
        #feature_vectors_val_groups[i,:] = int(i//NUM_SAMPLES)
    if i == int(len(dataset_fit)/3) - 1:
        break;

for i, [anchor, positive, negative] in enumerate(tqdm(dataset_val, desc="Making feature vectors matrix for plot...")):
        
    with torch.no_grad():
        feature_vectors_val[i, :], tiles_prediction[i, :] = model(anchor.unsqueeze(0))
        feature_vectors_val_groups[i,:] = int(i//NUM_SAMPLES)
        
        
    if i == len(dataset_val) - 1:
        break;
    #if i == 79:
    #    break;

    


feature_vectors_val_groups = feature_vectors_val_groups.squeeze(1)
pca = PCA(n_components=3 if IS_3D else 2)

pca.fit(feature_vectors_val)

X = pca.transform(feature_vectors_val[START_GROUP*NUM_SAMPLES:(GROUPS_TO_PLOT+START_GROUP)*NUM_SAMPLES, :])


fig = plt.figure(layout="constrained", figsize=(20, 12))
subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 1.2])


#plt.clf()
if IS_3D:
    ax = subfigs[0].add_subplot(111, projection = "3d")#, projection="2d", elev=48, azim=134)
else:
    ax = subfigs[0].add_subplot(111)
#ax.set_position([0, 0, 0.95, 1])


#plt.cla()
for name, label in [(str(i), i) for i in range(START_GROUP, START_GROUP+GROUPS_TO_PLOT)]:
    if IS_3D:
        ax.text(
            X[feature_vectors_val_groups[START_GROUP*NUM_SAMPLES:(GROUPS_TO_PLOT+START_GROUP)*NUM_SAMPLES] == label, 0].mean(),
            X[feature_vectors_val_groups[START_GROUP*NUM_SAMPLES:(GROUPS_TO_PLOT+START_GROUP)*NUM_SAMPLES] == label, 1].mean(),
            X[feature_vectors_val_groups[START_GROUP*NUM_SAMPLES:(GROUPS_TO_PLOT+START_GROUP)*NUM_SAMPLES] == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.7, edgecolor="b", facecolor="w"),
        )
    else:
        ax.text(
            X[feature_vectors_val_groups[START_GROUP*NUM_SAMPLES:(GROUPS_TO_PLOT+START_GROUP)*NUM_SAMPLES] == label, 0].mean(),
            X[feature_vectors_val_groups[START_GROUP*NUM_SAMPLES:(GROUPS_TO_PLOT+START_GROUP)*NUM_SAMPLES] == label, 1].mean(),
            #X[feature_vectors_val_groups[:GROUPS_TO_PLOT*40] == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.7, edgecolor="b", facecolor="w"),
        )
#Y = np.choose(feature_vectors_val_groups.squeeze(1), list(range(1000))).astype(float)
if IS_3D:
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = feature_vectors_val_groups[START_GROUP*NUM_SAMPLES:(GROUPS_TO_PLOT+START_GROUP)*NUM_SAMPLES].astype(float), cmap=plt.cm.nipy_spectral, edgecolor="k")
else:
    ax.scatter(X[:, 0], X[:, 1], c = feature_vectors_val_groups[START_GROUP*NUM_SAMPLES:(GROUPS_TO_PLOT+START_GROUP)*NUM_SAMPLES].astype(float), cmap=plt.cm.nipy_spectral, edgecolor="k")

axs = subfigs[1].subplots(5, GROUPS_TO_PLOT//5)
for i in range(START_GROUP, START_GROUP + GROUPS_TO_PLOT):
    axs[(i-START_GROUP)%5][(i-START_GROUP)//5].imshow(np.load(DATASET_FOLDER_VAL + "/group_{}/sample_19.npy".format(i)))
    axs[(i-START_GROUP)%5][(i-START_GROUP)//5].set_title(str(i) + " {}".format(NUM_TO_TILES[np.argmax(tiles_prediction[i*40+20]).astype(int)]))
#manager = plt.get_current_fig_manager()
#manager.full_screen_toggle()
fig.savefig("pca_plot_{}_val.png".format(VERSION))
plt.show()