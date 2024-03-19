import numpy as np
import os
import torch
from PIL import Image
from env import MyEnv
from tqdm import tqdm



DATASET_DIR = "./dataset_rot_pos_clf"
NUM_GROUPS = 1000
NUM_SAMPLES = 5000
TILES_TO_NUM = {"grass" : 0, "tile" : 1, "straight" : 2, "straight_r" : 3, "turn_bl" : 4, "turn_br" : 5, "turn_fl" : 6, "turn_fr" : 7}

LIGHT_VALS = np.linspace(0.05, 0.4, int(7))
TEMP_VALS = np.linspace(0.2, 0.6, int(7))
ANGLE_VALS = np.linspace(-15, 15, int(3))
POS_VALS = [-1, 0, 1]
if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

env = MyEnv(render_mode=None, map_file="./maps/random.yaml", light_rand=True)

#accum_group = []
for group in range(NUM_GROUPS):
    env.reset(map_file="./maps/random.yaml")
    if not os.path.exists(DATASET_DIR + "/group_{}".format(group)):
        os.mkdir(DATASET_DIR + "/group_{}".format(group))
   # Y = torch.nn.functional.one_hot(torch.tensor(TILES_TO_NUM[env.start_tile.name]), num_classes=8).float()
    Y = torch.zeros([10]).int()
    i = 0;
    for j, row in enumerate(reversed(env.tiles)):
        if j >=3:
            continue
        for tile in row:
            Y[i] = TILES_TO_NUM[tile.name]
            i += 1
    Y[i] = TILES_TO_NUM[env.start_tile.name]

    torch.save(Y, DATASET_DIR + "/group_{}/".format(group) + "Y.pth")
    accum_samples = []
    for sample in tqdm(range(len(LIGHT_VALS) + len(TEMP_VALS)), desc="Proccesing group {}".format(group)):
        for i, angle in enumerate(ANGLE_VALS):
            for j, pos in enumerate(POS_VALS):
                if sample < len(LIGHT_VALS):
                    obs, info = env.reset(light_color_factor=LIGHT_VALS[sample], init_angle = angle, init_pos = pos)
                else:
                    obs, info = env.reset(light_temp_factor=TEMP_VALS[sample-len(LIGHT_VALS)], init_angle = angle, init_pos = pos)
                accum_samples.append(torch.tensor(obs.copy()).unsqueeze(0))
           # np.save(DATASET_DIR + "/group_{}/".format(group) + "sample_{}".format(sample*len(ANGLE_VALS) + i), obs)
                if group == 0:
                    img = Image.fromarray(obs)
                    img.save(DATASET_DIR + "/group_{}/".format(group) + "sample_{}.jpg".format(sample*len(ANGLE_VALS)*len(POS_VALS) + i*len(ANGLE_VALS) + j))
    accum_samples = torch.cat(accum_samples, dim = 0).float()/255
    torch.save(accum_samples, DATASET_DIR + "/group_{}".format(group) + "/data.pth")
    #accum_group.append(accum_samples)

#accum_group = np.concatenate(accum_group)
#np.save(DATASET_DIR + "/dataset", accum_group)














