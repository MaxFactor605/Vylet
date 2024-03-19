from car_racing import CarRacing

from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import timeout_decorator
import sys

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE 

@timeout_decorator.timeout(0.5)
def get_img(seed, checkpoints, noise, rad, right):
    env = CarRacing(checkpoints=checkpoints, noise=noise, rad=rad, full_randomize=True, right=right, car_randomize_factor=0.3)
    
    env.reset(seed=seed)

    img = env.take_screenshot()

    return img

def go_through():
       # fig, axs = plt.subplots(nrows=6, ncols=4)
    checkpoints = 3
    step = (2 * math.pi * 1 / checkpoints)/8 #(TRACK_RAD - TRACK_RAD/3)/12#
    
    save_dir = "./envs3"
    rad = 50
    
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for check in range(1, 20):
        if not os.path.exists(save_dir + "/check_{}".format(check)):
            os.mkdir(save_dir + "/check_{}".format(check))
        for rad in range(int(TRACK_RAD/3), int(TRACK_RAD), 10):
            noise = 0
            stacked_imgs = []
            for j in range(4):
                imgs = []
                for i in range(2):
                
                    try:
                        img = get_img(1, check, noise, rad, 0)
                        print(rad, noise)
                    except TimeoutError:
                        print("{}: {} - Skipped".format(rad, noise))
                        img = np.random.randint(0, 255, [350, 600, 3])
                    # print(img)
                    

                    noise += step
                    imgs.append(img[:350])
                    #img.save(save_dir + "test.jpg")
                    #sys.exit()
                collage = np.concatenate(imgs)
                stacked_imgs.append(collage)
            collage = np.concatenate(stacked_imgs, axis=1)
        #print(collage.shape)
            collage = Image.fromarray(collage.astype(np.uint8))
            collage.save(save_dir + "/check_{}".format(check) +  "/rad_{}.jpg".format(rad))
       



if __name__ == '__main__':
    
    noise = 0
    step = 0.1
    save_dir = "envs3"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in range(1):
        stacked_images = []
        for j in range(2):
            imgs = []
            check = 21
            for i in range(2):
                rad = [140, 140][i]
                noise = 0.1
                try:
                    img = get_img(1, check, noise, rad, 1)
                    print(rad, noise)
                except TimeoutError:
                    print("{}: {} - Skipped".format(rad, noise))
                    img = np.random.randint(0, 255, [400, 600, 3])
                        # print(img)
                imgs.append(img)

            stacked_images.append(np.concatenate(imgs))


        collage = np.concatenate(stacked_images, axis=1)
        collage = Image.fromarray(collage.astype(np.uint8))
        collage.save(save_dir + "/test1.jpg".format(noise))
        noise += step