import gymnasium as gym
import torch.nn as nn
import torch 
import os
import json
import numpy as np
import keyboard
import math
import time
import matplotlib.pyplot as plt 

from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from PIL import Image
from car_racing import CarRacing

COLORS = ["aqua", "black", "blue", "blueviolet",
          "brown", "burlywood", "chocolate", "crimson",
          "cadetblue", "chartreuse", "darkblue", "darkgray",
          "darkgreen", "darkmagenta", "darkorange" ,"deeppink" ,
          "dimgrey", "fuchsia" ,"lightcoral", "maroon",
          "mediumspringgreen", "navy", "purple", "salmon",
          "sienna", "yellow" ,"thistle" ,"yellowgreen",
          "peachpuff", "orangered", "olive" ,"orange",]


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=16, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3), # out_channels = 128
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #print(observations.shape)
        return self.linear(self.cnn(observations))

def init_env(env_id):
    right = False
    if env_id >= 16:
        right = True
        env_id -= 16
    check = [2,3,4,5,6,7,8,19][env_id//2]
    #check = [19, 8, 7, 6, 6, 7, 8, 19][env_id//2]
    rad = [70, 140][env_id%2]

    if(check == 3):
        noise = 0.8
    elif(check == 4):
        noise = 0.5
    elif(check == 7):
        noise = 0.2
    else:
        noise = 0.1
    
    def init():

        env = CarRacing( full_randomize=False, determined_randomize=False,continuous=True, custom_continuous=False, noise = noise, rad = rad, checkpoints = check, right = right, car_randomize_factor=0.5)
        env.reset()
        return env
    return init


if __name__ == '__main__':
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=64), # 512
        n_lstm_layers=1,
        lstm_hidden_size = 16, # 256
        enable_critic_lstm = False,
        shared_lstm = True,
        ortho_init = True,
        share_features_extractor = True,
    )

    model_dir = './model_ppo_stable_envs_Cont_big_bet'
    screenshot_dir = "./screen_test"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    final_screenshot = False
   
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    

    if final_screenshot and not os.path.exists(screenshot_dir):
        os.mkdir(screenshot_dir)
    


    steps = 48_000_000 
    #Load previous logs
    if os.path.exists("./rewards_true7.npy"):
        rewards = np.load("./rewards_true7.npy").tolist()
        print("Logs rew loaded!")
    else:
        rewards = []
    if os.path.exists("./percents_true7.npy"): 
        percents = np.load("./percents_true7.npy").tolist()
        print("Logs perc loaded!")
    else:
        percents = []
   
    for t in range(0, 200):
        perc_per_check = {2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 19:[]}
        reward_per_check = {2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 19:[]}
        for ep in range(4, 5):

            try:
                env = init_env(ep)()
             
                model = RecurrentPPO.load(model_dir+'/ppo_{}'.format(steps), env=env, device="cpu")
           
        
                scores = []
                for rep in range(1):
                    score = 0
                    obs, info = env.reset()
                    done = False
                    lstm_states = None
                    num_envs = 1
                # Episode start signals are used to reset the lstm states
                   
                    episode_starts = np.ones((num_envs,), dtype=bool)
                    for i in range(2048):
                        action, lstm_states = model.predict(obs, deterministic=False, state=lstm_states, episode_start=episode_starts)
                        #print(obs.shape)
                        episode_starts = np.zeros((num_envs,), dtype=bool)
                     
                        obs, reward, term, trunc, info = env.step(action)
                        score += reward
                        done = term or trunc
                      
                        #print("{} [{:.2f}, {:.2f}, {:.2f}]\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(i, *action, reward, score, env.tile_visited_count/len(env.track),
                        #  math.sqrt(env.car.hull.linearVelocity[0]**2 + env.car.hull.linearVelocity[1]**2), env.car.hull.angularVelocity, env.car.wheels[0].joint.angle, env.t, ep))
                  
                        #time.sleep(0.1)
                    
                        if done:
                            break

                    # take a screenshot of completed (or not) path
                    if final_screenshot:     
                        img = env.take_screenshot()
                        img = Image.fromarray(img[:350, :,:])
                        img.save(screenshot_dir + "/env_{}_epoch_{}.jpg".format(ep, steps))

                print("Rad: {} Check: {} Right: {}".format(env.rad, env.checkpoints, env.right))
                print(model_dir+'/ppo_{}: env:{}  Average score : {} Percent: {}'.format(steps, ep, score, env.tile_visited_count/len(env.track)))
                if ep >= 16:
                    check = [2,3,4,5,6,7,8,19][(ep-16)//2]
                else:
                    check = [2,3,4,5,6,7,8,19][ep//2]
                perc_per_check[check].append(env.tile_visited_count/len(env.track))
                reward_per_check[check].append(score)
            except KeyboardInterrupt:
                print("Skipped")
        steps += 500_000

        perc_tmp = []
        reward_tmp = []
        for check in [2,3,4,5,6,7,8,19]:
            perc_tmp.append(sum(perc_per_check[check])/len(perc_per_check[check]))
            reward_tmp.append(sum(reward_per_check[check])/len(reward_per_check[check]))
        percents.append(perc_tmp)
        rewards.append(reward_tmp)

        # plot and save results
        percents_np = np.array(percents)
        rewards_np = np.array(rewards)
        #np.save("./percents_true7", percents_np)
        #np.save("./rewards_true7", rewards_np)


        fig, ax = plt.subplots(nrows=1,ncols=1)
        for j in range(8):
            ax.plot(np.array(range(len(rewards_np))) * 500_000 + 500_000, rewards_np[:, j], color = COLORS[j], label="env_id-{}".format(j))
        ax.legend()
        #fig.savefig("plot_rewards_true7")
        plt.close(fig)


        fig, ax = plt.subplots(nrows=1, ncols=1)
        for j in range(8):
            ax.plot(np.array(range(len(percents_np))) * 500_000 + 500_000, percents_np[:,j], color = COLORS[j], label="env_id-{}".format(j))
        ax.legend()
        #fig.savefig("plot_percents_true7")
        plt.close(fig)

        
    
    
