import gymnasium as gym
import torch.nn as nn
import torch 
import os
import matplotlib.pyplot as plt
import numpy as np

from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
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
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),  # 32 
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
    """
    Initiate env
    :param env_id: (int) id of environment from 0 to 31
    """
    right = False
    if env_id >= 16:
        right = True # 0 to 15 conter clockwise direction, 16 to 31 clockwise
        env_id -= 16
    
    check = [2,3,4,5,6,7,8,19][env_id//2] # number of checkpoints 2 - 2 turn trace, 3 - triangle, 4- rectangle etc.
    rad = [70, 140][env_id%2] # even small road, odd big roads
    
    if(check == 3):
        noise = 0.8
    elif(check == 4):
        noise = 0.5
    elif(check == 7):
        noise = 0.2
    else:
        noise = 0.1
    def init():
        if np.random.random() > 1: # generate instead train trase a random trase
            env = CarRacing(custom_continuous = False, train_randomize=True, continuous=True, noise = noise, rad = rad, checkpoints = 21, right = False, random_trase = True)
        else:
            env = CarRacing(custom_continuous = False, train_randomize=True, continuous=True, noise = noise, rad = rad, checkpoints = check, right = right)
        env.reset()
        return env
    return init


if __name__ == '__main__':
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128), # 64
        n_lstm_layers=1,
        lstm_hidden_size = 64, # 16
        enable_critic_lstm = False,
        shared_lstm = True,
        ortho_init = False,
        share_features_extractor = True,
    )

    model_dir = 'model_ppo_stable_envs_Cont_big_rand_max'
    logsdir = 'logs_Cont_big_rand_max'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(logsdir):
        os.mkdir(logsdir)
    if not os.path.exists(logsdir + "/tensorboard"):
        os.mkdir(logsdir + "/tensorboard")

    env = SubprocVecEnv([init_env(i, True) for i in range(32)]) # Wrap our envs
  
   
    
    t = 185_000_000
    # Init model
    if os.path.exists(model_dir + "/ppo_{}.zip".format(t)):
        model = RecurrentPPO.load(model_dir + "/ppo_{}".format(t), env = env, device='cuda:3', custom_objects = {"learning_rate":5e-5, "ent_coef":0.01, "target_kl":0.02, "n_steps":2048})
        print("Model loaded!")
    else:
        model = RecurrentPPO(RecurrentActorCriticCnnPolicy, env, verbose=1,  batch_size = 128, n_steps=2048, normalize_advantage=True,
            max_grad_norm=0.5, policy_kwargs=policy_kwargs, n_epochs = 10,
            learning_rate=1e-4, device= "cpu" if not torch.has_cuda else "cuda:1", target_kl = 0.03, tensorboard_log = logsdir + "/tensorboard")

    TIMESTEPS = 500_000
    
    # Load previous logs it exists
    if os.path.exists(logsdir + "/envs.log.npy"):
        logs = np.load(logsdir+"/envs.log.npy").tolist()
        print("Logs loaded!")
    else:
        logs = []
    if os.path.exists(logsdir + "/mean.log.npy"): 
        logs_mean = np.load(logsdir+"/mean.log.npy").tolist()
        print("Logs mean loaded!")
    else:
        logs_mean = []
    if os.path.exists(logsdir + "/percent.log.npy"):
        logs_perc = np.load(logsdir+"/percent.log.npy").tolist()
        print("Logs perc loaded!")
    else:
        logs_perc = []

    for it in range(1, 151):
        
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True, log_interval=6)
        t += TIMESTEPS
        model.save("{}/ppo_{}".format(model_dir, t))
        perc_per_check = {2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 19:[]}
        score_per_check = {2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 19:[]}
        print("Evaluation...")
        for env in range(0, 32):
                    
            test_env = init_env(env, True)()
            model_test = RecurrentPPO.load(model_dir + "/ppo_{}".format(t), env = test_env, device="cuda:3")
            score = 0
            obs, info = test_env.reset()
            done = False
            lstm_states = None
            num_envs = 1
            episode_starts = np.ones((num_envs,),dtype=bool)
            for step in range(2048):
                action, lstm_states = model_test.predict(obs, deterministic=False, state=lstm_states, episode_start=episode_starts)
                obs, reward, term, trunc, info = test_env.step(action)
                score += reward
                done = term or trunc
                episode_start = np.zeros((num_envs,),dtype=bool)
                if done:
                     break
            if env >= 16:
                check = [2,3,4,5,6,7,8,19][(env-16)//2]
            else:
                check = [2,3,4,5,6,7,8,19][env//2]
            perc_per_check[check].append(test_env.tile_visited_count/len(test_env.track))
            score_per_check[check].append(score)

        perc_tmp = []
        score_tmp = []
        for check in [2,3,4,5,6,7,8,19]:
            perc_tmp.append(sum(perc_per_check[check])/len(perc_per_check[check]))
            score_tmp.append(sum(score_per_check[check])/len(score_per_check[check]))

        logs.append(score_tmp)
        logs_perc.append(perc_tmp)
        logs_mean.append(sum(score_tmp)/len(score_tmp))


        #plot test results
        fig, ax = plt.subplots(nrows=1,ncols=1)
        logs_np = np.array(logs)
        np.save(logsdir + "/envs.log", logs_np)
        np.save(logsdir + "/mean.log", np.array(logs_mean))
        for j in range(8):
            ax.plot(range(len(logs)), logs_np[:, j], color = COLORS[j], label="env_id-{}".format(j))
        ax.legend()
        fig.savefig("plot_log_envs_cont_big_rand_max")
        plt.close(fig)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        logs_perc_np = np.array(logs_perc)
        np.save(logsdir + "/percent.log", logs_perc_np)
        for j in range(8):
            ax.plot(range(len(logs_perc)), logs_perc_np[:,j], color = COLORS[j], label="env_id-{}".format(j))
        ax.legend()
        fig.savefig("plot_log_perc_envs_cont_big_rand_max")
        plt.close(fig)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(range(len(logs_mean)), logs_mean)
        fig.savefig("plot_log_mean_cont_big_rand_max")
        plt.close(fig)



        


        """
        model_test = RecurrentPPO.load("{}/ppo_{}".format(model_dir, TIMESTEPS*i), env=env_test)
        scores = []
        for ep in range(5):
            score = 0
            obs, info = env_test.reset()
            done = False
            lstm_states = None
            num_envs = 1
            # Episode start signals are used to reset the lstm states
            episode_starts = np.ones((num_envs,), dtype=bool)
            for step in range(1000):
                action, lstm_states = model_test.predict(obs, deterministic=True, state=lstm_states, episode_start=episode_starts)
                #print(obs.shape)
                obs, reward, term, trunc, info = env_test.step(action)
                score += reward
                done = term or trunc
                #env.render()
                #print(action, reward)
                if done:
                    break
            scores.append(score)
        logs.append(sum(scores)/5)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(range(len(logs)), logs)
        fig.savefig("plot_log_big")
        plt.close(fig)
        """
    '''
    for ep in range(1):
        obs = env.reset()
        done = False

        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
    
            if done:
                break
    '''
    
