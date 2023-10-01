import argparse
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gc


from gymnasium import spaces
from env import MyEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from PIL import Image



COLORS = ["aqua", "black", "blue", "blueviolet",
          "brown", "burlywood", "chocolate", "crimson",
          "cadetblue", "chartreuse", "darkblue", "darkgray",
          "darkgreen", "darkmagenta", "darkorange" ,"deeppink" ,
          "dimgrey", "fuchsia" ,"lightcoral", "maroon",
          "mediumspringgreen", "navy", "purple", "salmon",
          "sienna", "yellow" ,"thistle" ,"yellowgreen",
          "peachpuff", "orangered", "olive" ,"orange",]

MAPS = ["round", "round-grass", "round-r", "round-grass-r", "curvy", "curvy-r", "long", "long-r", "big", "big-r",
            "zig-zag", "zig-zag-r", "plus", "plus-r", "H", "H-r"]

MAPS_DIR = "./maps"
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
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3), # out_channels = 128
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
    #map_name = ["ETH_small_loop_2_bordered", "ETH_large_loop", "ETHZ_loop_bordered", 
    #            "experiment_loop", "loop_empty", "MOOC_modcon", "ETHZ_autolab_technical_track_bordered", "zigzag_dists_bordered"][env_id]
    map_name = MAPS[env_id]
    def init():
        env = MyEnv(render_mode = None, map_file = "{}/{}.yaml".format(MAPS_DIR, map_name), max_n_steps = 2048, frame_skip = 0)
        
        return env
    return init



if __name__ == "__main__":
    model_dir = "model_big_v5"
    logs_dir = "log_big_v5"
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256), # 512
        n_lstm_layers=1,
        lstm_hidden_size = 64, # 256
        enable_critic_lstm = False,
        shared_lstm = True,
        ortho_init = False,
        share_features_extractor = True,
    )
    #logging.basicConfig(level=logging.INFO)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    if not os.path.exists(logs_dir + "/tensorboard"):
        os.mkdir(logs_dir + "/tensorboard")

    env = SubprocVecEnv([init_env(env) for env in range(16)])
    steps = 24_500_000
    if os.path.exists(model_dir + "/ppo_{}.zip".format(steps)):
        model = RecurrentPPO.load(model_dir+"/ppo_{}".format(steps), env = env, device = "cpu" if not torch.has_cuda else "cuda:1", custom_objects={"target_kl":0.02, "learning_rate":1e-5, "max_grad_norm":0.5, "n_steps":2048})
        print("ppo_{} - Model loaded".format(steps))
    else:
        
        model = RecurrentPPO(RecurrentActorCriticCnnPolicy, env, verbose=1,  batch_size = 128, n_steps=1024, normalize_advantage=True,
            max_grad_norm=0.5, policy_kwargs=policy_kwargs, n_epochs = 10, ent_coef=0.01,
            learning_rate=1e-5, device= "cpu" if not torch.has_cuda else "cuda:3", target_kl = 0.02, tensorboard_log = logs_dir + "/tensorboard", use_sde = False)#, sde_sample_freq = 64)

    TIMESTEPS = 500_000

    if os.path.exists(logs_dir + "/reward_log.npy"):
        reward_logs = np.load(logs_dir+"/reward_log.npy").tolist()
        print("Reward logs loaded")
    else:
        reward_logs = []

    if os.path.exists(logs_dir + "/step_log.npy"):
        step_logs = np.load(logs_dir+"/step_log.npy").tolist()
        print("Step logs loaded")
    else:
        step_logs = []

    

    test_env = init_env(0)()
    for epoch in range(1, 151):
        model.learn(TIMESTEPS, reset_num_timesteps=False, progress_bar=True, log_interval=6)
        steps += TIMESTEPS
        model.save(model_dir+"/ppo_{}".format(steps))
        gc.collect()
        continue
        print("Eval...")
        scores = []
        step_count = []
        test_model = RecurrentPPO.load(model_dir+"/ppo_{}".format(steps), env = test_env, device = "cpu" if not torch.has_cuda else "cuda:0")
        for env in range(16):

            #test_env = init_env(env)()
            score = 0
            obs, info = test_env.reset(map_file = "{}/{}.yaml".format(MAPS_DIR, MAPS[env]))
            done = False
            lstm_states = None
            num_envs = 1
            episode_starts = np.ones((num_envs,),dtype=bool)
            for step in range(1024):
                action, lstm_states = test_model.predict(obs.copy(), deterministic=False, state=lstm_states, episode_start=episode_starts)
                obs, reward, term, trunc, info = test_env.step(action)
                score += reward
                done = term or trunc
                episode_start = np.zeros((num_envs,),dtype=bool)
                if done:
                     break
            #test_env.close()
            scores.append(score)
            step_count.append(step)
            print("Map: {} Score: {} Steps: {}".format(MAPS[env], score, step))
        reward_logs.append(scores)
        step_logs.append(step_count)

        reward_logs_np = np.array(reward_logs)
        step_logs_np = np.array(step_logs)
        np.save(logs_dir + "/reward_log", reward_logs_np)
        np.save(logs_dir + "/step_log", step_logs_np)

        fig, ax = plt.subplots(nrows=1,ncols=1)
        for j in range(8):
            ax.plot(range(len(reward_logs_np)), reward_logs_np[:, j], color = COLORS[j], label=MAPS[j])
        ax.legend()
        fig.savefig("plot_rewards")
        plt.close(fig)


        fig, ax = plt.subplots(nrows=1, ncols=1)
        for j in range(8):
            ax.plot(range(len(step_logs_np)), step_logs_np[:,j], color = COLORS[j], label=MAPS[j])
        ax.legend()
        fig.savefig("plot_steps")
        plt.close(fig)



