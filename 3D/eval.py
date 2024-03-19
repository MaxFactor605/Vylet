import argparse
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2


from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy
from PIL import Image

from env import MyEnv


COLORS = ["aqua", "black", "blue", "blueviolet",
          "brown", "burlywood", "chocolate", "crimson",
          "cadetblue", "chartreuse", "darkblue", "darkgray",
          "darkgreen", "darkmagenta", "darkorange" ,"deeppink" ,
          "dimgrey", "fuchsia" ,"lightcoral", "maroon",
          "mediumspringgreen", "navy", "purple", "salmon",
          "sienna", "yellow" ,"thistle" ,"yellowgreen",
          "peachpuff", "orangered", "olive" ,"orange",]

RECORD_VIDEO = True

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
        print(n_input_channels)
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
    map_name = ["round", "round-grass", "round-r", "round-grass-r", "curvy", "curvy-r", "long", "long-r", "big", "big-r",
            "zig-zag", "zig-zag-r", "plus", "plus-r", "H", "H-r"]
    def init():
        env = MyEnv(render_mode =  "human", view_mode = "back-follow", map_file = "./maps/{}.yaml".format(map_name[env_id]), frame_skip=0, max_n_steps=4096, light_rand= False)
    
        return env
    return init



if __name__ == "__main__":
    model_dir = "model_big_v5"
    logs_dir = "log_big_v5"
    video_dir = "video_local"
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
    #logging.basicConfig(level=logging.INFO)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    if not os.path.exists(logs_dir + "/tensorboard"):
        os.mkdir(logs_dir + "/tensorboard")
    if RECORD_VIDEO and not os.path.exists(video_dir):
        os.mkdir(video_dir)
    #env = init_env(0)()
    steps = 114_000_000
    #if os.path.exists(model_dir + "/ppo_{}.zip".format(steps)):
    #    model = RecurrentPPO.load(model_dir+"/ppo_{}".format(steps), env = env, device = "cpu" if not torch.has_cuda else "cuda:3")
    #    print("Model loaded")
    #else:
        
    #    model = RecurrentPPO(RecurrentActorCriticCnnPolicy, env, verbose=1,  batch_size = 8, n_steps=1024, normalize_advantage=True,
    #        max_grad_norm=0.5, policy_kwargs=policy_kwargs, n_epochs = 10, ent_coef=0.01,
    #        learning_rate=1e-4, device= "cpu" if not torch.has_cuda else "cuda:3", target_kl = 0.03, tensorboard_log = logs_dir + "/tensorboard")

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

    if os.path.exists(logs_dir + "/tiles_log.npy"):
        tiles_logs = np.load(logs_dir+"/tiles_log.npy").tolist()
        print("Tiles logs loaded")
    else:
        tiles_logs = []
    
    map_name = ["round", "round-grass", "round-r", "round-grass-r", "curvy", "curvy-r", "long", "long-r", "big", "big-r",
            "zig-zag", "zig-zag-r", "plus", "plus-r", "H", "H-r"]
    
    print("Eval...")
    test_env = init_env(0)()
    
    for s in range(150):
        scores = []
        step_count = []
        tiles_visited = []
        if RECORD_VIDEO and not os.path.exists("{}/ppo_{}".format(video_dir, steps)):
            os.mkdir("{}/ppo_{}".format(video_dir, steps))

        for env in range(16):
            if RECORD_VIDEO:
                out = cv2.VideoWriter('{}/ppo_{}/{}.mp4'.format(video_dir, steps, map_name[env]), cv2.VideoWriter_fourcc(*'mp4v'), 25, (128, 128))

            #test_env = init_env(env)()
            #if os.path.exists(model_dir+"/ppo_{}.zip".format(steps)):
            #    test_model = RecurrentPPO.load(model_dir+"/ppo_{}".format(steps), env = test_env, device = "cpu" if not torch.has_cuda else "cuda:3")
            #else:
            #    test_model = RecurrentPPO(RecurrentActorCriticCnnPolicy, test_env, verbose=1,  batch_size = 8, n_steps=1024, normalize_advantage=True,
            #                            max_grad_norm=0.5, policy_kwargs=policy_kwargs, n_epochs = 10, ent_coef=0.01,
            #                            learning_rate=1e-4, device= "cpu" if not torch.has_cuda else "cuda:3", target_kl = 0.03, tensorboard_log = logs_dir + "/tensorboard")
            
            test_model = RecurrentPPO.load(model_dir+"/ppo_{}".format(steps), env = test_env, device = "cpu" if not torch.has_cuda else "cuda:3")#, custom_objects={"use_sde":False})
            score = 0
            obs, info = test_env.reset(map_file = "./maps/{}.yaml".format(map_name[env]))
            img = Image.fromarray(obs)
            img.save("env11_{}.jpg".format(env))
            done = False
            lstm_states = None
            num_envs = 1
            episode_starts = np.ones((num_envs,),dtype=bool)
            try:
                for step in range(4096):
                    action, lstm_states = test_model.predict(obs.copy(), deterministic=False, state=lstm_states, episode_start=episode_starts)
                    obs, reward, term, trunc, info = test_env.step(action)
                    done = term or trunc
                    #print(lstm_states[0][:, :, :5])
                    score += reward
                    if RECORD_VIDEO:
                        obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                        out.write(obs_bgr)
                        #plt.imshow(obs)
                        #plt.show()
                   # print("{} [{:.3f} {:.3f} {:.3f}], {:.3f} {:.3f} {:.3f}".format(step, *action, reward, score, test_env.complete_percent))
                    episode_starts = np.zeros((num_envs,),dtype=bool)
                    if done:
                        
                        break
                    
                    #test_env.render(mode = "human")
            except KeyboardInterrupt:
                print("Skipped")
                
                continue
           # test_env.render(mode = "human", close = True)
            #test_env.close()
            print("Map: {}\tModel: {}\n\tReward: {}\n\tSteps: {}\n\tTiles: {}".format(map_name[env],"ppo_{}".format(steps), score, step, test_env.complete_percent))
            scores.append(score)
            step_count.append(step)
            tiles_visited.append(test_env.complete_percent)
            if RECORD_VIDEO:
                out.release()
        reward_logs.append(scores)
        step_logs.append(step_count)
        tiles_logs.append(tiles_visited)
        steps += 500_000
        continue
        reward_logs_np = np.array(reward_logs)
        step_logs_np = np.array(step_logs)
        tiles_logs_np = np.array(tiles_logs)

        np.save(logs_dir + "/reward_log", reward_logs_np)
        np.save(logs_dir + "/step_log", step_logs_np)
        np.save(logs_dir + "/tiles_log", tiles_logs_np)

        fig, ax = plt.subplots(nrows=1,ncols=1)
        for j in range(16):
            ax.plot(range(len(reward_logs_np)), reward_logs_np[:, j], color = COLORS[j], label=map_name[j])
        ax.legend()
        fig.savefig("plot_rewards_true_big_v4")
        plt.close(fig)


        fig, ax = plt.subplots(nrows=1, ncols=1)
        for j in range(16):
            ax.plot(range(len(step_logs_np)), step_logs_np[:,j], color = COLORS[j], label=map_name[j])
        ax.legend()
        fig.savefig("plot_steps_true_big_v4")
        plt.close(fig)


        fig, ax = plt.subplots(nrows=1, ncols=1)
        for j in range(16):
            ax.plot(range(len(tiles_logs_np)), tiles_logs_np[:,j], color = COLORS[j], label=map_name[j])
        ax.legend()
        fig.savefig("plot_tiles_true_big_v4")
        plt.close(fig)
        
        
