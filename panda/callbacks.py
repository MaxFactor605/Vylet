import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from typing import Any, Dict

import gymnasium as gym
import torch as th
import os
import cv2

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video



MAPS = ["round", "round-grass", "round-r", "round-grass-r", "curvy", "curvy-r", "long", "long-r", "big", "big-r",
            "zig-zag", "zig-zag-r", "plus", "plus-r", "H", "H-r"]
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, num_envs, verbose=0):
        super().__init__(verbose)
        self.scores = np.zeros(num_envs)
        self.num_envs = num_envs

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        #print(self.training_env.complete_percent)
        self.scores += self.locals["rewards"]
        for i in range(self.num_envs):
            if self.locals["dones"][i]:
                self.logger.record("completeness/{}".format(MAPS[i]), self.locals["infos"][i]["completeness"])
                self.logger.record("rewards/{}".format(MAPS[i]), self.scores[i])
                self.scores[i] = 0
            
        
      

        return True



class VideoRecorderCallback(BaseCallback):
    def __init__(self, timesteps, video_dir, num_envs):
        super().__init__()
        self.video_dir = video_dir
        self.num_envs = num_envs
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        self.timesteps = timesteps
        #self.label = label

    def _on_step(self):
       # print(self.model.get_env() == self.training_env)
        #print(len(self.training_env))
        return True
    
    def _on_training_end(self) -> bool:
        writters = [cv2.VideoWriter('{}/{}.mp4'.format(self.video_dir, MAPS[i]), cv2.VideoWriter_fourcc(*'mp4v'), 25, (128, 128)) for i in range(self.num_envs)]
        obs = self.training_env.reset()
        episode_starts = np.zeros((self.num_envs,), dtype=bool)
        lstm_states = None
        done = False
        
        for step in range(self.timesteps):
            #print(obs.shape, info.shape)
            action, lstm_states = self.model.predict(obs.copy(), deterministic=False, state=lstm_states, episode_start=episode_starts)
            #print(action)
            self.training_env.step_async(action)
            obs, reward, dones, info = self.training_env.step_wait()
            print(obs.shape)
            #done = term or trunc
            #print(lstm_states[0][:, :, :5])
            for i, ob in enumerate(obs):
                if dones[i]:
                    continue
                print(ob.shape)
                obs_bgr = cv2.cvtColor(ob.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                writters[i].write(obs_bgr)
                #plt.imshow(obs)
                #plt.show()
                # print("{} [{:.3f} {:.3f} {:.3f}], {:.3f} {:.3f} {:.3f}".format(step, *action, reward, score, test_env.complete_percent))
            episode_starts = np.zeros((self.num_envs,),dtype=bool)
            if not (False in dones):
                        
                break
        for writter in writters:
            writter.release()
        return True

