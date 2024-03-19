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

    def _on_rollout_start(self) -> None:
        self.scores = np.zeros(self.num_envs)
        return True

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


class EvalCallback(BaseCallback):
    def __init__(self, timesteps, num_envs, reps, logdir):
        super().__init__()
        self.num_envs = num_envs
        self.timesteps = timesteps
        self.reps = reps
        self.logdir = logdir
        #self.label = label

    def _on_step(self):
       # print(self.model.get_env() == self.training_env)
        #print(len(self.training_env))
        return True
    
    def _on_training_end(self) -> bool:
        scores_rep = []
        completeness_rep = []
        for rep in range(self.reps):
            obs = self.training_env.reset()
            episode_starts = np.ones((self.num_envs,), dtype=bool)
            lstm_states = None
            done_signal = np.zeros((self.num_envs,), dtype=bool)
            scores = np.zeros((self.num_envs,), dtype=float)
            completeness = np.zeros((self.num_envs,), dtype=float)
            for step in range(self.timesteps):
                #print(obs.shape, info.shape)
                action, lstm_states = self.model.predict(obs.copy(), deterministic=False, state=lstm_states, episode_start=episode_starts)
                #print(action)
                self.training_env.step_async(action)
                obs, rewards, dones, info = self.training_env.step_wait()
    
                #done = term or trunc
                #print(lstm_states[0][:, :, :5])
                #print(dones)
                for i, reward in enumerate(rewards):
                    if dones[i]:
                        done_signal[i] = 1
                        if completeness[i] == 0.0:
                            completeness[i] = info[i]["completeness"]

                    
                    if not done_signal[i]: 
                        scores[i] += reward
                    #plt.imshow(obs)
                    #plt.show()
                    # print("{} [{:.3f} {:.3f} {:.3f}], {:.3f} {:.3f} {:.3f}".format(step, *action, reward, score, test_env.complete_percent))
                episode_starts = np.zeros((self.num_envs,),dtype=bool)
                if not (False in done_signal):
                        
                    break
                if step == self.timesteps - 1:
                    for i, infos in enumerate(info):
                        if completeness[i] == 0.0:
                            completeness[i] = infos["completeness"]
            scores_rep.append(scores)
            completeness_rep.append(completeness)
        #print(scores_rep)
        #print(completeness_rep)

        scores_rep = np.stack(scores_rep, axis = 1)
        print("After stack: ", scores_rep.shape)
        scores_rep = np.mean(scores_rep, axis = 1)
        print("After mean: ", scores_rep.shape)
        completeness_rep = np.stack(completeness_rep, axis = 1)
        completeness_rep = np.mean(completeness_rep, axis = 1)
        if os.path.exists(self.logdir+"/reward_log.npy") and os.path.exists(self.logdir + "/complete_log.npy"):
            reward_log = np.load(self.logdir+"/reward_log.npy")
            print("reward_log: ", reward_log.shape)
            complete_log = np.load(self.logdir+"/complete_log.npy")
            print("compelete_log: ", complete_log.shape)
            reward_log = np.concatenate([reward_log, np.expand_dims(scores_rep, axis = 0)], axis =0)
            complete_log = np.concatenate([complete_log, np.expand_dims(completeness_rep, axis = 0)], axis = 0)
        else:
            reward_log = np.expand_dims(scores_rep, axis = 0)
            complete_log = np.expand_dims(completeness, axis = 0)

        np.save(self.logdir+"/reward_log", reward_log)
        np.save(self.logdir+"/complete_log", complete_log)

        for i, score in enumerate(scores_rep):
            #print(score, completeness_rep[i], end = " ")
            
            self.logger.record("EvalReward/{}".format(MAPS[i]), score)
            self.logger.record("EvalCompleteness/{}".format(MAPS[i]), completeness_rep[i])
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
        episode_starts = np.ones((self.num_envs,), dtype=bool)
        lstm_states = None
        done = False
        
        for step in range(self.timesteps):
            #print(obs.shape, info.shape)
            action, lstm_states = self.model.predict(obs.copy(), deterministic=False, state=lstm_states, episode_start=episode_starts)
            #print(action)
            self.training_env.step_async(action)
            obs, reward, dones, info = self.training_env.step_wait()
         
            #done = term or trunc
            #print(lstm_states[0][:, :, :5])
            for i, ob in enumerate(obs):
                if dones[i]:
                    continue
                
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

