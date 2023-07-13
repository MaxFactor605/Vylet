import gymnasium as gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pynput import keyboard
from model import Agent
from car_racing import CarRacing



def gamma_decompress(img):
    """
    Make pixel values perceptually linear.
    """
    img_lin = ((img + 0.055) / 1.055) ** 2.4
    i_low = np.where(img <= .04045)
    img_lin[i_low] = img[i_low] / 12.92
    return img_lin


def gamma_compress(img_lin):
    """
    Make pixel values display-ready.
    """
    img = 1.055 * img_lin ** (1 / 2.4) - 0.055
    i_low = np.where(img_lin <= .0031308)
    img[i_low] = 12.92 * img_lin[i_low]
    return img


def rgb2gray_linear(rgb_img):
    """
    Convert *linear* RGB values to *linear* grayscale values.
    """
    red = rgb_img[:, :, 0]
    green = rgb_img[:, :, 1]
    blue = rgb_img[:, :, 2]

    gray_img = (
        0.2126 * red
        + 0.7152 * green
        + 0.0722 * blue)

    return gray_img
def rgb2gray(rgb_img):
    """
    rgb_img is a 3-dimensional Numpy array of type float with
    values ranging between 0 and 1.
    Dimension 0 represents image rows, left to right.
    Dimension 1 represents image columns top to bottom.
    Dimension 2 has a size of 3 and
    represents color channels, red, green, and blue.

    For more on what this does and why:
    https://brohrer.github.io/convert_rgb_to_grayscale.html

    Returns a gray_img 2-dimensional Numpy array of type float.
    Values range between 0 and 1.
    """
    return gamma_compress(rgb2gray_linear(gamma_decompress(rgb_img)))



PRESSED_KEY = None
def on_press(key):
    global PRESSED_KEY
    PRESSED_KEY = key.char
    #print(key.char)
def on_release(key):
    global PRESSED_KEY
    PRESSED_KEY = None


STACK = False
BATCH_SIZE = 1
GAMMA = 0.99
MANUAL = False
def stack_image(stacked_image, new_image):
    new_image = new_image.squeeze(0)
    if(stacked_image is None):
        stacked_image = torch.stack([new_image.clone(), new_image.clone(), new_image.clone()], dim=-1)
    else:
        stacked_image[:, :, 0] = stacked_image[:, :, 1].clone()
        stacked_image[:, :, 1] = stacked_image[:, :, 2].clone()
        stacked_image[:, :, 2] = new_image.clone()
    return stacked_image



def compute_advantage(rewards, gamma, normalize=True):

    advantages = torch.zeros_like(rewards)
    R = 0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        advantages[t] = R
    
    if normalize:
        mean = torch.mean(advantages)
        std = torch.std(advantages)
        std = std if std > 0 else 1
        advantages = (advantages-mean)/std
    
    return advantages

def policy_loss(log_probs, advantages):
    w_probs = -log_probs * advantages
    #print(log_probs)
    #print(w_probs)
    #print(advantages)
    mean = torch.mean(w_probs)
    #print(mean)
    return mean


if __name__ == '__main__':
    """
    Test model without training
    """
    save_path = "./model_lstm"
    env = CarRacing(render_mode='human', continuous=False, domain_randomize=False, train_randomize=False)
    agent = Agent(in_channels=3, n_actions=5, input_dims=[80, 96], random_state_init=False).double()
    if os.path.exists(save_path):
        agent.load_state_dict(torch.load(save_path))
        print("Model loaded!")
    
    stacked_image = None
   
    
   
 
    for episode in range(15000):
        observation, info = env.reset()
        agent.reset_state()
        
        for t in range(100):
            observation, reward, terminated, truncated, info = env.step(0)
        env.inactive_mult = 0
        for t in range(1000):
            observation = observation[:80]
            
            observation = torch.tensor(observation).double()
            
            if STACK:
                observation = rgb2gray(observation)
                stacked_image = stack_image(stacked_image, observation).double()
                out = agent(stacked_image.clone())
                
            else:
                out = agent(observation.clone())
                
        
            action_dist = torch.distributions.Categorical(out)
            action = action_dist.sample()
            #action = torch.argmax(out)
           
            print(out, action)
            
            observation, reward, terminated, truncated, info = env.step(action.item())
            #print(t, reward)
            
            
            if terminated or truncated:
                break
            
       

        
        
            
        
        
         


