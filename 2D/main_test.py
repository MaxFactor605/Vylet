import torch
import numpy as np
import os
import matplotlib.pyplot as plt

from model import *
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



STACK = False
BATCH_SIZE = 1 
GAMMA = 0.92
INPUT_NORM = True
RANDOMIZE = False
REWARD_NORM = False
GRADIENT_CLIP = True
CONTINUOUS = False
PLOT = False
UNITE = True

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

def orthogonal_init(layer):
    if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(layer.weight)
        layer.bias.data.fill_(0.0)


def normal_dist(x, mean, std):
    return  (1/(std*np.sqrt(2 * np.pi))) * np.exp((-1/2) * np.power((x - mean)/std, 2))

if __name__ == '__main__':
    """
    Test model without training
    """
    save_path = "./model_small_786_19"
    version_suff = ''
    
    env = CarRacing(render_mode='human',continuous=CONTINUOUS, domain_randomize=False, train_randomize=False)
   
    if UNITE:
        model = ActorCritic_Small().double()

        if os.path.exists(save_path + version_suff):
            model.load_state_dict(torch.load(save_path+version_suff, map_location=torch.device("cpu")))
            print("United model loaded!")
    else:
        if CONTINUOUS:
            agent = AgentCont(in_channels=3, n_actions=5, input_dims=[80, 96], random_state_init=False).double()
        else:

            agent = AgentGRU(in_channels=3, n_actions=5, input_dims=[80, 96], random_state_init=False).double()
        critic = Critic().double()
    
        if os.path.exists(save_path + "_agent" + version_suff):
            agent.load_state_dict(torch.load(save_path+"_agent" + version_suff))
            print("Agent loaded!")
        else:
            agent.apply(orthogonal_init)
        if os.path.exists(save_path + "_critic" + version_suff):
            critic.load_state_dict(torch.load(save_path+"_critic" + version_suff))
            print("Critic Loaded!")
        else:
            critic.apply(orthogonal_init)

    if PLOT: 
        plt.figure(figsize=(9,3))
        x = np.arange(-1.5, 1.5, 0.1)
        x_ = np.arange(0, 1.5, 0.1)
 
    for episode in range(15000):
        observation, info = env.reset()
        if UNITE:
            model.reset_state()
        else:
            agent.reset_state()
            critic.reset_state()

        
        for t in range(5000):
            observation = observation[:80]
            
             
            observation = torch.tensor(observation).double()
            
            
              
           
            if INPUT_NORM:
                observation /= 255
            if UNITE:
                out, v = model(observation.clone())
            else:
                if CONTINUOUS:
                    means, stds = agent(observation.clone())
                    print(means, stds, reward)
                else:
                    out = agent(observation.clone())
                v = critic(observation)
          
                
        
            if CONTINUOUS:
                actions_ = torch.normal(means, stds)
                actions = []
                actions.append(torch.clip(actions_[0, 0], min = 0, max = 1).item())
                actions.append(torch.clip(actions_[0, 1], min = -1, max = 1).item())
                actions.append(torch.clip(actions_[0, 2], min = 0, max = 1).item())
                print(actions)
                
            else:
                action_dist = torch.distributions.Categorical(out)
                action = action_dist.sample()
                #action = torch.argmax(out)
                
            if PLOT:
                plt.clf()
                g = normal_dist(x_, means[0, 0].item(), stds[0, 0].item())
                s = normal_dist(x, means[0, 1].item(), stds[0, 1].item())
                b = normal_dist(x_, means[0, 2].item(), stds[0, 2].item())
                plt.subplot(131)
                plt.title("Gas")
                plt.plot(x_, g)
                plt.subplot(132)
                plt.title("Steer")
                plt.plot(x, s)
                plt.subplot(133)
                plt.title("Brake")
                plt.plot(x_, b)
                plt.pause(0.1)
            
            
            observation, reward, term, trunc, info = env.step(actions if CONTINUOUS else action.item())
            #print(t, reward)
            done = term or trunc
            if not CONTINUOUS:
                print(out, action.item(), v.item(), reward)
            if done:
                break
            
       

        
        
            
        
        
         


