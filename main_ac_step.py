import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from pynput import keyboard
from model import Agent, Critic, AgentGRU, CriticGRU
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
GAMMA = 0.99
INPUT_NORM = True
RANDOMIZE = False
REWARD_NORM = False
GRADIENT_CLIP = True


def stack_image(stacked_image, new_image):
    new_image = new_image.squeeze(0)
    if(stacked_image is None):
        stacked_image = torch.stack([new_image.clone(), new_image.clone(), new_image.clone()], dim=-1)
    else:
        stacked_image[:, :, 0] = stacked_image[:, :, 1].clone()
        stacked_image[:, :, 1] = stacked_image[:, :, 2].clone()
        stacked_image[:, :, 2] = new_image.clone()
    return stacked_image



def compute_advantage(rewards, Qval, gamma, normalize=True):

    advantages = torch.zeros_like(rewards)
    
    for t in reversed(range(len(rewards))):
        Qval = rewards[t] + gamma * Qval
        advantages[t] = Qval
    
    if normalize:
        mean = torch.mean(advantages)
        std = torch.std(advantages)
        std = std if std > 0 else 1
        advantages = (advantages-mean)/std
    
    return advantages


def value_loss(advantages):

    return torch.mean(0.5 * advantages.pow(2))



def orthogonal_init(layer):
    if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(layer.weight)
        layer.bias.data.fill_(0.0)

if __name__ == '__main__':
    """
    Training with advantage actor-critic algorithm
    """
    torch.autograd.set_detect_anomaly(True)
    save_path = "./model_a2c_step0"
    env = CarRacing(continuous=False, domain_randomize=False, train_randomize=False)
    agent = Agent(in_channels=3, n_actions=5, input_dims=[80, 96], random_state_init=False).double()
    critic = Critic().double()
    agent.train()
    critic.train()
    if os.path.exists(save_path + "_agent"):
        agent.load_state_dict(torch.load(save_path+"_agent"))
        print("Agent loaded!")
    else:
        agent.apply(orthogonal_init)
    if os.path.exists(save_path + "_critic"):
        critic.load_state_dict(torch.load(save_path+"_critic"))
        print("Critic Loaded!")
    else:
        critic.apply(orthogonal_init)

    stacked_image = None
    optim_agent = torch.optim.Adam(agent.parameters(), lr = 0.000001)
    optim_critic = torch.optim.Adam(critic.parameters(), lr = 0.000001)

    
    for episode in range(15000):
        observation, info = env.reset()
        agent.reset_state()
        critic.reset_state()
        score = 0
        I = 1
        for t in range(50):
            observation, reward, terminated, truncated, info = env.step(0)
        env.inactive_mult = 0
        observation = observation[:80]
        observation = torch.tensor(observation).double()
        if STACK:
            observation = rgb2gray(observation)

            stacked_image = stack_image(stacked_image, observation).double()
            if INPUT_NORM:
                stacked_image = stack_image(stacked_image, observation).double()/255
            else:
                stacked_image = stack_image(stacked_image, observation).double()
            
            observation = stacked_image
        else:
            if INPUT_NORM:
                observation = observation/255
            #out = agent(observation.clone())

        for t in range(1000):
            
            if STACK:
                out = agent(stacked_image)
            else:
                out = agent(observation.clone())
            action_dist = torch.distributions.Categorical(out)
            action = action_dist.sample()
            #action = torch.argmax(out)
            if np.random.uniform(low = 0, high = 1) > 0.999:
                print(out, action)
                print(critic(observation))
           

            new_observation, reward, terminated, truncated, info = env.step(action.item())
            new_observation = new_observation[:80]
            new_observation = torch.tensor(new_observation).double()
            score += reward
            #print(t, reward)
            if STACK:
                new_observation = rgb2gray(new_observation)
                #plt.imshow(observation, cmap='gray', vmin=0, vmax=1)
                #plt.show()
                stacked_image = stack_image(stacked_image, new_observation).double()
                if INPUT_NORM:
                    stacked_image = stack_image(stacked_image, new_observation).double()/255
                else:
                    stacked_image = stack_image(stacked_image, new_observation).double()
                v_next = critic(stacked_image)[0]
               # print(stacked_image)
            else:
                if INPUT_NORM:
                    new_observation = new_observation / 255
                v_next = critic(new_observation)[0]
            
            if terminated or truncated or t == 999:
                v_next = torch.tensor([0]).double()
            v = critic(observation.clone())[0]

            critic_loss = F.mse_loss(reward + GAMMA*v_next, v)
            #critic_loss =  critic_loss * I
            
            delta = reward + GAMMA * v_next.item() - v.item()
            policy_loss = -torch.log(out[0, action.item()])
            policy_loss = policy_loss * delta
           # policy_loss = policy_loss * I


            optim_agent.zero_grad()
            policy_loss.backward(retain_graph = True)
            if GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                
            optim_agent.step()

            optim_critic.zero_grad()
            critic_loss.backward(retain_graph = True)
            if GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optim_critic.step()

            I *= GAMMA
            observation = new_observation.clone()

            if terminated or truncated:
                break
        print("Episode {}: score: {}".format(episode, score))
        torch.save(agent.state_dict(), save_path+"_agent")
        torch.save(critic.state_dict(), save_path+"_critic")