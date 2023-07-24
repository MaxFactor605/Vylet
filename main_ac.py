import gymnasium as gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pynput import keyboard
from model import Agent, Critic, AgentGRU
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

def policy_loss(log_probs, advantages):
    w_probs = -log_probs * advantages
    #print(log_probs)
    #print(w_probs)
    mean = torch.mean(w_probs)
    #print(mean)
    return mean


def orthogonal_init(layer):
    if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(layer.weight)
        layer.bias.data.fill_(0.0)

if __name__ == '__main__':
    """
    Training with advantage actor-critic algorithm
    """
    save_path = "./model_a2c_gru"
    env = CarRacing(continuous=False, domain_randomize=False, train_randomize=False)
    agent = AgentGRU(in_channels=3, n_actions=5, input_dims=[80, 96], random_state_init=False).double()
    critic = Critic().double()
    if os.path.exists(save_path + "_agent_v1"):
        agent.load_state_dict(torch.load(save_path+"_agent_v1"))
        print("Agent loaded!")
    else:
        agent.apply(orthogonal_init)
    if os.path.exists(save_path + "_critic_v1"):
        critic.load_state_dict(torch.load(save_path+"_critic_v1"))
        print("Critic Loaded!")
    else:
        critic.apply(orthogonal_init)

    stacked_image = None
    optim_agent = torch.optim.Adam(agent.parameters(), lr = 0.000001)
    optim_critic = torch.optim.Adam(critic.parameters(), lr = 0.000001)

    accum_rewards = []
    batch_disc_reward = []
    batch_states = []
    batch_actions = []
    batch_counter = 0
    
    for episode in range(15000):
        reward_memory = []
        action_memory = []
        state_memory = []
        observation, info = env.reset()
        agent.reset_state()
        
        for t in range(50):
            observation, reward, terminated, truncated, info = env.step(0)
        env.inactive_mult = 0
        for t in range(500):
            observation = observation[:80]
            
            observation = torch.tensor(observation).double()
            
            if STACK:
                observation = rgb2gray(observation)
                #plt.imshow(observation, cmap='gray', vmin=0, vmax=1)
                #plt.show()
                stacked_image = stack_image(stacked_image, observation).double()
                if INPUT_NORM:
                    stacked_image = stack_image(stacked_image, observation).double()/255
                else:
                    stacked_image = stack_image(stacked_image, observation).double()
                out = agent(stacked_image.clone())
               # print(stacked_image)
                state_memory.append(stacked_image)
            else:
                if INPUT_NORM:
                    observation /= 255
                out = agent(observation.clone())
                state_memory.append(observation)
        
            action_dist = torch.distributions.Categorical(out)
            action = action_dist.sample()
            #action = torch.argmax(out)
            action_memory.append(action.item())
            if np.random.uniform(low = 0, high = 1) > 0.999:
                print(out, action)
                print(critic(observation))
           

            observation, reward, terminated, truncated, info = env.step(action.item())
            #print(t, reward)
            reward_memory.append(reward)
            
            if terminated or truncated:
                break
        observation = observation[:80] 
        observation = torch.tensor(observation).double()
            
        if STACK:
            observation = rgb2gray(observation)
            #plt.imshow(observation, cmap='gray', vmin=0, vmax=1)
            #plt.show()
            stacked_image = stack_image(stacked_image, observation).double()
            if INPUT_NORM:
                stacked_image = stack_image(stacked_image, observation).double()/255
            else:
                stacked_image = stack_image(stacked_image, observation).double()
            Qval = critic(stacked_image)
            Qval = Qval.detach()
        else:
            if INPUT_NORM:
                observation /= 255
            Qval = critic(observation)
            Qval = Qval.detach()

        batch_states.extend(state_memory)
        batch_actions.extend(action_memory)
        batch_disc_reward.extend(compute_advantage(torch.FloatTensor(reward_memory), Qval, GAMMA, False))
        batch_counter += 1

        accum_rewards.append(sum(reward_memory))
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(range(len(accum_rewards)), accum_rewards)
        fig.savefig("plot_batch")
        plt.close(fig)
        print("Episode {}: Total reward: {}, last reward: {}".format(episode, sum(reward_memory), reward_memory[-1]))

        if batch_counter == BATCH_SIZE:
            optim_agent.zero_grad()
            optim_critic.zero_grad()

            agent.reset_state()
            states_tensor = torch.stack(batch_states)
            reward_tensor = torch.FloatTensor(batch_disc_reward)

            action_tensor = torch.LongTensor(batch_actions)
            probs = agent(states_tensor)

            probs = probs[torch.arange(probs.size(0)), action_tensor]
            
            
            log_probs = torch.log(probs)
            
            values = critic(states_tensor)
            
            reward_tensor = reward_tensor - values

            loss_agent = policy_loss(log_probs, reward_tensor)
            
            
            loss_critic = value_loss(reward_tensor)

            loss = loss_agent + loss_critic
            loss.backward()
            if GRADIENT_CLIP:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.5)
            optim_critic.step()
            optim_agent.step()

            print("Batch completed! loss_agent {}, loss_critic {}, loss {}".format(loss_agent, loss_critic, loss))

            batch_states = []
            batch_actions = []
            batch_disc_reward = []
            batch_counter = 0

            torch.save(agent.state_dict(), save_path + "_agent_v2")
            torch.save(critic.state_dict(), save_path + "_critic_v2")
            print("Model saved!")
            
        
        
         


