import gymnasium as gym
import torch
import torch.nn as nn
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
GAMMA = 0.99
INPUT_NORM = True
RANDOMIZE = False
REWARD_NORM = False
GRADIENT_CLIP = True
CONTINUOUS = False


def stack_image(stacked_image, new_image):
    new_image = new_image.squeeze(0)
    if(stacked_image is None):
        stacked_image = torch.stack([new_image.clone(), new_image.clone(), new_image.clone()], dim=-1)
    else:
        stacked_image[:, :, 0] = stacked_image[:, :, 1].clone()
        stacked_image[:, :, 1] = stacked_image[:, :, 2].clone()
        stacked_image[:, :, 2] = new_image.clone()
    return stacked_image

def compute_advantage_ac(rewards, Qval, gamma, normalize=True):

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


def compute_advantage(rewards, gamma, normalize=True):

    advantages = torch.zeros_like(rewards)
    R = 0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        advantages[t] = R
    
    if normalize:
        mean = torch.mean(advantages)
        std = torch.std(advantages)
       # std = std if std != 0 else 1
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
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.orthogonal_(layer.weight)
        layer.bias.data.fill_(0.0)



def REINFORCE(agent, env, optim, save_path='model', version_save_suff='', gamma = 0.99, stack = False, gradient_clip = True, input_norm = True,
              batch_size = 1, reward_norm = False):
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
            
            if stack:
                observation = rgb2gray(observation)
                #plt.imshow(observation, cmap='gray', vmin=0, vmax=1)
                #plt.show()
                stacked_image = stack_image(stacked_image, observation).double()
                if input_norm:
                    stacked_image = stack_image(stacked_image, observation).double()/255
                else:
                    stacked_image = stack_image(stacked_image, observation).double()
                out = agent(stacked_image.clone())
               # print(stacked_image)
                state_memory.append(stacked_image)
            else:
                if input_norm:
                    observation /= 255
                out = agent(observation.clone())
                state_memory.append(observation)
        
            action_dist = torch.distributions.Categorical(out)
            action = action_dist.sample()
            #action = torch.argmax(out)
            action_memory.append(action.item())
            if np.random.uniform(low = 0, high = 1) > 0.999:
                print("Probs\t", out, "Action: ", action.item())
        
            observation, reward, terminated, truncated, info = env.step(action.item())
            #print(t, reward)
            reward_memory.append(reward)
            
            if terminated or truncated:
                break
            
        batch_states.extend(state_memory)
        batch_actions.extend(action_memory)
        batch_disc_reward.extend(compute_advantage(torch.FloatTensor(reward_memory), gamma, reward_norm))
        batch_counter += 1

        accum_rewards.append(sum(reward_memory))
        #fig, ax = plt.subplots(nrows=1, ncols=1)
        #ax.plot(range(len(accum_rewards)), accum_rewards)
        ##fig.savefig("plot_batch")
        #plt.close(fig)
        print("Episode {}: Total reward: {}, last reward: {}".format(episode, sum(reward_memory), reward_memory[-1]))

        if batch_counter == batch_size:
            optim.zero_grad()
            agent.reset_state()
            states_tensor = torch.stack(batch_states)
            reward_tensor = torch.FloatTensor(batch_disc_reward)

            action_tensor = torch.LongTensor(batch_actions)
            #print(states_tensor.shape)
            #print(reward_tensor.shape)
            #print(action_tensor.shape)
            probs = agent(states_tensor)

            probs = probs[torch.arange(probs.size(0)), action_tensor] # Take probs of taken actions
            
            #print(probs)
            log_probs = torch.log(probs)
            #print(log_probs)
            loss = policy_loss(log_probs, reward_tensor)
            #print(loss)
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.5)

            optim.step()
            print("Batch completed! loss {}".format(loss))

            batch_states = []
            batch_actions = []
            batch_disc_reward = []
            batch_counter = 0

            torch.save(agent.state_dict(), save_path+version_save_suff)
            print("Model saved!")

def ActorCriticAlg(agent, critic, env, optim_agent, optim_critic, save_path = 'model', num_steps = 1000, gradient_clip = True, input_norm = True, stack = False,
                gamma = 0.99, batch_size = 1, version_save_suff = ''):
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
        for t in range(num_steps):
            observation = observation[:80]
            
            observation = torch.tensor(observation).double()
            
            if stack:
                observation = rgb2gray(observation)
                #plt.imshow(observation, cmap='gray', vmin=0, vmax=1)
                #plt.show()
                stacked_image = stack_image(stacked_image, observation).double()
                if input_norm:
                    stacked_image = stack_image(stacked_image, observation).double()/255
                else:
                    stacked_image = stack_image(stacked_image, observation).double()
                out = agent(stacked_image.clone())
               # print(stacked_image)
                state_memory.append(stacked_image)
            else:
                if input_norm:
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
            
        if stack:
            observation = rgb2gray(observation)
            #plt.imshow(observation, cmap='gray', vmin=0, vmax=1)
            #plt.show()
            stacked_image = stack_image(stacked_image, observation).double()
            if input_norm:
                stacked_image = stack_image(stacked_image, observation).double()/255
            else:
                stacked_image = stack_image(stacked_image, observation).double()
            Qval = critic(stacked_image)
            Qval = Qval.detach()
        else:
            if input_norm:
                observation /= 255
            Qval = critic(observation)
            Qval = Qval.detach()

        batch_states.extend(state_memory)
        batch_actions.extend(action_memory)
        batch_disc_reward.extend(compute_advantage_ac(torch.FloatTensor(reward_memory), Qval, GAMMA, False))
        batch_counter += 1

        accum_rewards.append(sum(reward_memory))
        #fig, ax = plt.subplots(nrows=1, ncols=1)
        #ax.plot(range(len(accum_rewards)), accum_rewards)
        #fig.savefig("plot_batch")
        #plt.close(fig)
        print("Episode {}: Total reward: {}, last reward: {}".format(episode, sum(reward_memory), reward_memory[-1]))

        if batch_counter == batch_size:
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
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.5)
            optim_critic.step()
            optim_agent.step()

            print("Batch completed! loss_agent {}, loss_critic {}, loss {}".format(loss_agent, loss_critic, loss))

            batch_states = []
            batch_actions = []
            batch_disc_reward = []
            batch_counter = 0

            torch.save(agent.state_dict(), save_path + "_agent" + version_save_suff)
            torch.save(critic.state_dict(), save_path + "_critic" + version_save_suff)
            print("Model saved!")

def ActorCriticStep(agent, critic, env, optim_agent, optim_critic, save_path = 'model', version_suff_save='', num_steps = 1000, continuous = False,
                    stack = False, input_norm = True, gradient_clip = True, gamma = 0.99):
    for episode in range(15000):
        observation, info = env.reset()
        agent.reset_state()
        critic.reset_state()
        score = 0
        I = 1
    
        observation = observation[:80]
        observation = torch.tensor(observation).double()
        if stack:
            observation = rgb2gray(observation)

            stacked_image = stack_image(stacked_image, observation).double()
            if input_norm:
                stacked_image = stack_image(stacked_image, observation).double()/255
            else:
                stacked_image = stack_image(stacked_image, observation).double()
            
            observation = stacked_image
        else:
            if input_norm:
                observation = observation/255
            #out = agent(observation.clone())

        for t in range(num_steps):
            
            if stack:
                if continuous:
                    means, stds = agent(stacked_image)
                else:
                    out = agent(stacked_image)
            else:
                if continuous:
                    means, stds = agent(observation.clone())
                else:
                    out = agent(observation.clone())
            
            if continuous:
                actions_ = torch.normal(means, stds).detach()
                
                actions = []
                actions.append(actions_[0, 0].item())
                actions.append(actions_[0, 1].item())
                actions.append(actions_[0, 2].item())
                if np.random.uniform(low = 0, high = 1) > 0.999:
                    print(means)
                    print(stds)
                    print(actions, actions_)
            else:
                action_dist = torch.distributions.Categorical(out)
                action = action_dist.sample()
                #action = torch.argmax(out)
                if np.random.uniform(low = 0, high = 1) > 0.999:
                    print(out, action)
                    print(critic(observation))
           

            new_observation, reward, terminated, truncated, info = env.step(actions if continuous else action.item())
            new_observation = new_observation[:80]
            new_observation = torch.tensor(new_observation).double()
            score += reward
            #print(t, reward)
            if stack:
                new_observation = rgb2gray(new_observation)
                #plt.imshow(observation, cmap='gray', vmin=0, vmax=1)
                #plt.show()
                stacked_image = stack_image(stacked_image, new_observation).double()
                if input_norm:
                    stacked_image = stack_image(stacked_image, new_observation).double()/255
                else:
                    stacked_image = stack_image(stacked_image, new_observation).double()
                v_next = critic(stacked_image)[0]
               # print(stacked_image)
            else:
                if input_norm:
                    new_observation = new_observation / 255
                v_next = critic(new_observation)[0]
            
            if terminated or truncated or t == 499:
                v_next = torch.tensor([0]).double()
            v = critic(observation.clone())[0]

            critic_loss = F.mse_loss(reward + gamma*v_next, v)
            critic_loss =  critic_loss * I
            
            delta = reward + gamma * v_next.item() - v.item()
            if continuous:
                policy_loss = -torch.log( (1/(stds * torch.sqrt(torch.Tensor([2 * torch.pi])))) * torch.exp( (-1/2) * torch.pow( (actions_ - means)/stds, 2 )) )
                policy_loss = torch.sum(policy_loss)
            else:
                policy_loss = -torch.log(out[0, action.item()])

            policy_loss = policy_loss * delta
            policy_loss = policy_loss * I

            

            optim_agent.zero_grad()
            policy_loss.backward(retain_graph = True)
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 2)
                
            optim_agent.step()

            optim_critic.zero_grad()
            critic_loss.backward(retain_graph = True)
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 2)
            optim_critic.step()

            I *= gamma
            observation = new_observation.clone()

            if terminated or truncated:
                break
        print("Episode {}: score: {} time: {}".format(episode, score, t))
        torch.save(agent.state_dict(), save_path+"_agent" + version_suff_save)
        torch.save(critic.state_dict(), save_path+"_critic" + version_suff_save)

def ActorCriticStepUnite(model, env, optim, save_path = "model", version_suff_save='', gamma = 0.99, gradient_clip = True, input_norm = True, device = "cpu",
                         num_steps = 1000, continuous = False):
    for episode in range(15000):
        observation, info = env.reset()
        model.reset_state()
        score = 0
        I = 1
       
        observation = observation[:80]
        observation = torch.tensor(observation, device=device).double()
        
        
        if INPUT_NORM:
            observation = observation/255
            

        for t in range(num_steps):
            
           
            if continuous:
                means, stds = model(observation.clone())
            else:
                out, value = model(observation.clone())
            
            if continuous:
                actions_ = torch.normal(means, stds).detach()
                
                actions = []
                actions.append(actions_[0, 0].item())
                actions.append(actions_[0, 1].item())
                actions.append(actions_[0, 2].item())
                if np.random.uniform(low = 0, high = 1) > 0.999:
                    print(means)
                    print(stds)
                    print(actions, actions_)
            else:
                action_dist = torch.distributions.Categorical(out)
                action = action_dist.sample()
                #action = torch.argmax(out)
                if np.random.uniform(low = 0, high = 1) > 0.9995:
                    print(out, action)
                    print(value)
           

            new_observation, reward, term, trunc, info = env.step(actions if continuous else action.item())
            done = term or trunc
            new_observation = new_observation[:80]
            new_observation = torch.tensor(new_observation, device=device).double()
            score += reward
            #print(t, reward)
           
            if input_norm:
                new_observation = new_observation / 255
            with torch.no_grad():
                _, v_next = model(new_observation)
               
            
            if done or t == num_steps - 1:
                v_next = torch.tensor([0], device=device).double().unsqueeze(0)
           

            critic_loss = F.mse_loss(reward + gamma*v_next, value)
            #critic_loss =  critic_loss * I
            
            delta = reward + gamma * v_next.item() - value.item()
            if continuous:
                policy_loss = -torch.log( (1/(stds * torch.sqrt(torch.Tensor([2 * torch.pi])))) * torch.exp( (-1/2) * torch.pow( (actions_ - means)/stds, 2 )) )
                policy_loss = torch.sum(policy_loss)
            else:
                policy_loss = -torch.log(out[0, action.item()])

            policy_loss = policy_loss * delta
            #policy_loss = policy_loss * I

            loss = policy_loss + 0.5 * critic_loss

            optim.zero_grad()
            loss.backward()

            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                
            optim.step()

            I *= gamma
            observation = new_observation.clone()

            if done:
                break
        print("Episode {}: score: {} time: {}".format(episode, score, t))
        torch.save(model.state_dict(), save_path+version_suff_save)

def ActorCriticStepUniteEp(model, env, optim, save_path = "model", version_suff_save='', gamma = 0.99, gradient_clip = True, input_norm = True, device = "cpu",
                         num_steps = 1000, continuous = False):
    
    
    for episode in range(15000):
        observation, info = env.reset()
        model.reset_state()
        score = 0
        I = 1
       
        observation = observation[:80]
        observation = torch.tensor(observation, device=device).double()
        
        losses = []
        if INPUT_NORM:
            observation = observation/255
            

        for t in range(num_steps):
            
           
            if continuous:
                means, stds = model(observation.clone())
            else:
                out, value = model(observation.clone())
            
            if continuous:
                actions_ = torch.normal(means, stds).detach()
                
                actions = []
                actions.append(actions_[0, 0].item())
                actions.append(actions_[0, 1].item())
                actions.append(actions_[0, 2].item())
                if np.random.uniform(low = 0, high = 1) > 0.999:
                    print(means)
                    print(stds)
                    print(actions, actions_)
            else:
                action_dist = torch.distributions.Categorical(out)
                action = action_dist.sample()
                #action = torch.argmax(out)
                if np.random.uniform(low = 0, high = 1) > 0.9995:
                    print(out, action)
                    print(value)
           

            new_observation, reward, term, trunc, info = env.step(actions if continuous else action.item())
            done = term or trunc
            new_observation = new_observation[:80]
            new_observation = torch.tensor(new_observation, device=device).double()
            score += reward
            #print(t, reward)
           
            if input_norm:
                new_observation = new_observation / 255
            with torch.no_grad():
                _, v_next = model(new_observation)
               
            
            if done or t == num_steps - 1:
                v_next = torch.tensor([0], device=device).double().unsqueeze(0)
           

            critic_loss = F.mse_loss(reward + gamma*v_next, value)
            #critic_loss =  critic_loss * I
            
            delta = reward + gamma * v_next.item() - value.item()
            if continuous:
                policy_loss = -torch.log( (1/(stds * torch.sqrt(torch.Tensor([2 * torch.pi])))) * torch.exp( (-1/2) * torch.pow( (actions_ - means)/stds, 2 )) )
                policy_loss = torch.sum(policy_loss)
            else:
                policy_loss = -torch.log(out[0, action.item()])

            policy_loss = policy_loss * delta
            #policy_loss = policy_loss * I

            loss = policy_loss + 0.5 * critic_loss
            losses.append(loss)
            #optim.zero_grad()
            #loss.backward()

            #if gradient_clip:
            #    torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                
            #optim.step()

            #I *= gamma
            observation = new_observation.clone()

            if done:
                break
        loss = sum(losses)
        optim.zero_grad()
        loss.backward()

        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optim.step()

        print("Episode {}: score: {} time: {}".format(episode, score, t))
        torch.save(model.state_dict(), save_path+version_suff_save)
        
if __name__ == '__main__':
    """
    Training with just policy gradient 
    """
    torch.autograd.set_detect_anomaly(True)
    save_path = "./model_ac_big"
    version_suff = ''
    version_suff_save = ''
    env = CarRacing(continuous=CONTINUOUS, domain_randomize=False, train_randomize=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    if CONTINUOUS:
        model = AgentCont(in_channels=3, n_actions=5, input_dims=[80, 96], random_state_init=False).double()
    else:

        model = ActorCritic_Small(in_channels=3, n_actions=5, input_dims=[80, 96], random_state_init=False).double()
    
    model.train()
    if os.path.exists(save_path+version_suff):
        model.load_state_dict(torch.load(save_path+version_suff))
        print("Model_loaded!")

    model.to(device)
    stacked_image = None
    optim_model = torch.optim.Adam(model.parameters(), lr = 1e-10)
    
    ActorCriticStepUniteEp(model, env, optim_model)
            
        
        
         

# Save internal states before each step, for batch learning 
# PPO algorithm 

