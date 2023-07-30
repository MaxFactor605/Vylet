import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

class Agent(nn.Module):
    def __init__(self, in_channels=3, n_actions=5, linear_size=4*6*32, input_dims=[80, 96], random_state_init = False):
        super(Agent, self).__init__()
        self.in_channels = in_channels
        self.linear_size = linear_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.l1 = nn.Linear(linear_size, 128)
        self.l2 = nn.Linear(128, 32)
        self.l3 = nn.Linear(32, n_actions)
        self.random_state_init = random_state_init
        if random_state_init:
            self.hidden_state = torch.rand([1, 8], dtype=torch.double)
            self.cell_state = torch.rand([1, 8], dtype=torch.double)
        else:
            self.hidden_state = None
            self.cell_state = None

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, self.in_channels, self.input_dims[0], self.input_dims[1])
     
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = F.relu(x, inplace=False)
        x = self.conv3(x)
        x = F.relu(x, inplace=False)
        x = self.conv4(x)
        x = F.relu(x, inplace=False)
       
        x = x.view(-1, self.linear_size)
        x = self.l1(x)
        x = F.relu(x, inplace=False)
        x = self.l2(x)
        x = F.relu(x, inplace=False)
        x = self.l3(x)

        x = F.softmax(x, dim=-1)
        return x
    

    def reset_state(self):
        if self.random_state_init:
            self.hidden_state = torch.rand([1, 8], dtype=torch.double)
            self.cell_state = torch.rand([1, 8], dtype=torch.double)
        else:
            self.hidden_state = None
            self.cell_state = None

class AgentLSTM(nn.Module):
    def __init__(self, in_channels=3, n_actions=5, linear_size=4*6*32, input_dims=[80, 96], random_state_init = False):
        super(AgentLSTM, self).__init__()
        self.in_channels = in_channels
        self.linear_size = linear_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.l1 = nn.Linear(linear_size, 64)
        self.l2 = nn.Linear(64, 16)
        self.lstm = nn.LSTM(16, 8)
        self.l3 = nn.Linear(8, n_actions)
        self.random_state_init = random_state_init
        if random_state_init:
            self.hidden_state = torch.rand([1, 32], dtype=torch.double)
            self.cell_state = torch.rand([1, 32], dtype=torch.double)
        else:
            self.hidden_state = None
            self.cell_state = None

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, self.in_channels, self.input_dims[0], self.input_dims[1])
     
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = F.relu(x, inplace=False)
        x = self.conv3(x)
        x = F.relu(x, inplace=False)
        x = self.conv4(x)
        x = F.relu(x, inplace=False)
       
        x = x.view(-1, self.linear_size)
        x = self.l1(x)
        x = F.relu(x, inplace=False)
        x = self.l2(x)
        x = F.relu(x, inplace=False)
        if x.shape[0] > 1: # When training process all sequence at once
            
            
            x, (h_0, c_0) = self.lstm(x)
            
   

        else: # When playing process frame by frame and save state beatween
            #x = x.unsqueeze(dim = 0)
         
            if self.hidden_state is None or self.cell_state is None:
                x, (self.hidden_state, self.cell_state) = self.lstm(x)

            else:
                x, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
            
            
            x = self.hidden_state

        
        if np.random.uniform(low = 0, high = 1) > 0.999:
            
            print("Hidden_state\t", self.hidden_state)
            print("Cell_state\t", self.cell_state)
       
        x = self.l3(x)

        x = F.softmax(x, dim=-1)
        return x
    

    def reset_state(self):
        if self.random_state_init:
            self.hidden_state = torch.rand([1, 32], dtype=torch.double)
            self.cell_state = torch.rand([1, 32], dtype=torch.double)
        else:
            self.hidden_state = None
            self.cell_state = None

class AgentGRU(nn.Module):
    def __init__(self, in_channels=3, n_actions=5, linear_size=4*6*32, input_dims=[80, 96], random_state_init = False):
        super(AgentGRU, self).__init__()
        self.in_channels = in_channels
        self.linear_size = linear_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.l1 = nn.Linear(linear_size, 64)
        self.l2 = nn.Linear(64, 10)
        self.gru = nn.GRU(10, 5)
        self.l3 = nn.Linear(5, n_actions)
        #self.l1 = nn.Linear(linear_size, 128)
        #self.l2 = nn.Linear(128, 64)
        #self.gru = nn.GRU(64, 32)
        #self.l3 = nn.Linear(32, n_actions)
        self.random_state_init = random_state_init
        if random_state_init:
            self.hidden_state = torch.rand([1, 8], dtype=torch.double)
            
        else:
            self.hidden_state = None
        

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, self.in_channels, self.input_dims[0], self.input_dims[1])
     
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = F.relu(x, inplace=False)
        x = self.conv3(x)
        x = F.relu(x, inplace=False)
        x = self.conv4(x)
        x = F.relu(x, inplace=False)
       
        x = x.view(-1, self.linear_size)
        x = self.l1(x)
        x = F.relu(x, inplace=False)
        x = self.l2(x)
        x = F.relu(x, inplace=False)
        if x.shape[0] > 1: # When training process all sequence at once
            
            
            x, h_0 = self.gru(x)
            
   

        else: # When playing process frame by frame and save state beatween
            #x = x.unsqueeze(dim = 0)
         
            if self.hidden_state is None:
                x, self.hidden_state_tmp = self.gru(x)

            else:
                x, self.hidden_state_tmp = self.gru(x, self.hidden_state)
            
            self.hidden_state = self.hidden_state_tmp.detach()
            
            
            

        
        if np.random.uniform(low = 0, high = 1) > 0.999:
            
            print("Hidden_state\t", self.hidden_state)
          
           
        x = self.l3(x)

        x = F.softmax(x, dim=-1)
        return x
    

    def reset_state(self):
        if self.random_state_init:
            self.hidden_state = torch.rand([1, 32], dtype=torch.double)
        
        else:
            self.hidden_state = None
           
class AgentGRUCont(nn.Module):
    def __init__(self, in_channels=3, n_actions=5, linear_size=4*6*32, input_dims=[80, 96], random_state_init = False):
        super(AgentGRUCont, self).__init__()
        self.in_channels = in_channels
        self.linear_size = linear_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.l1 = nn.Linear(linear_size, 64)
        self.l2 = nn.Linear(64, 10)
        self.gru = nn.GRU(10, 5)
        self.mean_head = nn.Linear(5, 3)
        self.std_head = nn.Linear(5, 3)
        self.random_state_init = random_state_init
        if random_state_init:
            self.hidden_state = torch.rand([1, 8], dtype=torch.double)
            
        else:
            self.hidden_state = None
        

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, self.in_channels, self.input_dims[0], self.input_dims[1])
     
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = F.relu(x, inplace=False)
        x = self.conv3(x)
        x = F.relu(x, inplace=False)
        x = self.conv4(x)
        x = F.relu(x, inplace=False)
       
        x = x.view(-1, self.linear_size)
        x = self.l1(x)
        x = F.relu(x, inplace=False)
        x = self.l2(x)
        x = F.relu(x, inplace=False)
        if x.shape[0] > 1: # When training process all sequence at once
            
            
            x, h_0 = self.gru(x)
            
   

        else: # When playing process frame by frame and save state beatween
            #x = x.unsqueeze(dim = 0)
         
            if self.hidden_state is None:
                x, self.hidden_state_tmp = self.gru(x)

            else:
                x, self.hidden_state_tmp = self.gru(x, self.hidden_state)
            
            self.hidden_state = self.hidden_state_tmp.detach()
            
            
            

        
        if np.random.uniform(low = 0, high = 1) > 0.999:
            
            print("Hidden_state\t", self.hidden_state)
          
           
        means = self.mean_head(x)
        stds = self.std_head(x)

        stds = torch.exp(stds)
        return means, stds
    

    def reset_state(self):
        if self.random_state_init:
            self.hidden_state = torch.rand([1, 32], dtype=torch.double)
        
        else:
            self.hidden_state = None
           


class Critic (nn.Module):
    def __init__(self, in_channels = 3, input_dims = [80, 96], linear_size = 4*6*64):
        super(Critic, self).__init__()
        self.in_channels = in_channels
        self.input_dims = input_dims
        self.linear_size = linear_size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.l1 = nn.Linear(linear_size, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)



    def forward(self, x):
        x = x.view(-1, self.in_channels, self.input_dims[0], self.input_dims[1])
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = F.relu(x, inplace=False)
        x = self.conv3(x)
        x = F.relu(x, inplace=False)
        x = self.conv4(x)
        x = F.relu(x, inplace=False)
    
        x = x.view(-1, self.linear_size)
        x = self.l1(x)
        x = F.relu(x, inplace=False)
        x = self.l2(x)
        x = F.relu(x, inplace=False)
        x = self.l3(x)
        return x
    
    def reset_state(self):
        pass

class CriticGRU (nn.Module):
    def __init__(self, in_channels = 3, input_dims = [80, 96], linear_size = 4*6*64, random_state_init = False):
        super(CriticGRU, self).__init__()
        self.in_channels = in_channels
        self.input_dims = input_dims
        self.linear_size = linear_size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.l1 = nn.Linear(linear_size, 64)
        self.l2 = nn.Linear(64, 16)
        self.gru = nn.GRU(16, 8)
        self.l3 = nn.Linear(8, 1)
        self.random_state_init = random_state_init
        if random_state_init:
            self.hidden_state = torch.rand([1, 16], dtype=torch.double)
            
        else:
            self.hidden_state = None


    def forward(self, x):
        x = x.view(-1, self.in_channels, self.input_dims[0], self.input_dims[1])
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = F.relu(x, inplace=False)
        x = self.conv3(x)
        x = F.relu(x, inplace=False)
        x = self.conv4(x)
        x = F.relu(x, inplace=False)
    
        x = x.view(-1, self.linear_size)
        x = self.l1(x)
        x = F.relu(x, inplace=False)
        x = self.l2(x)
        x = F.relu(x, inplace=False)
        if x.shape[0] > 1: # When training process all sequence at once
            
            
            x, h_0 = self.gru(x)
            
   

        else: # When playing process frame by frame and save state beatween
            #x = x.unsqueeze(dim = 0)
         
            if self.hidden_state is None:
                x, self.hidden_state_tmp = self.gru(x)

            else:
                x, self.hidden_state_tmp = self.gru(x, self.hidden_state)
            
            self.hidden_state = self.hidden_state_tmp.detach()
    

        
        if np.random.uniform(low = 0, high = 1) > 0.999:
            
            print("Hidden_state\t", self.hidden_state)
          
        x = self.l3(x)
        return x
    
    def reset_state(self):
        if self.random_state_init:
            self.hidden_state = torch.rand([1, 16], dtype=torch.double)
        
        else:
            self.hidden_state = None

class AgentCont(nn.Module):
    def __init__(self, in_channels=3, n_actions=5, linear_size=4*6*32, input_dims=[80, 96], random_state_init = False):
        super(AgentCont, self).__init__()
        self.in_channels = in_channels
        self.linear_size = linear_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.l1 = nn.Linear(linear_size, 128)
        self.l2 = nn.Linear(128, 32)
        self.mean_head = nn.Linear(32, 3)
        self.std_head = nn.Linear(32, 3)
        self.random_state_init = random_state_init
        if random_state_init:
            self.hidden_state = torch.rand([1, 8], dtype=torch.double)
            self.cell_state = torch.rand([1, 8], dtype=torch.double)
        else:
            self.hidden_state = None
            self.cell_state = None

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(-1, self.in_channels, self.input_dims[0], self.input_dims[1])
     
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = F.relu(x, inplace=False)
        x = self.conv3(x)
        x = F.relu(x, inplace=False)
        x = self.conv4(x)
        x = F.relu(x, inplace=False)
       
        x = x.view(-1, self.linear_size)
        x = self.l1(x)
        x = F.relu(x, inplace=False)
        x = self.l2(x)
        x = F.relu(x, inplace=False)
        means = self.mean_head(x)
        stds =  torch.exp(self.std_head(x))
        
        return means, stds
    

    def reset_state(self):
        if self.random_state_init:
            self.hidden_state = torch.rand([1, 8], dtype=torch.double)
            self.cell_state = torch.rand([1, 8], dtype=torch.double)
        else:
            self.hidden_state = None
            self.cell_state = None





class ActorCritic(nn.Module):
    def __init__(self, in_channels=3, n_actions=5, linear_size=4*6*32, input_dims=[80, 96], random_state_init = False):
        super(ActorCritic, self).__init__()
        self.in_channels = in_channels
        self.linear_size = linear_size
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.l1 = nn.Linear(linear_size, 64)
        self.l2 = nn.Linear(64, 10)
        self.gru = nn.GRU(10, 5)
        self.actor_head = nn.Linear(5, n_actions)
        self.critic_head = nn.Linear(5, 1)
        self.random_state_init = random_state_init
        #self.l1 = nn.Linear(linear_size, 128)
        #self.l2 = nn.Linear(128, 64)
        #self.gru = nn.GRU(64, 32)
        #self.l3 = nn.Linear(32, n_actions)
        self.random_state_init = random_state_init
        if random_state_init:
            self.hidden_state = torch.rand([1, 8], dtype=torch.double)
            
        else:
            self.hidden_state = None

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.input_dims[0], self.input_dims[1])
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.conv2(x)
        x = F.relu(x, inplace=False)
        x = self.conv3(x)
        x = F.relu(x, inplace=False)
        x = self.conv4(x)
        x = F.relu(x, inplace=False)
    
        x = x.view(-1, self.linear_size)
        x = self.l1(x)
        x = F.relu(x, inplace=False)
        x = self.l2(x)
        x = F.relu(x, inplace=False)
        if x.shape[0] > 1: # When training process all sequence at once
            
            
            x, h_0 = self.gru(x)
            
   

        else: # When playing process frame by frame and save state beatween
            #x = x.unsqueeze(dim = 0)
         
            if self.hidden_state is None:
                x, self.hidden_state_tmp = self.gru(x)

            else:
                x, self.hidden_state_tmp = self.gru(x, self.hidden_state)
            
            self.hidden_state = self.hidden_state_tmp.detach()
    

        
        if np.random.uniform(low = 0, high = 1) > 0.999:
            
            print("Hidden_state\t", self.hidden_state)
          
        probs = F.softmax(self.actor_head(x), dim=-1)
        
        value = self.critic_head(x)
        return probs, value
    

    def reset_state(self):
        if self.random_state_init:
            self.hidden_state = torch.rand([1, 8], dtype=torch.double)
            self.cell_state = torch.rand([1, 8], dtype=torch.double)
        else:
            self.hidden_state = None
            self.cell_state = None