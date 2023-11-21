import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

def apply_transforms(x, p = 1.0):
    x = x.reshape((3, 128, 128))
    
    t_pool = [
        #v2.Pad(padding=np.random.randint(1, 5)),
        #v2.Resize(size=128 - np.random.randint(1, 60)),
        #v2.RandomCrop(size=(120,120)),
        v2.AugMix(),
        v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)
        #v2.RandomResizedCrop(size = 128 - np.random.randint(1, 60))
    ]
    for aug in t_pool:
        x = aug(x)
  #  print("size transform, {}, {}".format(, x.shape))
    x = x.reshape((128,128,3))
    return x


class TripletsDataSetRandom(Dataset):
    def __init__(self, dataset_dir, group_folder_prefix, num_groups, num_samples, sample_name_prefix, device, start_group = 0, augmentation_val = None):
        self.dataset_dir = dataset_dir
        self.group_folder_prefix = group_folder_prefix
        self.num_groups = num_groups
        self.num_samples = num_samples
        self.sample_name_prefix = sample_name_prefix
        self.augmentation_val = augmentation_val
        self.groups = []
        self.start_group = start_group
        self.device = device
        self.generate_dataset()
    def generate_dataset(self):
        
        for group in tqdm(range(self.start_group, self.start_group+self.num_groups), desc = "Importing dataset..."):
            
            group_t = torch.load(os.path.join(self.dataset_dir, self.group_folder_prefix + str(group), "data.pth"))
            self.groups.append(group_t.to(self.device))


    def __len__(self):
        return self.num_groups * self.num_samples
    
    def __getitem__(self, index):
        current_group = index//self.num_samples
        current_sample = index%self.num_samples
        
        group = self.groups[current_group]
        anchor = group[current_sample]
        
        positive_sample = current_sample
        while(positive_sample == current_sample):
            positive_sample = np.random.randint(0, self.num_samples)
        
        positive = group[positive_sample]
        
        negative_group = current_group
        while(negative_group == current_group):
            negative_group = np.random.randint(0, self.num_groups)

        negative_sample = np.random.randint(0, self.num_samples)
        n_group = self.groups[negative_group]
        negative = n_group[negative_sample]

        if self.augmentation_val is not None:
            applier = v2.RandomApply(transforms=[v2.AugMix(), v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)], p=self.augmentation_val)
            anchor = applier(anchor.reshape((3,128,128))).reshape((128,128, 3))
            positive =  applier(positive.reshape((3,128,128))).reshape((128,128, 3))
            negative =  applier(negative.reshape((3,128,128))).reshape((128,128, 3))
        return [anchor, positive, negative]

class TripletsDataSetRandomLocal(Dataset):
    def __init__(self, dataset_dir, group_folder_prefix, num_groups, num_samples, sample_name_prefix, augmentation = False):
        self.dataset_dir = dataset_dir
        self.group_folder_prefix = group_folder_prefix
        self.num_groups = num_groups
        self.num_samples = num_samples
        self.sample_name_prefix = sample_name_prefix
        self.augmentation = augmentation

    def __len__(self):
        return self.num_groups * self.num_samples
    
    def __getitem__(self, index):
        current_group = index//self.num_samples
        current_sample = index%self.num_samples

        anchor = torch.tensor(np.load(os.path.join(self.dataset_dir, self.group_folder_prefix + str(current_group), self.sample_name_prefix + str(current_sample) + ".npy")), dtype=torch.float)/255
        positive_sample = current_sample
        while(positive_sample == current_sample):
            positive_sample = np.random.randint(0, self.num_samples)

        positive = torch.tensor(np.load(os.path.join(self.dataset_dir, self.group_folder_prefix + str(current_group), self.sample_name_prefix + str(positive_sample)+ ".npy")), dtype=torch.float)/255

        negative_group = current_group
        while(negative_group == current_group):
            negative_group = np.random.randint(0, self.num_groups)

        negative_sample = np.random.randint(0, self.num_samples)
        negative = torch.tensor(np.load(os.path.join(self.dataset_dir, self.group_folder_prefix + str(negative_group), self.sample_name_prefix + str(negative_sample)+ ".npy")), dtype=torch.float)/255

        if self.augmentation:
            applier = v2.RandomApply(transforms=[v2.AugMix(), v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)], p=0.5)
            anchor = applier(anchor.reshape((3,128,128))).reshape((128,128, 3))
            positive =  applier(positive.reshape((3,128,128))).reshape((128,128, 3))
            negative =  applier(negative.reshape((3,128,128))).reshape((128,128, 3))
        return [anchor, positive, negative]

class TripletsDataSetRandomCLF(Dataset):
    def __init__(self, dataset_dir, group_folder_prefix, num_groups, num_samples, sample_name_prefix, device, start_group = 0, augmentation_val = None):
        self.dataset_dir = dataset_dir
        self.group_folder_prefix = group_folder_prefix
        self.num_groups = num_groups
        self.num_samples = num_samples
        self.sample_name_prefix = sample_name_prefix
        self.augmentation_val = augmentation_val
        self.groups = []
        self.start_group = start_group
        self.device = device
        self.generate_dataset()
    def generate_dataset(self):

        for group in tqdm(range(self.start_group, self.start_group+self.num_groups), desc = "Importing dataset..."):

            group_t = torch.load(os.path.join(self.dataset_dir, self.group_folder_prefix + str(group), "data.pth"))
            Y = torch.load(os.path.join(self.dataset_dir, self.group_folder_prefix + str(group), "Y.pth"))
            self.groups.append([group_t.to(self.device), Y.to(self.device)])


    def __len__(self):
        return self.num_groups * self.num_samples

    def __getitem__(self, index):
        current_group = index//self.num_samples
        current_sample = index%self.num_samples

        group = self.groups[current_group]
        anchor = group[0][current_sample]
        anchor_y = group[1]

        positive_sample = current_sample
        while(positive_sample == current_sample):
            positive_sample = np.random.randint(0, self.num_samples)

        positive = group[0][positive_sample]
        positive_y = group[1]

        negative_group = current_group
        while(negative_group == current_group):
            negative_group = np.random.randint(0, self.num_groups)

        negative_sample = np.random.randint(0, self.num_samples)
        n_group = self.groups[negative_group]
        
        negative = n_group[0][negative_sample]
        negative_y = n_group[1]

        if self.augmentation_val is not None:
            applier = v2.RandomApply(transforms=[v2.AugMix(), v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)], p=self.augmentation_val)
            anchor = applier(anchor.reshape((3,128,128))).reshape((128,128, 3))
            positive =  applier(positive.reshape((3,128,128))).reshape((128,128, 3))
            negative =  applier(negative.reshape((3,128,128))).reshape((128,128, 3))
        return [(anchor, anchor_y), (positive,positive_y), (negative,negative_y)]



class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

class FE(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, n_input_channels: int = 3, features_dim: int = 64, dropout_rate = None, augmentation_rate = None):
        super().__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.n_input_channels = n_input_channels
        if dropout_rate is None:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=n_input_channels, out_channels=16, kernel_size=8, stride=4, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2), # out_channels = 128
                nn.ReLU(),
                nn.Flatten()
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=n_input_channels, out_channels=16, kernel_size=8, stride=4, padding=1),
                nn.ReLU(),
                #nn.Dropout2d(p=dropout_rate),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
                nn.ReLU(),
                #nn.Dropout2d(p=dropout_rate),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
                #nn.MaxPool2d(kernel_size=3, stride=3),
                nn.ReLU(),
               # nn.Dropout2d(p=dropout_rate),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2),
                #nn.MaxPool2d(kernel_size=3, stride=3), # out_channels = 128
                nn.ReLU(),
                nn.Flatten()
            )


        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(torch.rand([1, 3,128,128])).float()
            ).shape[1]
            
        if dropout_rate is None:
            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), L2NormalizationLayer())
        else:
            self.linear = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(n_flatten, features_dim), L2NormalizationLayer())
        
        if augmentation_rate is not None:
            self.augmentation_layer = v2.RandomApply(transforms=[v2.AugMix(), v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)], p=augmentation_rate)
        else:
            self.augmentation_layer = None
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
#        print(observations.shape)
        batch_size = observations.shape[0]
        observations = torch.reshape(observations, [batch_size, self.n_input_channels, 128, 128])
        if self.augmentation_layer is not None:
            observations = self.augmentation_layer(observations)
        return self.linear(self.cnn(observations))

class FE_class(nn.Module):
    def __init__(self, n_input_channels: int = 3, features_dim: int = 64, dropout_rate = None, augmentation_rate = None):
        super().__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        self.n_input_channels = n_input_channels
        self.FE = FE(n_input_channels, features_dim, dropout_rate, augmentation_rate);
        self.clf = nn.Sequential(nn.Linear(features_dim, 16), nn.ReLU(), nn.Linear(16,8), nn.Sigmoid())


    def forward(self, observations):
        features = self.FE(observations)
        classes = self.clf(features)
        return features, classes
    
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)



def euclidean_distance(x, y):
    """
    Compute Euclidean distance between two tensors.
    """
    #print(torch.pow((x-y), 2).sum(dim=1))
    return torch.pow(torch.pow((x-y), 2).sum(dim=1) + 1e-6, 0.5)

def compute_distance_matrix(anchor, positive, negative):
    """
    Compute distance matrix between anchor, positive, and negative samples.
    """
    #print(anchor.shape)
    distance_matrix = torch.zeros(anchor.size(0), 3)
    distance_matrix[:, 0] = euclidean_distance(anchor, anchor)
    #print(di)
    distance_matrix[:, 1] = euclidean_distance(anchor, positive)
    distance_matrix[:, 2] = euclidean_distance(anchor, negative)
   # print(distance_matrix)
    return distance_matrix

def batch_hard_triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Compute triplet loss using the batch hard strategy.
    """
    distance_matrix = compute_distance_matrix(anchor, positive, negative)
    hard_negative = torch.argmin(distance_matrix[:, 2])
   
    #print(distance_matrix[:, 2][hard_negative])
    loss = torch.max(torch.tensor(0.0), distance_matrix[:, 1]- (distance_matrix[:, 0] + distance_matrix[:, 2][hard_negative]) + margin)
    #print(torch.mean(loss))
    #loss += torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 2][hard_negative] + margin)
    #print(torch.max(torch.tensor(0.0), distance_matrix[:, 0][hard_negative] - distance_matrix[:, 2] + margin))
    return torch.mean(loss)

def train(model, dataloader, num_epoch, lr, log_dir="./FE_log_v0.3", model_dir="./FE_model_v0.3", start_epoch = 1):
    optim = torch.optim.Adam(model.parameters(), lr)
    loss = nn.TripletMarginLoss()
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    tensorboard_writer = SummaryWriter(log_dir)
    loss_val = torch.tensor(0.0)
    for epoch in range(start_epoch, start_epoch + num_epoch):
        batch_loss = []
        progress_bar = tqdm(dataloader, desc="Epoch: {}".format(epoch))
        for anchor, positive, negative in progress_bar:
            #print(anchor, positive, negative)       
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
    #        print(anchor_emb)
    #        print(positive_emb)
    #        print(negative_emb)
            loss_val = batch_hard_triplet_loss(anchor_emb, positive_emb, negative_emb, margin=0.2)
            #print(torch.cdist(anchor_emb, positive_emb).mean(), torch.cdist(anchor_emb, negative_emb).mean(), loss_val)
            optim.zero_grad()
            loss_val.backward()
            optim.step()
            batch_loss.append(loss_val.item())
            progress_bar.set_description("Epoch: {}, Loss: {:.3f}".format(epoch, loss_val.item()))
        #print(batch_loss)
        tensorboard_writer.add_scalar("Loss/train", np.mean(batch_loss), epoch)
        tensorboard_writer.flush()
        if epoch % 5 == 0:
            torch.save(model.state_dict(), model_dir+"/epoch_{}".format(epoch))
            print("Model_saved!")

def train_clf(model, dataloader, num_epoch, lr, log_dir="./FE_log_v0.3", model_dir="./FE_model_v0.3", start_epoch = 1, clf_val = 0.1):
    optim = torch.optim.Adam(model.parameters(), lr)
    loss_trp = nn.TripletMarginLoss()
    loss_clf = nn.CrossEntropyLoss()
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    tensorboard_writer = SummaryWriter(log_dir)
    loss_val = torch.tensor(0.0)
    for epoch in range(start_epoch, start_epoch + num_epoch):
        batch_loss = []
        progress_bar = tqdm(dataloader, desc="Epoch: {}".format(epoch))
        for anchor, positive, negative in progress_bar:
            #print(anchor, positive, negative)       
            anchor_emb, anchor_cls = model(anchor[0])
            positive_emb, positive_cls = model(positive[0])
            negative_emb, negative_cls = model(negative[0])
    #        print(anchor_emb)
    #        print(positive_emb)
    #        print(negative_emb)
            loss_trp_val = batch_hard_triplet_loss(anchor_emb, positive_emb, negative_emb, margin=0.2)
            loss_clf_val = loss_clf(anchor_cls, anchor[1])
            loss_val = loss_trp_val + clf_val * loss_clf_val
            #print(torch.cdist(anchor_emb, positive_emb).mean(), torch.cdist(anchor_emb, negative_emb).mean(), loss_val)
            optim.zero_grad()
            loss_val.backward()
            optim.step()
            batch_loss.append([loss_trp_val.item(), loss_clf_val.item(), loss_val.item()])
            progress_bar.set_description("Epoch: {}, Loss: {:.3f}".format(epoch, loss_val.item()))
        #print(batch_loss)
        tensorboard_writer.add_scalar("Loss/All", np.mean(np.array(batch_loss)[:, -1]), epoch)
        tensorboard_writer.add_scalar("Loss/Triplet", np.mean(np.array(batch_loss)[:, 0]), epoch)
        tensorboard_writer.add_scalar("Loss/Classificator", np.mean(np.array(batch_loss)[:, 1]), epoch)

        tensorboard_writer.flush()
        if epoch % 5 == 0:
            torch.save(model.state_dict(), model_dir+"/epoch_{}".format(epoch))
            print("Model_saved!")

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

if __name__ == "__main__":
    device = "cuda:0" if torch.backends.cuda.is_built() else "cpu"
    print("Device: {}".format(device))
    model_dir = "./FE_model_v2.6"
    log_dir = "./FE_log_v2.6"
    print("Model dir: {}".format(model_dir))
    model = FE_class(features_dim = 32, dropout_rate=0.01, augmentation_rate = 0.05)
    to_device(model, device)
    model.train()

    ds = TripletsDataSetRandomCLF("./dataset_trp_rot_clf", "group_", 1000, 60, "sample_", device, augmentation_val = None)
    dataloader = DeviceDataLoader(DataLoader(ds, batch_size=256, shuffle=True), device)
    model.load_state_dict(torch.load(model_dir + "/epoch_{}".format(500)))
    train_clf(model, dataloader, 500, 1e-5,log_dir, model_dir, start_epoch = 501, clf_val = 0.5)
    


