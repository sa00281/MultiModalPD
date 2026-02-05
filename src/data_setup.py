import os
import numpy as np
import csv

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from transform import ToTensor, RandomTranslation, RandomRotation, Norm2D


TRANSFORM = transforms.Compose([ToTensor(), 
                                RandomTranslation(), 
                                RandomRotation(),
                                Norm2D()])

GRAPH_TRANSFORM = transforms.Compose([ToTensor(),
                                      RandomTranslation(),
                                      RandomRotation()])

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

class TimTremor(Dataset):
    def __init__(self, root, instancePath, transform=None):       
        self.root = root
        with open(os.path.join(root, instancePath), 'r') as f:
            self.samples = [line.strip('"\n').replace('\\', '/') for line in f.readlines()]
        self.transform = transform
        self.classes = list(set([sample.split(',')[-1] for sample in self.samples]))
    
    def __len__(self):
        return len(self.samples)   
    
    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)].split(',')

        landmark, graph_type, acc, label = sample[0], int(sample[1]), sample[2], int(sample[3])

        landmark = os.path.join(self.root, landmark)
        landmark = np.loadtxt(landmark, delimiter=',', dtype=np.float32)

        T, _ = landmark.shape
        landmark = landmark.reshape(T, -1, 3)

        if self.transform:
            landmark = self.transform(landmark) 

        landmark = landmark.permute(2, 0, 1)
        C, T, V = landmark.shape
        landmark = landmark.reshape(C, T, V, 1)

        acc = os.path.join(self.root, acc)
        acc = np.loadtxt(acc, delimiter=',', dtype=np.float32)

        T, _ = acc.shape
        acc = acc.reshape(T, -1, 3)

        if self.transform:
            acc = self.transform(acc)

        acc = acc.reshape(T, 3)
        
        return landmark, graph_type, acc, label


class TimTremorGraph(Dataset):
    def __init__(self, root, instancePath, transform=None):       
        self.root = root
        with open(os.path.join(root, instancePath), 'r') as f:
            self.samples = [line.strip('"\n').replace('\\', '/') for line in f.readlines()]
        self.transform = transform
        self.classes = list(set([sample.split(',')[-1] for sample in self.samples]))
    
    def __len__(self):
        return len(self.samples)   
    
    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)].split(',')

        landmark, graph_type, _, label = sample[0], int(sample[1]), sample[2], int(sample[3])

        # landmark = landmark.replace('\\', '/')

        landmark = os.path.join(self.root, landmark)
        landmark = np.loadtxt(landmark, delimiter=',', dtype=np.float32)

        T, _ = landmark.shape
        landmark = landmark.reshape(T, -1, 3)

        if self.transform:
            landmark = self.transform(landmark) 

        landmark = landmark.permute(2, 0, 1)
        C, T, V = landmark.shape
        landmark = landmark.reshape(C, T, V, 1)
        
        return landmark, graph_type, label
    

class TimTremorSensor(Dataset):
    def __init__(self, root, instancePath, transform=None):       
        self.root = root
        with open(os.path.join(root, instancePath), 'r',) as f:
            self.samples = [line.strip('"\n').replace('\\', '/') for line in f.readlines()]
        self.transform = transform
        self.classes = list(set([sample.split(',')[-1] for sample in self.samples]))

    def __len__(self):
        return len(self.samples)   
    
    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)].split(',')
        
        _, _, acc, label = sample[0], sample[1], sample[2], int(sample[3])

        acc = os.path.join(self.root, acc)
        acc = np.loadtxt(acc, delimiter=',', dtype=np.float32)

        T, _ = acc.shape
        acc = acc.reshape(T, -1, 3)

        if self.transform:
            acc = self.transform(acc)

        acc = acc.reshape(T, 3)
        
        return acc, label
    

def create_dataloader(root_dir: str,
                        instance_path: str,
                        data: str,
                        transform: transforms.Compose=TRANSFORM, 
                        batch_size: int=BATCH_SIZE, 
                        num_workers: int=NUM_WORKERS,
                        shuffle: bool=True,
                        pin_memory: bool=False):
    if data == 'full':
        data = TimTremor(root_dir, instance_path, transform)
    elif data == 'graph':
        data = TimTremorGraph(root_dir, instance_path, transform)
    elif data == 'sensor':
        data = TimTremorSensor(root_dir, instance_path, transform)

    class_names = data.classes

    dataloader = DataLoader(data,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory)  

    return dataloader, class_names