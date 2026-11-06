import torch
from HelperFunctions import tensor_from_map_values
import random

class PathDataset(torch.utils.data.Dataset):
    def __init__(self, maps,goals,paths):
        self.maps = maps
        self.goals = goals
        self.paths = paths

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, i):
        k = random.randint(0, len(self.paths[i] - 2))
        return tensor_from_map_values(self.maps[i], self.goals[i], self.paths[i], k)