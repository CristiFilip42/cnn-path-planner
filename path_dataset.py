import torch
import random
from pathlib import Path
from HelperFunctions import load_map_sample, tensor_from_map_values


def load_split(folder, limit=None):
    files = sorted(Path(folder).glob("*.pt"))
    if limit is not None:
        files = files[:limit]
    n = len(files)
    maps = torch.empty(n, 100, 100, dtype=torch.uint8)
    goals = torch.empty(n, 2, dtype=torch.int64)
    paths = []
    for i, f in enumerate(files):
        grid, start, goal, path = load_map_sample(f)
        maps[i] = grid
        goals[i] = goal
        paths.append(path)
    return maps, goals, paths


class PathDataset(torch.utils.data.Dataset):
    def __init__(self, maps, goals, paths):
        self.maps = maps
        self.goals = goals
        self.paths = paths

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, i):
        k = random.randint(0, len(self.paths[i]) - 2)
        return tensor_from_map_values(self.maps[i], self.goals[i], self.paths[i], k)