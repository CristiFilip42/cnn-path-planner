print("start")

from pathlib import Path
from torch.utils.data import DataLoader

from path_dataset import load_split, PathDataset
from path import pathCNN

folder = Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset/train/")

maps, goals, paths = load_split(folder, 200)
ds = PathDataset(maps, goals, paths)
print(len(ds))
x, y = ds[0]
print(x.shape, y)
loader = DataLoader(ds, batch_size=64, shuffle=True)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)
print(pathCNN()(xb).shape)