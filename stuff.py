from HelperFunctions import load_map_sample, show_map, tensor_from_map_values
from pathlib import Path
from path_dataset import PathDataset

folder = Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset/train/")
files = sorted(folder.glob("*.pt"))



