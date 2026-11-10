from pathlib import Path
from HelperFunctions import load_map_sample
import shutil

files_from = sorted(Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset").rglob("*.pt"))
files_to = Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset/training_bad")

print(len(files_from))
print (len(list(files_to.glob("*.pt"))))

for files in files_from:
    grid, start, goal, path = load_map_sample(files)
    r, c = int(goal[0]), int(goal[1])
    r2, c2 = int(start[0]), int(start[1])
    if(grid[r, c] == 1 or grid[r2, c2] == 1):
        shutil.move(files, files_to)

files_from = sorted(Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset").rglob("*.pt"))
files_to = Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset/training_bad")

print (len(files_from))
print (len(list(files_to.glob("*.pt"))))
