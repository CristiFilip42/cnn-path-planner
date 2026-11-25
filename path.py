import torch.nn as nn
import torch

# i want to create a 17x17 (car goes in too) patch around the car;

def create_patch(x, y, map, outside):
    map_padded = torch.full((116, 116), float(outside), device=map.device)
    map_padded[8:108, 8:108] = map
    map_around_car = map_padded[x:x + 17, y:y + 17]
    flattened_map = map_around_car.flatten()
    return flattened_map

def create_patches(batch):
    all_patches = []
    for sample in batch:
        car = (sample[1] == 1).nonzero()[0]
        x = int(car[0])
        y = int(car[1])

        walls = create_patch(x, y, sample[0], 1)
        goal = create_patch(x, y, sample[2], 0)
        all_patches.append(torch.cat([walls, goal]))
    return torch.stack(all_patches)


class pathCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 12 * 12, 512)
        self.fc2 = nn.Linear(512 + 2 * 17 * 17, 512)
        self.fc3 = nn.Linear(512, 8)

    def forward(self, x):
        patches = create_patches(x)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = torch.cat([x, patches], dim=1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x