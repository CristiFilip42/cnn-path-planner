import zipfile2
import torch

directionz = {(-1, 0): 0, (-1, 1): 1, (0, 1): 2, (1, 1): 3, (1, 0): 4, (1, -1): 5, (0, -1): 6, (-1, -1): 7}
# x up down, y left right

def load_map_sample(path):
    with zipfile2.ZipFile(path) as z:
        prefix = z.namelist()[0].split('/')[0]
        read = lambda k: bytearray(z.read(f"{prefix}/data/{k}"))

        grid = torch.frombuffer(read(0), dtype=torch.float32).reshape(100, 100).to(torch.uint8)
        start = torch.frombuffer(read(1), dtype=torch.int64)
        goal = torch.frombuffer(read(2), dtype=torch.int64)
        path = torch.frombuffer(read(3), dtype=torch.int64).reshape(-1, 2)
        return grid, start, goal, path

def show_map(grid):
    grid = grid.to(int)
    grid = grid.reshape(100, 100)
    for i in range(100):
        for j in range(100):
            if int(grid[i][j]) == 0:
                print(".", end = '')
            else:
                print("#", end = '')
        print('\n', end = '')

def tensor_from_map_values(grid, goal, path, k):
    pos = torch.zeros(100, 100)
    pos[path[k, 0], path[k, 1]] = 1

    goal_layer = torch.zeros(100, 100)
    goal_layer[goal[0], goal[1]] = 1

    walked = torch.zeros(100, 100)
    walked[path[:k, 0], path[:k, 1]] = 1

    x = torch.stack([grid.float(), pos, goal_layer, walked])

    move = path[k+1] - path[k]
    label = directionz[(move[0].item(), move[1].item())]
    return x, label


