from HelperFunctions import directionz
import torch
from path_dataset import PathDataset,load_split

def evaluate_model(model, folder, device, limit=1000):
    MOVES = {label: move for move, label in directionz.items()}
    maps, goals, paths = load_split(folder, limit)
    model.eval()
    correct = 0

    for i in range(len(maps)):
        grid = maps[i]
        pos = (int(paths[i][0][0]), int(paths[i][0][1]))
        goal = (int(goals[i][0]), int(goals[i][1]))

        x = torch.zeros(4, 100, 100)
        x[0] = grid.float()
        x[1, pos[0], pos[1]] = 1
        x[2, goal[0], goal[1]] = 1

        for _ in range(200):
            if pos == goal:
                correct += 1
                break
            with torch.no_grad():
                dr, dc = MOVES[model(x.unsqueeze(0).to(device)).argmax().item()]
            x[1, pos[0], pos[1]] = 0
            x[3, pos[0], pos[1]] = 1
            pos = (pos[0] + dr, pos[1] + dc)
            if not (0 <= pos[0] < 100 and 0 <= pos[1] < 100) or grid[pos[0], pos[1]] == 1:
                break
            x[1, pos[0], pos[1]] = 1

    print(f"finished {correct}/{len(maps)} ({100 * correct / len(maps):.1f}%)")