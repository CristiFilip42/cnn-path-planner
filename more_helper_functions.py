from HelperFunctions import directionz
import torch
from path_dataset import load_split
import heapq
import math
import numpy as np

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

        for _ in range(400):
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

    return str((f"finished {correct}/{len(maps)} ({100 * correct / len(maps):.1f}%)"))

def distance_field(grid, goal):
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    COSTS = [1, 1, 1, 1, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]
    grid = np.asarray(grid)
    g = (int(goal[0]), int(goal[1]))

    H, W = grid.shape
    H = H-1
    W = W-1
    dist = np.full((H, W), np.inf)
    dist[g] = 0
    q = [(0.0, g)]
    while q:
        d, (r, c) = heapq.heappop(q)
        if d > dist[r, c]:
            continue
        for (dr, dc), cost in zip(MOVES, COSTS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != 1:
                nd = d + cost
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(q, (nd, (nr, nc)))
    return dist