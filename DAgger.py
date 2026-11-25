import math
import random
import torch
from pathlib import Path
from more_helper_functions import distance_field
from HelperFunctions import load_map_sample
from path import pathCNN

HARD_LIMIT = 200
NORMAL_LIMIT = 5000

DATA_DIR = Path('/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset')
MODEL_PATH = DATA_DIR / 'models' / 'path_cnn_rl_2026-11-19 04:43:44_89.6%.pt'
OUT_FILE = DATA_DIR / 'train_DAgger_after_expert' / 'dagger_iter0.pt'

model = pathCNN().to('mps')
model.load_state_dict(torch.load(MODEL_PATH, map_location='mps'))
hard_maps = sorted((DATA_DIR / 'train_DAgger').glob('*.pt'))[:HARD_LIMIT]
normal_maps = random.sample(sorted((DATA_DIR / 'train').glob('*.pt')), NORMAL_LIMIT)
train_direc = hard_maps + normal_maps
OUT_FILE.parent.mkdir(exist_ok=True)

directionz = {(-1, 0): 0, (-1, 1): 1, (0, 1): 2, (1, 1): 3, (1, 0): 4, (1, -1): 5, (0, -1): 6, (-1, -1): 7}
MOVES = {label: move for move, label in directionz.items()}

MAPS_WITH_EXPERT = 5200
BETA = 20
TIE_MARGIN = 1e-3

collected = []
reached_count = 0
total_states = 0
model.eval()

for i in range(MAPS_WITH_EXPERT):
    grid, start, goal, path = load_map_sample(train_direc[i])
    dist = distance_field(grid, goal)

    pos = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    x = torch.zeros(4, 100, 100)
    x[0] = grid.float()
    x[1, pos[0], pos[1]] = 1
    x[2, goal[0], goal[1]] = 1

    positions = []
    good_moves = []
    expert_steps = 0
    model_wrong = 0
    end_reason = 'step limit'

    for k in range(400):
        if pos == goal:
            end_reason = 'goal'
            break

        costs = {}
        for index in range(8):
            dr, dc = MOVES[index]
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < 100 and 0 <= nc < 100):
                continue
            step = math.sqrt(2) if dr != 0 and dc != 0 else 1
            costs[index] = step + dist[nr, nc]
        minimum = min(costs.values()) if costs else float('inf')
        if minimum == float('inf'):
            end_reason = 'boxed in'
            break
        good = [index for index, cost in costs.items() if cost <= minimum + TIE_MARGIN]

        positions.append(pos)
        good_moves.append(good)

        with torch.no_grad():
            model_label = model(x.unsqueeze(0).to('mps')).argmax().item()
        if model_label not in good:
            model_wrong += 1

        if random.randint(1, 100) <= BETA:
            dr, dc = MOVES[random.choice(good)]
            expert_steps += 1
        else:
            dr, dc = MOVES[model_label]

        x[1, pos[0], pos[1]] = 0
        x[3, pos[0], pos[1]] = 1
        pos = (pos[0] + dr, pos[1] + dc)

        if not (0 <= pos[0] < 100 and 0 <= pos[1] < 100) or grid[pos[0], pos[1]] == 1:
            end_reason = 'crash'
            break
        x[1, pos[0], pos[1]] = 1

    if pos == goal:
        end_reason = 'goal'
        reached_count += 1
    total_states += len(positions)

    print(f'map {i}: {end_reason}, steps {len(positions)}, expert drove {expert_steps}, '
          f'model wrong {model_wrong}, optimal length {dist[int(start[0]), int(start[1])]:.1f}')

    collected.append({'grid': grid, 'goal': goal, 'positions': positions, 'good': good_moves})

print(f'\nreached goal: {reached_count}/{MAPS_WITH_EXPERT} ({100 * reached_count / MAPS_WITH_EXPERT:.1f}%)')
print(f'states collected: {total_states}')

torch.save(collected, OUT_FILE)
print(f'saved to {OUT_FILE}')