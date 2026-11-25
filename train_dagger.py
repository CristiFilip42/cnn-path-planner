import time
import random
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from HelperFunctions import load_map_sample
from more_helper_functions import evaluate_model
from path import pathCNN

DATA_DIR = Path('/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset')
START_MODEL = DATA_DIR / 'models' / 'path_cnn_rl_2026-11-19 04:43:44_89.6%.pt'
DAGGER_FILES = [DATA_DIR / 'train_DAgger_after_expert' / 'dagger_iter0.pt']
BC_DIR = DATA_DIR / 'train'
train_log_file = DATA_DIR / 'training_log' / 'dagger_log.txt'

BC_MAPS = 250
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 1e-6
EVAL_LIMIT = 1000

directionz = {(-1, 0): 0, (-1, 1): 1, (0, 1): 2, (1, 1): 3, (1, 0): 4, (1, -1): 5, (0, -1): 6, (-1, -1): 7}


def bc_episodes(files):
    episodes = []
    for f in files:
        grid, start, goal, path = load_map_sample(f)
        p = [(int(r), int(c)) for r, c in path]
        good = [[directionz[(b[0] - a[0], b[1] - a[1])]] for a, b in zip(p[:-1], p[1:])]
        episodes.append({'grid': grid, 'goal': (int(goal[0]), int(goal[1])), 'positions': p[:-1], 'good': good})
    return episodes


class StepData(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes
        self.index = [(e, t) for e, ep in enumerate(episodes) for t in range(len(ep['positions']))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        ep = self.episodes[e]
        x = torch.zeros(4, 100, 100)
        x[0] = ep['grid'].float()
        r, c = ep['positions'][t]
        x[1, r, c] = 1
        x[2, ep['goal'][0], ep['goal'][1]] = 1
        for pr, pc in ep['positions'][:t]:
            x[3, pr, pc] = 1
        mask = torch.zeros(8, dtype=torch.bool)
        mask[ep['good'][t]] = True
        return x, mask


def any_good_loss(logits, mask):
    log_probs = F.log_softmax(logits, dim=1)
    return -log_probs.masked_fill(~mask, float('-inf')).logsumexp(dim=1).mean()


def log(text):
    print(text)
    with train_log_file.open('a') as f:
        f.write(text + '\n')


def pct_of(eval_log):
    return float(eval_log.split('(')[-1].rstrip(')').rstrip('%'))


def main():
    device = 'mps'
    save_dir = DATA_DIR / 'models'

    dagger = []
    for f in DAGGER_FILES:
        dagger += torch.load(f, weights_only=False)
    bc = bc_episodes(random.sample(sorted(BC_DIR.glob('*.pt')), BC_MAPS))

    log(f'\n{str(datetime.now())[:19]}')
    log(f'start_model = {START_MODEL.name}\nbc_maps = {BC_MAPS}\nbatch_size = {BATCH_SIZE}\n'
        f'learning_rate = {LEARNING_RATE}\nepochs = {EPOCHS}')
    log(f'dagger states: {sum(len(e["positions"]) for e in dagger)}, '
        f'bc states: {sum(len(e["positions"]) for e in bc)}')

    loader = DataLoader(StepData(dagger + bc), batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, persistent_workers=True)

    model = pathCNN().to(device)
    model.load_state_dict(torch.load(START_MODEL, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    baseline = evaluate_model(model, DATA_DIR / 'test', device, limit=EVAL_LIMIT)
    log(f'baseline: {baseline}')
    best = pct_of(baseline)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, correct, total = 0.0, 0, 0
        t_epoch = time.time()
        for x, mask in loader:
            x, mask = x.to(device), mask.to(device)
            out = model(x)
            loss = any_good_loss(out, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            correct += mask[torch.arange(len(x), device=device), out.argmax(dim=1)].sum().item()
            total += len(x)

        log(f'epoch {epoch} done, {time.time() - t_epoch:.0f}s, '
            f'loss {running / len(loader):.4f}, accuracy {100 * correct / total:.1f}%')

        eval_log = evaluate_model(model, DATA_DIR / 'test', device, limit=EVAL_LIMIT)
        log(eval_log)
        pct = pct_of(eval_log)
        if pct > best:
            best = pct
            now = str(datetime.now())[:19]
            torch.save(model.state_dict(), save_dir / f'path_cnn_dagger_{now}_{pct}%.pt')
            log(f'new best {pct}%, saved')


if __name__ == '__main__':
    main()