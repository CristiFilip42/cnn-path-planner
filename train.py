import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from path_dataset import load_split, PathDataset
from more_helper_functions import evaluate_model
from path import pathCNN
from datetime import datetime

now = str(datetime.now())[:19]
train_log_file = Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset/training_log/train_log.txt")
with train_log_file.open("a") as f:
    f.write(now + "\n")


DATA_DIR = Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset")
test_files = sorted((DATA_DIR / "test").glob("*.pt"))
TRAIN_LIMIT = 48000
BATCH_SIZE = 128
LEARNING_RATE = 1.3e-3
learning_end = 1e-5
EPOCHS = 100
PRINT_EVERY = 200

def main():
    device = "mps"
    train_set = PathDataset(*load_split(DATA_DIR / "train", TRAIN_LIMIT))

    print(f"if you see this the data loaded succesfully")

    now = str(datetime.now())[:19]
    print(now)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    model = pathCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=learning_end / LEARNING_RATE, total_iters=EPOCHS
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, correct, total = 0.0, 0, 0
        t_epoch = time.time()
        for b, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            correct += (out.argmax(dim=1) == y).sum().item()
            total += len(y)
        print(f"epoch {epoch} done, {time.time() - t_epoch:.0f}s, "
              f"loss {running / len(train_loader):.4f}, accuracy {100 * correct / total:.1f}%")
        scheduler.step()
        if(epoch > 10 and epoch % 10 == 0):
            evaluate_model(model, DATA_DIR / "test", device, limit=100)


if __name__ == "__main__":
    main()