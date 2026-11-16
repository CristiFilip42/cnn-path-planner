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


DATA_DIR = Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset")
train_log_file = Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset/training_log/train_log.txt")
test_files = sorted((DATA_DIR / "test").glob("*.pt"))
TRAIN_LIMIT = 50000
BATCH_SIZE = 128
LEARNING_RATE = 2e-3
learning_end = 1e-5
EPOCHS = 30
PRINT_EVERY = 200

with train_log_file.open("a") as f:
    f.write(f"train_limit = {TRAIN_LIMIT}\nbatch_size = {BATCH_SIZE}\nlearning_rate = {LEARNING_RATE}\nlearning_end = {learning_end}"
            f"\nepoch = {EPOCHS}\n" + "\n")

def main():
    device = "mps"
    train_set = PathDataset(*load_split(DATA_DIR / "train", TRAIN_LIMIT))
    save_dir = DATA_DIR / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"if you see this the data loaded succesfully")

    now = str(datetime.now())[:19]
    with train_log_file.open("a") as f:
        f.write(now + "\n")
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    model = pathCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
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
        epoch_log = str((f"epoch {epoch} done, {time.time() - t_epoch:.0f}s, "
              f"loss {running / len(train_loader):.4f}, accuracy {100 * correct / total:.1f}%"))
        print(epoch_log)
        with train_log_file.open("a") as f:
            f.write(epoch_log + "\n")

        scheduler.step()
        if(epoch > 10 and epoch % 10 == 0):
            maps_completed_log = (
                evaluate_model(model, DATA_DIR / "test", device, limit=1000))
            print(maps_completed_log)
            with train_log_file.open("a") as f:
                f.write(maps_completed_log + "\n")
            now = str(datetime.now())[:19]
            pct = maps_completed_log.split("(")[-1].rstrip(")")
            torch.save(model.state_dict(), save_dir / f"path_cnn_{now}_{pct}.pt")



if __name__ == "__main__":
    main()