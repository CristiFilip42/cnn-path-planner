from datetime import datetime
from pathlib import Path

now = str(datetime.now())[:19]
train_log_file = Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset/training_log/train_log.txt")
with train_log_file.open("a") as f:
    f.write(now + "\n")

TRAIN_LIMIT = 48000
BATCH_SIZE = 128
LEARNING_RATE = 1.3e-3
LEARNING_RATE = 1e-5
EPOCHS = 100
PRINT_EVERY = 200

with train_log_file.open("a") as f:
    f.write(f"train_limit = {TRAIN_LIMIT}\nbatch_size = {BATCH_SIZE}\nlearning_rate = {LEARNING_RATE}"
            f"\nepoch = {EPOCHS}\n" + "\n")