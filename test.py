from datetime import datetime
from pathlib import Path

now = str(datetime.now())[:19]
train_log_file = Path("/Users/filipdinu/PycharmProjects/Proiect-SBC/Dataset/map_dataset/training_log/train_log.txt")
with train_log_file.open("a") as f:
    f.write(now + "\n")