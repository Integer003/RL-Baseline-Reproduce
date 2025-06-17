import threading
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_train_1(seed):
    os.system(f"CUDA_VISIBLE_DEVICES=6 python train.py seed=1")
def run_train_2(seed):
    os.system(f"CUDA_VISIBLE_DEVICES=6 python train.py seed=2")
def run_train_3(seed):
    os.system(f"CUDA_VISIBLE_DEVICES=7 python train.py seed=3")
def run_train_4(seed):
    os.system(f"CUDA_VISIBLE_DEVICES=7 python train.py seed=4")

threads = []

t = threading.Thread(target=run_train_1)
t.start()
threads.append(t)
t = threading.Thread(target=run_train_2)
t.start()
threads.append(t)
t = threading.Thread(target=run_train_3)
t.start()
threads.append(t)
t = threading.Thread(target=run_train_4)
t.start()
threads.append(t)


for t in threads:
    t.join()