import threading
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_train(seed):
    os.system(f"python train.py seed={seed}")

threads = []
for i in range(1, 3):
    t = threading.Thread(target=run_train, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()