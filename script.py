import threading
import os

sim_env = "dmc"
sim_task = "hopper_hop"
cuda_id = [1,2,3,4]

def run_train(cuda_id, seed):
    os.system(f"CUDA_VISIBLE_DEVICES={cuda_id} python train_{sim_env}.py seed={seed} task={sim_task}")

threads = []

for i in range(len(cuda_id)):
    t = threading.Thread(target=run_train, args=(cuda_id[i], i + 1))
    t.start()
    threads.append(t)

for t in threads:
    t.join()