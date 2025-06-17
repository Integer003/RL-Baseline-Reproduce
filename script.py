import threading
import os

# sim_env = "dmc"
# sim_task = "hopper_hop"
# agent = "drm"

sim_env = "mw"
sim_task = "coffee-push"
agent = "drm_mw"

cuda_id = [1]

def run_train(cuda_id, seed):
    os.system(f"CUDA_VISIBLE_DEVICES={cuda_id} python train_{sim_env}.py seed={seed} task={sim_task} agent={agent}")

threads = []

for i in range(len(cuda_id)):
    t = threading.Thread(target=run_train, args=(cuda_id[i], i + 1))
    t.start()
    threads.append(t)

for t in threads:
    t.join()