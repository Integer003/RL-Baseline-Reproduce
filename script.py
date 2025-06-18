import threading
import os

sim_env = "dmc"
sim_task = "dog_stand"


cuda_id = [1,2,3,4,5]
agents = ["drqv2", "drm", "mentor", "taco", "cp3er"]

def run_train(cuda_id, seed, agent_name):
    os.system(f"CUDA_VISIBLE_DEVICES={cuda_id} python train_{sim_env}.py seed={seed} task={sim_task} agent={agent_name}")

threads = []

for i in range(len(cuda_id)):
    t = threading.Thread(target=run_train, args=(cuda_id[i], i + 1, agents[i]))
    t.start()
    threads.append(t)

for t in threads:
    t.join()