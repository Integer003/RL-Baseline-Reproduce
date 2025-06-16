import metaworld
import gymnasium as gym

env = gym.make('Meta-World/MT1', env_name='reach-V3')
obs, info = env.reset()
print("✅ Installed and imported successfully!")