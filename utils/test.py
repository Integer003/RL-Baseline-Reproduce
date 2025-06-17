import os
os.environ["MUJOCO_GL"] = "egl"  # Force GPU rendering

from dm_control import _render

# New way to check available renderers
available_backends = _render.get_backends()  # Returns list of available backends
print("Available render backends:", available_backends)

# Check if EGL (GPU) is available
use_gpu = "egl" in available_backends
print("Can use GPU (EGL):", use_gpu)