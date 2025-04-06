import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyopencl as cl

from simulator import Simulator

# Defines
# Physics:
TIME_STEP = 0.001


# Space
WIDTH = 25
HEIGHT = WIDTH

KERN = 'sims/pressure.cl'


# OpenCL
with open(KERN, 'r') as f: #Open Kernel file
    kernel_code = f.read()

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)


# init space
num_cells = WIDTH * HEIGHT
num_u = num_cells
num_v = num_cells


#load map_dat
press0 = np.zeros((WIDTH,HEIGHT), dtype=np.float32)

for n in range(5):
    high = 30
    press0[n+10][10] = high
    press0[n+10][11] = high
    press0[n+10][12] = high
    press0[n+10][13] = high
    press0[n+10][14] = high

press1 = press0


# Buffers
mf = cl.mem_flags
press_buf_i = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=press0)
press_buf_o = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=press1)


# Compile kernel
program = cl.Program(context, kernel_code).build()

# Setup graph
fig, ax = plt.subplots()

simu = Simulator()

# Rendering
# ani = FuncAnimation(fig, simu.update_sim, frames=100, interval=50, blit=True)
im = ax.imshow(press0)
plt.show()
