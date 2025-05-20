import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyopencl as cl

from simulator import Simulator
from load import load_map
from config import *


# OpenCL
with open(KERN, 'r') as f: #Open Kernel file
    kernel_code = f.read()

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

#load map_dat
map_data = load_map(FILE_PATH)

# Constants
WIDTH = map_data.shape[1]
HEIGHT = map_data.shape[0]
num_cells = WIDTH * HEIGHT
num_u = num_cells
num_v = num_cells

print("Map data shape:", map_data.shape)

# init sim values
press0 = np.zeros((HEIGHT,WIDTH), dtype=np.float32)

# for n in range(5):
#     high = 30
#     press0[n+10][10] = high
#     press0[n+10][11] = high
#     press0[n+10][12] = high
#     press0[n+10][13] = high
#     press0[n+10][14] = high

# for i in range(HEIGHT):
#     press0[i][15] = 30

# for x in range(HEIGHT):
#     for y in range(WIDTH):
#         if map_data[x][y] == 1:
#             # press0[x][y] = x*y
#             press0[x][y] = np.random.randint(0, 50)

# Compile kernel
program = cl.Program(context, kernel_code).build()

# Setup graph
fig, ax = plt.subplots()

simu = Simulator(map_data, press0, ax)

# Rendering
ani = FuncAnimation(fig, simu.update_sim, frames=1000, interval=15, blit=False)
# im = ax.imshow(ani)

plt.show()
