import numpy as np
import pyopencl as cl
import matplotlib as mpl
from config import *

import time

class Simulator:
    def __init__(self, map, press, plt):
        self.map_data = map
        self.WIDTH = map.shape[1]
        self.HEIGHT = map.shape[0]
        self.press = press
        self.ax = plt

        self.ax.imshow(self.press, vmin=0, vmax=50) #show init state

        # Buffers
        self.map_buff = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.map_data)
        self.press_buf_i = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.press)
        self.press_buf_o = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.press)
        print(self.WIDTH, self.HEIGHT)

    def update_sim(self, frame):
        # Push last frame
        self.ax.imshow(self.press, vmin=0, vmax=50)

        program.upd_pressure(queue, (self.WIDTH,self.HEIGHT), None, self.press_buf_i, self.press_buf_o, self.map_buff, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(1))
        cl.enqueue_copy(queue, self.press_buf_i, self.press_buf_o).wait()
        cl.enqueue_copy(queue, self.press, self.press_buf_i) # Copy to update frame buffer

        print (frame, sum(sum(self.press)))
        return 


# OpenCL
with open(KERN, 'r') as f: #Open Kernel file
    kernel_code = f.read()

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Buffer init
mf = cl.mem_flags


# Compile kernel
program = cl.Program(context, kernel_code).build()
