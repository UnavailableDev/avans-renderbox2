import numpy as np
import pyopencl as cl
from config import *


class Simulator:
    def __init__(self, map, plt):
        self.press = map
        self.ax = plt

        # Buffers
        self.press_buf_i = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.press)
        self.press_buf_o = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.press)


    def update_sim(self, frame):
        program.upd_pressure(queue, (num_cells,), None, self.press_buf_i, self.press_buf_o, np.int32(WIDTH), np.int32(HEIGHT), np.float32(1))

        cl.enqueue_copy(queue, self.press, self.press_buf_o).wait()

        # Push new frame
        self.ax.imshow(self.press)
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
