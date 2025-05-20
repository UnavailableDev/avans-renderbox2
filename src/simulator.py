import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib as mpl
from config import *

import time

class Simulator:
    def __init__(self, map, press, plt):
        ### Constants
        self.interflow = 50.0
        self.cellsize = 1.0

        ### Variables
        self.map_data = map
        self.WIDTH = map.shape[1]
        self.HEIGHT = map.shape[0]
        self.press = press
        self.velocity = np.random.rand(self.WIDTH, self.HEIGHT).astype(cl_array.vec.float2)
        # self.velocity = np.zeros((self.WIDTH, self.HEIGHT), dtype=cl_array.vec.float2)
        # self.velocity = np.zeros((self.WIDTH, self.HEIGHT, 2), dtype=np.float32)
        self.divergence = np.zeros((self.WIDTH, self.HEIGHT), dtype=np.float32)
        self.ax = plt


        ### Wind Tunnel Inlet (middle third of left boundary)
        # y1 = self.HEIGHT // 3
        # y2 = 2 * self.HEIGHT // 3
        # print("yy", y1, y2)
        # map[y1:y2, 0] = 0.0  # make sure it's fluid
        # self.velocity[y1*2:y2*2, 0, 0] = self.interflow  # horizontal velocity at inlet

        ### Add a solid obstacle (e.g., a vertical bar in the center)
        # obstacle_x = self.WIDTH // 2
        # obstacle_y1 = self.HEIGHT // 3
        # obstacle_y2 = 2 * self.HEIGHT // 3
        # map[obstacle_x, obstacle_y1:obstacle_y2] = 1.0

        ### Buffers
        self.map_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.map_data)
        self.press_buf_i = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.press)
        self.press_buf_o = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.press)
        self.vel_buf_i = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity)
        self.vel_buf_o = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity)
        print(self.WIDTH, self.HEIGHT)


        self.div = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.divergence)

        ### Preload graph
        self.ax.imshow(self.press, vmin=0, vmax=50) #show init state

        

    def update_sim(self, frame):
        # # Push last frame
        # if frame % 10 == 0:
        if 1:
            cl.enqueue_copy(queue, self.press, self.press_buf_i).wait() # Copy to update frame buffer
            # cl.enqueue_copy(queue, self.velocity, self.vel_buf_i).wait() # Copy to update frame buffer
            # self.ax.imshow(self.velocity, vmin=-0.5, vmax=0.5)
            self.ax.imshow(self.press)
            # self.ax.imshow(self.press, vmin=-2, vmax=2)
            print(sum(sum(self.press)))
        # self.ax.imshow(self.press)

        program.advectVelocity(queue, (self.WIDTH,self.HEIGHT), None, self.vel_buf_i, self.vel_buf_o, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(1), np.float32(1)).wait()
        program.applyBoundary(queue, (self.WIDTH,self.HEIGHT), None, self.vel_buf_o, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(self.interflow)).wait()
        program.computeDivergence(queue, (self.WIDTH,self.HEIGHT), None, self.vel_buf_o, self.div, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(self.cellsize))

        for i in range(75):
            # Swap buffers
            if i % 2 == 0:
                program.pressureJacobi(queue, (self.WIDTH,self.HEIGHT), None, self.div, self.press_buf_o, self.press_buf_i, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(1.25), np.float32(0.25)).wait()
            else:
                program.pressureJacobi(queue, (self.WIDTH,self.HEIGHT), None, self.div, self.press_buf_i, self.press_buf_o, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(1.25), np.float32(0.25))
        
            # cl.enqueue_copy(queue, self.press_buf_i, self.press_buf_o).wait()
            # program.pressureJacobi(queue, (self.WIDTH,self.HEIGHT), None, self.div, self.press_buf_i, self.press_buf_o, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(1.225), np.float32(0.25))


        # cl.enqueue_copy(queue, self.press_buf_i, self.press_buf_o).wait()
        program.subtractPressureGradient(queue, (self.WIDTH,self.HEIGHT), None, self.press_buf_o, self.vel_buf_o, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(self.cellsize))
        cl.enqueue_copy(queue, self.press_buf_i, self.press_buf_o)
        cl.enqueue_copy(queue, self.vel_buf_i, self.vel_buf_o).wait() # Copy to update frame buffer
        # print (frame, sum(sum(self.press)))
        print (frame)
        # print (frame, sum(self.velocity[0:self.WIDTH-1][0:self.HEIGHT-1][0]))
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
