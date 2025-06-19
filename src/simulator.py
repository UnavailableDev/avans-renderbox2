import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib as mpl
from config import *

import time

class Simulator:
    def __init__(self, map, plt):
        ### Constants
        self.cellsize = 1.0

        ### Variables
        self.map_data = map
        self.WIDTH = map.shape[1]
        self.HEIGHT = map.shape[0]
        self.press = np.ones((self.HEIGHT,self.WIDTH), dtype=np.float32)
        self.press0 = self.press
        # self.velocity = np.random.rand(self.WIDTH, self.HEIGHT).astype(cl_array.vec.float2)
        # self.velocity = np.zeros((self.WIDTH, self.HEIGHT), dtype=cl_array.vec.float2)
        self.velocity = np.zeros((self.HEIGHT, self.WIDTH, 2), dtype=np.float32)
        self.divergence = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.float32)
        self.display = np.zeros((self.HEIGHT,self.WIDTH), dtype=np.float32)  # Display buffer for visualization
        # self.rgb_display = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                self.velocity[y][x][0] = FLOW_SPEED

        ### Buffers
        self.map_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.map_data)
        self.press_buf_i = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.press)
        self.press_buf_o = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.press)
        self.vel_buf_i = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity)
        self.vel_buf_o = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.velocity)
        self.div = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.divergence)
        self.displ_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.display)
        # self.rgb_displ_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.rgb_display)


    def update_map(self, map):
        self.map_data = map
        cl.enqueue_copy(queue, self.map_buf, self.map_data).wait()


    def update_sim(self, output_mode):

        program.advectVelocity(queue, (self.WIDTH,self.HEIGHT), None, self.vel_buf_i, self.vel_buf_o, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(0.1), np.float32(self.cellsize)).wait()
        program.applyForce(queue, (self.WIDTH,self.HEIGHT), None, self.vel_buf_o, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(FLOW_SPEED)).wait()
        program.computeDivergence(queue, (self.WIDTH,self.HEIGHT), None, self.vel_buf_o, self.div, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(self.cellsize))


        for i in range(40): # n * 2 itterations
            program.pressureJacobi(queue, (self.WIDTH,self.HEIGHT), None, self.div, self.press_buf_i, self.press_buf_o, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(self.cellsize), np.float32(R_BETA)).wait()
            # else:
            program.pressureJacobi(queue, (self.WIDTH,self.HEIGHT), None, self.div, self.press_buf_o, self.press_buf_i, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(self.cellsize), np.float32(R_BETA)).wait()
        
        # program.smoothPressure(queue, (self.WIDTH,self.HEIGHT), None, self.press_buf_i, self.press_buf_o, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(self.cellsize)).wait()        

        program.subtractPressureGradient(queue, (self.WIDTH,self.HEIGHT), None, self.press_buf_o, self.vel_buf_o, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(self.cellsize))
        cl.enqueue_copy(queue, self.press_buf_i, self.press_buf_o)
        cl.enqueue_copy(queue, self.vel_buf_i, self.vel_buf_o).wait()
        program.computeDivergence(queue, (self.WIDTH,self.HEIGHT), None, self.vel_buf_o, self.div, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(self.cellsize))

        if output_mode == 0:
            cl.enqueue_copy(queue, self.display, self.press_buf_i).wait()
        elif output_mode == 1:
            program.computeDivergence(queue, (self.WIDTH,self.HEIGHT), None, self.vel_buf_o, self.div, self.map_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT), np.float32(self.cellsize))
            cl.enqueue_copy(queue, self.display, self.div).wait()
        elif output_mode == 2:
            program.abs_velocity(queue, (self.WIDTH,self.HEIGHT), None, self.vel_buf_i, self.displ_buf, np.int32(self.WIDTH), np.int32(self.HEIGHT)).wait()
            cl.enqueue_copy(queue, self.display, self.displ_buf).wait()

        return self.display


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
