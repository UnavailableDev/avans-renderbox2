import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyopencl as cl

from simulator import Simulator
from load import load_map
from config import *
from gen_wingshape import generate_wing_cross_section


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

# # Setup graph
# fig, ax = plt.subplots()

# simu = Simulator(map_data, press0, ax)

# # Rendering
# ani = FuncAnimation(fig, simu.update_sim, frames=1000, interval=15, blit=False)
# # im = ax.imshow(ani)

# plt.show()


import pygame
import colorsys
# --- Initialize Pygame ---
SCALER = 1
pygame.init()
screen = pygame.display.set_mode((WIDTH * SCALER, HEIGHT * SCALER)) 
pygame.display.set_caption("Eulerian Fluid Simulator")
clock = pygame.time.Clock()

def render_density_field(density):
    field = np.clip(density*128 +128, 0, 255).astype(np.uint8)
    rgb = np.stack([field]*3, axis=-1)  # shape: (H, W, 3)
    return rgb

def render_velocity_field(velocity):
    field = np.clip(velocity*128, 0, 255).astype(np.uint8)
    rgb = np.stack([field]*3, axis=-1)  # shape: (H, W, 3)
    return rgb

def velocity_to_rgb(velocity):
    """Convert a 2D velocity vector field to an RGB image using HSV coloring."""
    vx = velocity[..., 0]
    vy = velocity[..., 1]
    angle = np.arctan2(vy, vx)  # [-π, π]
    magnitude = np.sqrt(vx**2 + vy**2)
    norm_mag = np.clip(magnitude / magnitude.max(), -1.0, 1.0)

    hue = (angle + np.pi) / (2 * np.pi)  # [0,1]
    sat = np.ones_like(hue)
    val = norm_mag

    # Convert HSV to RGB (vectorized)
    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = np.zeros_like(hsv)

    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            rgb[i, j] = colorsys.hsv_to_rgb(*hsv[i, j])

    rgb_uint8 = (rgb * 255).astype(np.uint8)
    return rgb_uint8

simu = Simulator(map_data, press0, None)
wing_angle = 5  # Initial angle for the wing
wing_camber = 0.05  # Camber of the wing

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                # Increase wing angle
                wing_angle += 1
            elif event.key == pygame.K_LEFT:
                # Decrease wing angle
                wing_angle -= 1

            print(f"Current wing angle: {wing_angle} degrees")
            generate_wing_cross_section(
                width=WIDTH,
                height=HEIGHT,
                wing_length=200,  # Adjusted for padding
                thickness=0.12,
                camber=0.05,
                angle_deg=-wing_angle,
                padding_x_per=0.3,  # 20% padding
                padding_y_per=0.05,  # 20% padding
                output_file=FILE_PATH
            )
            map_data = load_map(FILE_PATH)
            simu.update_map(map_data)

    # --- Run Simulation ---
    for _ in range(1):
        # Update simulation
        field = simu.update_sim(_)

    # --- Prepare Render ---
    # rgb = render_density_field(field)
    rgb = render_velocity_field(field)
    # rgb = velocity_to_rgb(field)
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))  # (W, H, 3)

    # --- Draw ---
    screen.blit(pygame.transform.scale(surf, (WIDTH * SCALER, HEIGHT * SCALER)), (0, 0))
    pygame.display.flip()

    # Cap framerate (optional)
    clock.tick(60)

pygame.quit()