# Defines
# Physics:
TIME_STEP = 0.001
R_BETA = 0.24  # Relaxation factor for pressure solver 0 < x < 0.25
ALPHA = 1.225  # Density of air at sea level in kg/m^3

# Space
DISPLAY_MODE = 2  # 0: Density, 1: Divergence, 2: Velocity
FLOW_SPEED = 40.0  # Speed of the flow in pixel/s

# FILE_PATH = "output.txt"
FILE_PATH = "wing_with_padding.png"
KERN = 'sims/gpt_flow.cl'
