import numpy as np
from PIL import Image, ImageDraw
import math


def generate_wing_cross_section(
   width=300,
   height=150,
   wing_length=200,
   thickness=0.12,
   camber=0.05,
   angle_deg=5,
   padding_x_per=20,
   padding_y_per=20,
   output_file="wing_cross_section.png"
):
   """
   Generate a binary image of a wing cross-section.

   Parameters:
      width (int): Total image width (pixels)
      height (int): Total image height (pixels)
      wing_length (int): Length of wing chord (in pixels, before padding)
      thickness (float): Thickness as a fraction of chord
      camber (float): Camber as a fraction of chord
      angle_deg (float): Angle of attack (degrees)
      padding_x (int): Horizontal padding (pixels)
      padding_y (int): Vertical padding (pixels)
      output_file (str): Output image file name
   """

   padding_x = (int)(width*padding_x_per)
   padding_y = (int)(width*padding_y_per)

   # Ensure wing fits with padding
   max_wing_width = width - 2 * padding_x
   max_wing_height = height - 2 * padding_y
   chord = min(wing_length, max_wing_width)
   
   # NACA-like airfoil shape
   x = np.linspace(0, 1, chord)
   yt = 5 * thickness * (
      0.2969 * np.sqrt(x)
      - 0.1260 * x
      - 0.3516 * x**2
      + 0.2843 * x**3
      - 0.1015 * x**4
   )

   if camber == 0:
      yc = np.zeros_like(x)
      dyc_dx = np.zeros_like(x)
   else:
      m = camber
      p = 0.4  # max camber position
      yc = np.where(
         x < p,
         m / p**2 * (2 * p * x - x**2),
         m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2),
      )
      dyc_dx = np.where(
         x < p,
         2 * m / p**2 * (p - x),
         2 * m / (1 - p)**2 * (p - x),
      )

   theta = np.arctan(dyc_dx)

   xu = x - yt * np.sin(theta)
   yu = yc + yt * np.cos(theta)
   xl = x + yt * np.sin(theta)
   yl = yc - yt * np.cos(theta)

   wing_x = np.concatenate([xu, xl[::-1]])
   wing_y = np.concatenate([yu, yl[::-1]])

   # Normalize and scale to desired chord
   wing_x = wing_x * chord
   wing_y = wing_y * chord  # scale height relative to chord

   # Centering and padding
   wing_x_centered = wing_x - np.mean(wing_x)
   wing_y_centered = wing_y - np.mean(wing_y)

   # Apply rotation
   angle_rad = math.radians(angle_deg)
   cos_a = np.cos(angle_rad)
   sin_a = np.sin(angle_rad)

   x_rot = cos_a * wing_x_centered - sin_a * wing_y_centered
   y_rot = sin_a * wing_x_centered + cos_a * wing_y_centered

   # Shift to image space
   x_img = x_rot + width / 2
   y_img = -y_rot + height / 2  # flip y to image coordinates

   # Clamp to padding region
   x_img = np.clip(x_img, padding_x, width - padding_x)
   y_img = np.clip(y_img, padding_y, height - padding_y)

   # Draw binary image
   image = Image.new("1", (width, height), 0)
   draw = ImageDraw.Draw(image)
   draw.polygon(list(zip(x_img, y_img)), fill=1)
   image.save(output_file)
   print(f"Wing cross-section saved to '{output_file}'.")


# === Example Configuration ===
if __name__ == "__main__":
   generate_wing_cross_section(
      width=500,
      height=300,
      wing_length=300,
      thickness=0.12,
      camber=0.06,
      angle_deg=-12,
      padding_x_per=0.30,
      padding_y_per=0.05,
      output_file="wing_with_padding.png"
   )
