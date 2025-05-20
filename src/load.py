import numpy as np
from PIL import Image


def load_map(file_path):
   """
   Reads a text file and generates a 2D list from its values.
   Values are separated by spaces and newlines.
   """
   if file_path.endswith('.txt'):
      with open(file_path, 'r') as file:
         return np.array([line.split() for line in file.readlines()], dtype=np.float32)
   
   
   """
   Reads a binary PNG wing image and returns a NumPy array (dtype=np.float32).

   Parameters:
      filepath (str): Path to the PNG file

   Returns:
      np.ndarray: 2D array with values 0.0 (background) and 1.0 (wing), shape (height, width)
   """
   if file_path.endswith('.png'):
      img = Image.open(file_path).convert("1")  # Ensure binary (mode '1')
      return np.array(img, dtype=np.float32)    # Convert to float32 (0.0 and 1.0)


# Example usage
if __name__ == "__main__":
   file_path = "output.txt"  # Replace with your file path
   data = load_map(file_path)
   print(data)