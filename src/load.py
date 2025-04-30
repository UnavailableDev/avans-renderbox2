import numpy as np

def load_map(file_path):
   """
   Reads a text file and generates a 2D list from its values.
   Values are separated by spaces and newlines.
   """
   with open(file_path, 'r') as file:
      return np.array([line.split() for line in file.readlines()], dtype=np.float32)

# Example usage
if __name__ == "__main__":
   file_path = "output.txt"  # Replace with your file path
   data = load_map(file_path)
   print(data)