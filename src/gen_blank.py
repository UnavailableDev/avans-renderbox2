# Script to generate a text file containing a dynamically sized x, y array of "1"s

def generate_array_file(x, y, filename="output.txt"):
   with open(filename, "w") as file:
      for i in range(y):
         file.write(" ".join(["1"] * x) + "\n")

# Example usage
x = int(input("Enter the number of columns (x): "))
y = int(input("Enter the number of rows (y): "))
generate_array_file(x, y)
print(f"File with {x}x{y} array of '1's has been generated.")