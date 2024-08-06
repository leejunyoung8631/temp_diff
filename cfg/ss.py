import matplotlib.pyplot as plt

# Read the values from the text file
with open('loss.log', 'r') as file:
    lines = file.readlines()

# Extract loss values from the lines
loss_values = [float(line.split(': ')[1]) for line in lines]

# Plotting the loss values
plt.figure(figsize=(10, 5))
plt.plot(loss_values, linestyle='-', color='b')
plt.title('Loss Values Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig("das.png")