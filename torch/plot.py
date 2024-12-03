import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialize lists to hold the x and y data
x_data = []
y_data = []

# Create a figure and an axis
fig, ax = plt.subplots()

# Initialize an empty line
(line,) = ax.plot([], [], lw=2)


# Function to initialize the plot
def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(-10, 10)
    line.set_data(x_data, y_data)
    return (line,)


# Function to update the plot
def update(frame):
    # Read new data from stdin
    input_line = sys.stdin.readline()
    if input_line:
        number = float(input_line.strip())
        x_data.append(len(x_data))  # Increment x to act as index
        y_data.append(number)

        # Update xlim and ylim dynamically
        ax.set_xlim(
            0, len(x_data)
        )  # Adjust x axis to the current number of data points
        ax.set_ylim(min(y_data) - 1, max(y_data) + 1)  # Adjust y axis to fit new data

        line.set_data(x_data, y_data)

    return (line,)


# Create the animation
ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=100)

# Show the plot
plt.show()
