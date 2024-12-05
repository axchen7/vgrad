import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x_data = []
y_data = []

fig, ax = plt.subplots()

scat = ax.scatter([], [])

# throw out initial input
sys.stdin.readline()


def update(frame):
    input_line = sys.stdin.readline()
    if input_line:
        # format: x,y
        x, y = map(float, input_line.split(","))
        x_data.append(x)
        y_data.append(y)
        print(input_line, end="")

        ax.set_xlim(min(x_data), max(x_data))
        ax.set_ylim(min(y_data) - 1, max(y_data) + 1)

        scat.set_offsets(list(zip(x_data, y_data)))

    return (scat,)


ani = animation.FuncAnimation(fig, update, blit=True, interval=1)
plt.show()
