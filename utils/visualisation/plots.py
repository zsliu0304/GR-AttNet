import matplotlib.pyplot as plt
import csv

def plot_data(time, actual, reference, x_label, y_label, title):
    """
    Plot actual and reference data over time.
    :param time: List of time values
    :param actual: List of actual values
    :param reference: List of reference values
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :param title: Title of the plot
    """
    plt.plot(time, actual, 'b', label='Actual')
    plt.plot(time, reference, 'r', label='Reference')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

# Initialize lists to store data
time = []
theta1, theta1ref = [], []
theta2, theta2ref = [], []
d, d_ref = [], []

# Read data from CSV file
csv_file_path = 'your_path'
with open(csv_file_path, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        time.append(float(row[0]))
        theta1.append(float(row[1]))
        theta1ref.append(float(row[2]))
        theta2.append(float(row[3]))
        theta2ref.append(float(row[4]))
        d.append(float(row[5]))
        d_ref.append(float(row[6]))

# Plot the data
plot_data(time, theta1, theta1ref, 'Time', 'Theta1', 'Theta1 vs Time')
plot_data(time, theta2, theta2ref, 'Time', 'Theta2', 'Theta2 vs Time')
plot_data(time, d, d_ref, 'Time', 'Distance', 'Distance vs Time')