from matplotlib import pyplot as plt


label_x1 = [2, 4, 6, 8, 10]
labek_x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# dev_y = [82.52, 83.40, 84.37, 83.84, 83.65]
# plt.plot(label_x1, dev_y, label="AlexNet", color="#FF0000", marker=".", linestyle="-")

# dev_y2 = [81.60, 84.34, 85.24, 27.11, 85.74]
# plt.plot(label_x1, dev_y2, label="ResNet-18", color="#00FFFF", marker=".", linestyle="-")

# dev_y3 = [68.70, 69.87, 69.25, 71.56, 4.01]
# plt.plot(label_x1, dev_y3, label="Sketch-a-Net", color="#0000FF", marker=".", linestyle="-")

py_dev_y = [75.14, 81.11, 83.09, 83.90, 84.72,	84.85, 85.29, 85.29, 85.36,	85.61]
plt.plot(labek_x2, py_dev_y, label="RNN_0006", color="#0000FF", marker=".", linestyle="-")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training of RNN")

plt.legend()

plt.show()