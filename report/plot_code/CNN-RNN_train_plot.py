from matplotlib import pyplot as plt


label_x1 = [2, 4, 6, 8, 10]
labek_x2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# dev_y = [82.52, 83.40, 84.37, 83.84, 83.65]
# plt.plot(label_x1, dev_y, label="AlexNet", color="#FF0000", marker=".", linestyle="-")

# dev_y2 = [81.60, 84.34, 85.24, 27.11, 85.74]
# plt.plot(label_x1, dev_y2, label="ResNet-18", color="#00FFFF", marker=".", linestyle="-")

# dev_y3 = [68.70, 69.87, 69.25, 71.56, 4.01]
# plt.plot(label_x1, dev_y3, label="Sketch-a-Net", color="#0000FF", marker=".", linestyle="-")

py_dev_y = [54.07, 80.83, 83.15, 84.35, 85.11,	85.59, 85.99, 86.33, 86.61,	85.67]
plt.plot(labek_x2, py_dev_y, label="CNNRNN_001", color="#0000FF", marker=".", linestyle="-")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training of CNN-RNN")

plt.legend()

plt.show()