import matplotlib.pyplot as plt
import numpy as np

categories = [
    "bear",
    "camel",
    "cat",
    "cow",
    "crocodile",
    "dog",
    "elephant",
    "flamingo",
    "giraffe",
    "hedgehog",
    "horse",
    "kangaroo",
    "lion",
    "monkey",
    "owl",
    "panda",
    "penguin",
    "pig",
    "raccoon",
    "rhinoceros",
    "sheep",
    "squirrel",
    "tiger",
    "whale",
    "zebra",
]


fig = plt.figure(figsize=(7,7))

for i,cate in enumerate(categories):
    ax = fig.add_subplot(5,5,i+1)
    ax.axis('off')
    ax.set_title(cate,y=-0.2)
    png = np.load(f"../dataset/png/{cate}_png.npz", allow_pickle=True, encoding="latin1")['train'][1]

    ax.imshow(png, cmap="gray")

plt.show()