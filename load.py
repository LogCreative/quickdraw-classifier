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

bear = np.load("data/sketchrnn_bear.npz", allow_pickle=True, encoding="latin1")

## show the structure of bear
# bear.files

bear_train = bear["train"]
bear_test = bear["test"]
bear_valid = bear["valid"]

# %%
print(bear_train[1].shape)
