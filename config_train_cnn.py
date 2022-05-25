import torch

class args:
    batch_size = 64
    val_batch_size = 64
    log_interval = 1000
    epochs = 5
    num_classes = 25
    dataroot = 'dataset/png'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = 'sketchnet'
    small_data = False