import torch

class Args:
    batch_size = 64
    val_batch_size = 64
    log_interval = 1000
    epochs = 10
    num_classes = 25
    dataroot = 'dataset/png'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = 'AlexNet'
    small_data = False
    weight_decay = 1e-3
    lr = 1e-3
    def __init__(self) -> None:
        pass
    def __str__(self) -> str:
        return f"batch_size: {self.batch_size}, val_batch_size: {self.val_batch_size}, log_interval: {self.log_interval}, epochs: {self.epochs}, num_classes: {self.num_classes}, dataroot: {self.dataroot}, device: {self.device}, model: {self.model}, small_data: {self.small_data}, weight_decay: {self.weight_decay}, lr: {self.lr}"
args = Args()