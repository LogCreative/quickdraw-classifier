# quickdraw-classifier
CS420 Project


## Data

Unzip the data into `dataset/seq` folder.

Use `RPCL-pix2seq` to covert the seq data into png data. (You may need to clone this repo by `--recursive` parameter to download the submodule.)

```cmd
cd RPCL-pix2seq
python seq2png.py --input_dir=../dataset/seq --output_dir=../dataset/png --png_width=28 --categories={'bear'}
```

> **NOTICE** You need to use python<=3.7 to install tensorflow 1.15.

## Train

Once the data is prepared, you could train the model by running python on one of the following scripts:
```
python train_cnn.py
python train_rnn.py
python train_cnnrnn.py
```
For CNN model, you may need to modify the type of the structure in `config_train_cnn.py`. The value of `model` could be `resnet18`, `ResNet`, or `sketchnet`. For RNN model, we use Bidirectional LSTM structure. For CNN-RNN model, we use Sketch-a-Net for CNN branch and BiLSTM for RNN branch.

The training process uses PyTorch. During training, the best model will be saved as `best_{model}.pth` in the root folder. The test accuracy could be viewed in [report/result.dat](report/result.dat).

## Report

Report (Chinese) could be found in [report/ML_CS420_Project_report.pdf](report/ML_CS420_Project_report.pdf).
