# quickdraw-classifier
CS420 Project


## Data

Unzip the data into `dataset/seq` folder.

Use `RPCL-pix2seq` to covert the seq data into png data.

```cmd
cd RPCL-pix2seq
python seq2png.py --input_dir=../dataset/seq --output_dir=../dataset/png --png_width=28 --categories={'bear'}
```

> **NOTICE** You need to use python<=3.7 to install tensorflow 1.15.