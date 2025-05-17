# patchvino-inspection

## Introduction
Anomalib provides a benchmarking tool to evaluate the performance of the anomaly detection models on a given dataset. 
Our custom anomalib library is located at src/anomalib.
We have added PatchVino to the anomalib library under the directory src/anomalib/models/image/patchvino.

## Installation
Set up a virtual environment using PyCharm or Conda. 
Download the GitHub code and install the libraries required for our project.
```bash
git clone http://github.com/wbzeng/patchvino-inspection.git
cd patchvino-inspection
pip install -r requirements.txt  
```

## Usage of our anomalib library

Install the anomalib library version 1.2.0 in the virtual environment, and then replace the anomalib library in the virtual environment with our anomalib library (Our anomalib library is located at src/anomalib).

## Training
```bash
python train_mvtec.py
```

## Transfer model format from .ckpt to .pt for testing
```bash
python transfer_model.py
```

## Testing. Calculate the Pixel-level AUROC, Pixel-level F1 score, Image-level AUROC, and Image-level F1 score metrics for the MVTec dataset.
```bash
python test_mvtec.py
```