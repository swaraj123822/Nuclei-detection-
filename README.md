# Nuclei Detection using U-Net in PyTorch

This repository contains a Jupyter Notebook that implements a U-Net model for nuclei detection using the PyTorch deep learning framework. The model is trained on the 2018 Data Science Bowl dataset to identify and segment cell nuclei from microscopy images.

## Dataset

The model is trained on the [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018) dataset. This dataset contains a large number of microscopy images and their corresponding masks, which are used for training and evaluating the model. The images are preprocessed by resizing them to 128x128 pixels.

## Model Architecture

The model uses a U-Net architecture, which is a popular convolutional neural network for biomedical image segmentation. The U-Net consists of a contracting path (encoder) to capture context and a symmetric expansive path (decoder) that enables precise localization. Skip connections are used between the contracting and expansive paths to combine low-level and high-level features.

## Dependencies

The following libraries are required to run the code:

* torch
* torchvision
* numpy
* pandas
* scikit-learn
* scikit-image
* matplotlib
* tqdm

## Usage

To use this code, you will need to:

1.  Download the 2018 Data Science Bowl dataset and place it in the `../input/` directory relative to the notebook.
2.  Install the required dependencies.
3.  Run the Jupyter Notebook `u-net-pytorch.ipynb`.

The notebook will:

1.  Load and preprocess the data.
2.  Define and compile the U-Net model.
3.  Train the model on the training data.
4.  Display example segmentations on the validation data.

## Results

After training for 120 epochs, the model achieves a validation loss of approximately 0.349. The notebook includes a function to visualize the model's predictions on the validation set, showing the original image, the predicted segmentation, and the ground truth mask.
