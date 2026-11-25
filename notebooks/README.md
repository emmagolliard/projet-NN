# Notebooks – Formula 1 Team Classification

This folder contains the main Jupyter notebook developed for our Deep Learning project at ESILV.

## Main Notebook

`projet_nn.ipynb`

The notebook presents the complete development and evaluation of deep learning models designed to classify Formula 1 cars into 10 teams.

## Notebook Content

### 1. Dataset Preparation

The F1 Car Recognition dataset is downloaded from Roboflow.

The original YOLO object detection annotations are used to crop individual cars and organize them into classification folders.

The prepared dataset contains:

- 9,993 training images
- 892 validation images
- 461 test images
- 10 Formula 1 teams

### 2. Custom CNN

A convolutional neural network is built from scratch using TensorFlow and Keras.

The notebook includes the model architecture, training process, accuracy and loss curves, and evaluation results.

The custom CNN achieved 82.00% test accuracy.

### 3. Transfer Learning

MobileNetV2, pretrained on ImageNet, is adapted to recognize the 10 Formula 1 teams.

Its pretrained layers are initially frozen, and a new classification head is trained.

This model achieved 80.91% test accuracy.

### 4. Fine-Tuning

The last 30 layers of MobileNetV2 are fine-tuned using a reduced learning rate.

The fine-tuned model achieved 84.16% test accuracy, outperforming the two previous models in terms of accuracy.

### 5. Model Evaluation

The notebook compares the models using accuracy, loss, classification reports, and confusion matrices.

These evaluations help identify which Formula 1 teams are recognized correctly and which teams are frequently confused.

### 6. Color Experiment

The fine-tuned MobileNetV2 is evaluated on grayscale images to investigate the influence of color information.

The results show a significant decrease in accuracy, suggesting that car liveries play an important role in classification.

### 7. Duplicate Image Verification

Perceptual hashing is used to identify potentially duplicated or visually similar images across the datasets.

Fourteen suspicious test images are excluded for an additional evaluation on 447 images.

The filtered results are 81.43% accuracy for the custom CNN and 83.67% for the fine-tuned MobileNetV2.

This analysis highlights a potential limitation of the original dataset split.

## How to Use the Notebook

1. Open `projet_nn.ipynb` in Google Colab.
2. Select a GPU runtime if available.
3. Install the necessary Python libraries.
4. Download or restore the dataset.
5. Run the notebook cells sequentially.

The notebook contains the source code, training results, visualizations, and interpretation of the experiments.

## Notes

The image dataset and trained model files are not stored in this GitHub repository.

Access to the Roboflow dataset or a prepared dataset archive is required to reproduce the experiments.

The reported results were obtained from the experimental dataset split. Potential similarities between images across splits mean that the evaluation should not be considered completely free from data leakage.
