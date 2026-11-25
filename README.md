# Formula 1 Team Classification Using Deep Learning

## Project Overview

This project aims to identify Formula 1 teams from images of racing cars using deep learning.

The objective is to compare a custom Convolutional Neural Network (CNN) trained from scratch with a pretrained MobileNetV2 model, and investigate how transfer learning, fine-tuning, and color information affect classification performance.

The project was developed as part of our Deep Learning course at ESILV.

## Dataset

We use the [F1 Car Recognition dataset from Roboflow](https://universe.roboflow.com/yoav-fogel-yia3f/f1-car-recognition).

The dataset originally contains images annotated for object detection. We converted the YOLO annotations into cropped car images for classification.

The dataset contains 10 Formula 1 teams:

- Alfa Romeo
- AlphaTauri
- Alpine
- Aston Martin
- Ferrari
- Haas
- McLaren
- Mercedes
- Red Bull Racing
- Williams

After preprocessing, we obtained 9,993 training images, 892 validation images, and 461 test images.

## Models

### Custom CNN

We developed a convolutional neural network from scratch using TensorFlow and Keras.

The architecture includes three convolutional blocks with MaxPooling, followed by GlobalAveragePooling, a Dense layer, Dropout, and a Softmax classification layer.

### MobileNetV2 – Transfer Learning

We used MobileNetV2 pretrained on ImageNet and replaced its original classification head with a new layer adapted to our 10 Formula 1 teams.

Initially, the pretrained convolutional layers were frozen.

### MobileNetV2 – Fine-Tuning

We then fine-tuned the last 30 layers of MobileNetV2 using a lower learning rate, while keeping BatchNormalization layers frozen.

## Results

The models achieved the following results on the original test set:

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| Custom CNN | 82.00% | 0.5755 |
| MobileNetV2 | 80.91% | 0.6497 |
| Fine-tuned MobileNetV2 | 84.16% | 0.5874 |

Fine-tuned MobileNetV2 achieved the highest test accuracy, although the custom CNN obtained a slightly lower test loss.

## Additional Experiments

### Impact of Color

We evaluated the fine-tuned MobileNetV2 on grayscale versions of the test images.

Accuracy decreased from approximately 84% on color images to approximately 28% on grayscale images.

This suggests that the model is highly sensitive to color information, which is consistent with the distinctive liveries of Formula 1 teams.

However, grayscale conversion also introduces a difference from the images used during training.

### Duplicate Image Analysis

We used perceptual hashing to identify potentially similar images across the training, validation, and test sets.

Fourteen distinct test images were flagged as potentially similar to images in the other datasets. Visual inspection confirmed that some images were modified versions of the same original photograph.

After excluding the 14 flagged test images, we evaluated the models on the remaining 447 images.

| Model | Filtered Test Accuracy |
|---|---:|
| Custom CNN | 81.43% |
| Fine-tuned MobileNetV2 | 83.67% |

The results remained close to the initial scores. However, this filtering does not guarantee the complete absence of data leakage.

A more rigorous evaluation would require splitting the original images before applying data augmentation and retraining the models.

## Technologies

- Python
- TensorFlow and Keras
- Google Colab
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Roboflow

## Repository Structure

```text
Projet-NN-/
├── README.md
└── notebooks/
    ├── README.md
    └── projet_nn.ipynb
```

## How to Run

1. Open `notebooks/projet_nn.ipynb` in Google Colab.
2. Enable GPU acceleration if available.
3. Install the required libraries.
4. Download the dataset from Roboflow or restore the prepared dataset from Google Drive.
5. Execute the notebook cells in order.
6. Explore the model comparisons, confusion matrices, and additional experiments.

A Roboflow API key may be required to download the dataset. API keys and private Google Drive files are not included in this repository.

## Conclusion

This project shows that transfer learning can be useful for Formula 1 team classification, particularly when pretrained features are adapted through fine-tuning.

The custom CNN achieved competitive results, while fine-tuned MobileNetV2 obtained the highest test accuracy.

The experiments also highlighted the influence of color information and the importance of checking datasets for potential data leakage.
