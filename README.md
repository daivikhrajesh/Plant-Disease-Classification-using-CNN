# Plant-Disease-Classification-using-CNN
Detect and classify plant diseases with high accuracy using deep learning models.

### Plant Disease Classification using Convolutional Neural Networks

This repository contains code for a deep learning model developed to classify plant diseases using Convolutional Neural Networks (CNNs). The model is trained on a dataset consisting of images of various plant diseases and healthy plants.

---

#### Overview

Plant diseases can significantly affect crop yield and quality. Early detection and classification of these diseases are crucial for effective management and mitigation strategies. This project aims to address this challenge by developing a deep learning-based solution for automated plant disease classification.

---

#### Features

- Utilizes TensorFlow and Keras for model development.
- Implements data augmentation techniques to improve model generalization.
- Provides visualization of training and validation metrics using Matplotlib.
- Supports multi-class classification for various plant diseases.
- Offers a straightforward pipeline for training, validation, and testing.

---

#### Dataset

The dataset used in this project consists of images of diseased plants as well as healthy plants. It is organized into different classes based on the type of disease or health status of the plant. The dataset is preprocessed and divided into training, validation, and testing sets for model training and evaluation.

---

#### Model Architecture

The model architecture employs a series of convolutional and pooling layers followed by fully connected layers for classification. It utilizes techniques such as resizing, rescaling, and data augmentation to enhance performance and robustness.

---

#### Usage

1. **Data Preparation**: Organize the dataset into appropriate directories (e.g., train, validation, test) and ensure proper class labeling.
2. **Model Training**: Train the CNN model using the provided training data. Adjust hyperparameters as needed.
3. **Model Evaluation**: Evaluate the trained model's performance on the validation and test datasets.
4. **Inference**: Use the trained model to make predictions on new images of plants to classify diseases.

---

#### Dependencies

- TensorFlow
- Matplotlib
- NumPy

