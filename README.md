# 🦷🪥 Dental Image Denoising and Classification with CNN, MLP, VGG, ResNet and Autoencoder

This repository contains implementations of various deep learning architectures for classifying images from the popular MNIST dataset 🖼️. The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. This project explores the performance of different models for image classification, including:

- 🌐 **CNN (Convolutional Neural Network)**
- 🖥️ **MLP (Multilayer Perceptron)**
- 🔍 **VGG-like CNN**
- 🌀 **ResNet-like CNN**
- 🔄 **Autoencoder for Feature Learning**

## 🚀 Project Overview

The goal is to classify handwritten digits using multiple deep learning models and evaluate them based on common metrics such as **accuracy**, **precision**, **recall**, **F1 score**, and **ROC-AUC**. The project includes:

1. **Data Preprocessing**: Reshaping, normalizing, and splitting the dataset.
2. **Model Architectures**:
   - CNN: A simple convolutional neural network.
   - MLP: A fully connected network for baseline performance.
   - VGG-like CNN: A deeper CNN model inspired by VGGNet.
   - ResNet-like CNN: A residual network with skip connections.
   - Autoencoder: For unsupervised feature learning.
3. **Training & Evaluation**: Models are trained for 10 epochs and evaluated on the test set.
4. **Visualization**:
   - Confusion Matrix 📊
   - ROC Curve 📈

## 📋 Evaluation Metrics

The models are evaluated using the following metrics:
- **Accuracy**: Overall correctness of the model's predictions.
- **Precision**: Proportion of correctly predicted positive observations to the total predicted positives.
- **Recall**: Proportion of correctly predicted positive observations to all observations in the actual class.
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC-AUC Score**: The Area Under the ROC Curve to evaluate model performance across all classification thresholds.

## ⚙️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist-classification.git
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training and evaluation script:
   ```bash
   python train_and_evaluate.py
   ```

## 📚 Libraries Used

- `numpy` for numerical operations
- `matplotlib` & `seaborn` for visualizations
- `tensorflow/keras` for deep learning models
- `scikit-learn` for evaluation metrics

## 📝 Conclusion

This project provides insights into how different neural network architectures perform on the MNIST dataset. Explore the results to see which model works best for your image classification tasks! 😊

---
