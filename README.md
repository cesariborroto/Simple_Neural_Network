# Classifying Fashion Mnist

This repository contains a Jupyter Notebook that demonstrates the training of a simple neural network to classify the Fashion Mnist dataset.

## Notebook Author

Cesar Borroto

## Description

The notebook begins by importing the necessary libraries, including TensorFlow and Keras. It then loads the Fashion Mnist dataset using the `fashion_mnist` module provided by Keras. The dataset is divided into training, validation, and testing sets. The notebook further normalizes the data to a range of [0, 1].

Next, the notebook constructs a neural network architecture using the Sequential model from Keras. The model consists of multiple dense layers with ReLU activation, followed by a final dense layer with softmax activation for multi-class classification.

The architecture of the neural network is summarized, and the weights and biases of the first hidden layer are extracted and printed. The model is compiled with the sparse categorical cross-entropy loss and stochastic gradient descent optimizer.

The model is trained on the training data and evaluated on the validation data. The training progress is visualized using matplotlib to plot the loss and accuracy for each epoch.

Finally, the trained model is evaluated on the test data, and predictions are made on a subset of the test data. The predicted probabilities and class labels are displayed.

## Dependencies

The notebook requires the following dependencies:
- TensorFlow
- Keras
- Matplotlib
- NumPy

Make sure to install these dependencies before running the notebook.

## Dataset

The Fashion Mnist dataset consists of 60,000 28x28 grayscale images of 10 fashion categories, along with a test set of 10,000 images. The categories include T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

## Usage

To run the notebook, follow these steps:
1. Install the required dependencies (TensorFlow, Keras, Matplotlib, NumPy).
2. Launch Jupyter Notebook.
3. Open the `Classifying_Fashion_Mnist.ipynb` file.
4. Execute each cell in the notebook sequentially.

## Results

The notebook trains a simple neural network on the Fashion Mnist dataset and achieves a certain accuracy on the test set. It also provides visualizations of the training progress and predictions made by the trained model.
