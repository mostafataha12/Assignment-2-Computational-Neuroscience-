# Neural Network Forward Pass and Backpropagation

This repository contains a Python implementation of a simple neural network with one hidden layer. The code performs a **forward pass**, **error computation**, and **backpropagation** for a given set of inputs and target outputs. The neural network uses the **sigmoid activation function** and **gradient descent** for weight updates.

## Features
- Implements a feedforward neural network with one hidden layer.
- Uses the sigmoid activation function.
- Computes total error using the squared error function.
- Performs backpropagation to adjust weights and biases.
- Uses gradient descent for optimization.

## Code Explanation

### 1. Forward Pass
The forward pass computes the activations of the hidden and output layers.

- Hidden layer:
  - Takes inputs and calculates the net sum using initial weights and biases.
  - Applies the sigmoid activation function.
- Output layer:
  - Takes the hidden layer outputs and computes the final output values.
  - Applies the sigmoid activation function.
- The outputs are printed for verification.

### 2. Error Computation
The error is computed using the squared error function:

\[
E = \frac{1}{2} \sum (target - output)^2
\]

The total error is printed after computation.

### 3. Backpropagation
Backpropagation updates the weights and biases based on the error gradient.

- Computes gradients for output layer weights using the derivative of the sigmoid function.
- Computes gradients for hidden layer weights by propagating the error backward.
- Updates the weights using gradient descent:

\[
w = w - \eta \times \frac{\partial E}{\partial w}
\]

where \( \eta \) is the learning rate.

The updated weights and biases are printed after one iteration of backpropagation.


## Requirements
- Python 3.x
- NumPy

## Future Enhancements
- Implement multiple training iterations.
- Add more activation functions (ReLU, Tanh, etc.).
- Extend to multi-class classification.

## License
This project is open-source and available under the MIT License.

