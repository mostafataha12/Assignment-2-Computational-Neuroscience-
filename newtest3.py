import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Inputs
i1, i2 = 0.05, 0.10

target_o1, target_o2 = 0.01, 0.99  # Target outputs

# Initial weights
w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30  # Input to hidden weights
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55  # Hidden to output weights
b1, b2 = 0.35, 0.60  # Biases

# Forward pass (Hidden layer)
net_h1 = (i1 * w1) + (i2 * w3) + b1
net_h2 = (i1 * w2) + (i2 * w4) + b1
h1 = sigmoid(net_h1)
h2 = sigmoid(net_h2)
print(f"Hidden layer activations: h1={h1:.5f}, h2={h2:.5f}")

# Forward pass (Output layer)
net_o1 = (h1 * w5) + (h2 * w7) + b2
net_o2 = (h1 * w6) + (h2 * w8) + b2
o1 = sigmoid(net_o1)
o2 = sigmoid(net_o2)
print(f"Output layer activations: o1={o1:.5f}, o2={o2:.5f}")

# Calculate error
error_o1 = 0.5 * (target_o1 - o1) ** 2
error_o2 = 0.5 * (target_o2 - o2) ** 2
total_error = error_o1 + error_o2
print(f"Total Error: {total_error:.5f}")

# Backpropagation (Output layer)
delta_o1 = (o1 - target_o1) * sigmoid_derivative(o1)
delta_o2 = (o2 - target_o2) * sigmoid_derivative(o2)

# Gradients for hidden-to-output weights
dw5 = h1 * delta_o1
dw6 = h1 * delta_o2
dw7 = h2 * delta_o1
dw8 = h2 * delta_o2

delta_b2 = delta_o1 + delta_o2  # Bias gradient

# Backpropagation (Hidden layer)
delta_h1 = (delta_o1 * w5 + delta_o2 * w6) * sigmoid_derivative(h1)
delta_h2 = (delta_o1 * w7 + delta_o2 * w8) * sigmoid_derivative(h2)

# Gradients for input-to-hidden weights
dw1 = i1 * delta_h1
dw2 = i1 * delta_h2
dw3 = i2 * delta_h1
dw4 = i2 * delta_h2

delta_b1 = delta_h1 + delta_h2  # Bias gradient

print(f"Gradients (hidden-to-output): dw5={dw5:.5f}, dw6={dw6:.5f}, dw7={dw7:.5f}, dw8={dw8:.5f}")
print(f"Gradients (input-to-hidden): dw1={dw1:.5f}, dw2={dw2:.5f}, dw3={dw3:.5f}, dw4={dw4:.5f}")

# Learning rate
eta = 0.5

# Weight updates
w1 -= eta * dw1
w2 -= eta * dw2
w3 -= eta * dw3
w4 -= eta * dw4
w5 -= eta * dw5
w6 -= eta * dw6
w7 -= eta * dw7
w8 -= eta * dw8
b1 -= eta * delta_b1
b2 -= eta * delta_b2

print(f"Updated weights: w1={w1:.5f}, w2={w2:.5f}, w3={w3:.5f}, w4={w4:.5f}")
print(f"Updated weights: w5={w5:.5f}, w6={w6:.5f}, w7={w7:.5f}, w8={w8:.5f}")
print(f"Updated biases: b1={b1:.5f}, b2={b2:.5f}")