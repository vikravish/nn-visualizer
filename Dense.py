import numpy as np

class Dense:
    # Initialize a Dense layer: a layer that is fully connected to a previous layer)
    # Input size is size of previous layer
    # Output size is size of this layer
    def __init__(self, input_size, output_size):
        # Note: np.something((rows, cols)) creates a matrix of "something"-s
        # I.e. np.random.randn(rows, cols) creates a matrix of random numbers around a normal distribution
        # I.e. np.zeroes((rows, cols)) creates a matrix of zeros
        
        # Create a matrix of random weights
        # Input size = rows, output size = cols because of matrix multiplication
        # Weights belong the layer they are producing outputs FOR
        self.W = np.random.randn(input_size, output_size) * 0.01
        
        # Create a vector of biases, one per output neuron
        self.b = np.zeros((1, output_size))
    
    # Forward propagation method -> moving data through the layer
    # X is the layer/vector of neurons
    # Takes in previous layer as input, returns output layer 
    def forward(self, X) -> np.ndarray:
        # Store input layer (X) locally in order for accessing during backpropagation
        self.X = X
        
        # Return next layer via matrix multiplication (@ operator) between + bias
        return X @ self.W + self.b

# ReLU class in order to apply ReLU activation function to a Dense activation vector
# Incorporating ReLU avoids linearity, therefore making the use of multiple layers meaningful
# A "ReLU" is a vector in which the activations undergo ReLU transformation
class ReLU:
    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.Z = Z
        # Returns an augmented Z in which any activation < 0 gets treated as 0
        return np.maximum(0,Z)
    
X = np.random.randn(1,784) # Fake input data for testing -> 1 row x 784 cols
layer1 = Dense(784, 128) # Layers are hardcoded
z1 = layer1.forward(X) # Computes matrix-vector mutiplication between output weights matrix and input vector
relu = ReLU() # Initialize a ReLU vector
z1_relu = relu.forward(z1) # Apply ReLU function to z1 and store at z1_relu

print("Input shape:", X.shape)
print("Weights shape:", layer1.W.shape)
print("Bias shape:", layer1.b.shape)
print(z1_relu)