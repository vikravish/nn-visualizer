import numpy as np

class Dense:
    # Initialize a Dense layer: a layer that is fully connected to a previous layer)
    # Input size is size of previous layer
    # Output size is size of this layer
    def __init__(self, input_size, output_size):
        # Note: np.something((rows, cols)) creates a matrix of "something"-s
        # I.e. np.random.randn(rows, cols) creates a matrix of random numbers around a normal distribution
        # I.e. np.zeros((rows, cols)) creates a matrix of zeros
        
        # Create a matrix of random weights
        # Input size = rows, output size = cols because of matrix multiplication
        # Weights belong the layer they are producing outputs FOR
        # Columns represent neurons, rows represent weights
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

class Softmax:
    # Converts output activation scores into probabilites using exponents to amplify differences
    '''Softmaxing includes:
    - Calculating the maximum value of each output row (for each image in the batch)
    - Subtracting that from each value in the row (to avoid large numbers when exponentating)
    - Exponentiating each value via e^x where x is the original value after subtracting
    - Calculating the probability of each output using the new shifted values
    '''
    def forward(self, logits: np.ndarray) -> np.ndarray:
        # Find max along each row
        max_vals = np.max(logits, axis=1, keepdims=True)
        
        # Subtract maximums for each row
        shifted_logits = logits - max_vals
        
        # Apply e^x via np.exp()
        exp_values = np.exp(shifted_logits)
        
        # Sum to find probability divisor
        sum_exp = np.sum(exp_values, axis=1, keepDims=True)
        
        # Calculate probabilites (normalize)
        probabilities = exp_values/sum_exp
        
        # Store probabilities to use with backpropagation
        self.probabilities = probabilities
        
        return probabilities

# Loss class to use for learning using loss = -log(p_correct) for each softmaxed output vector for each image in batch
class Loss:
    # Takes in the softmaxed probabilities and true classifications for each digit
    # Outputs total loss using cross-entropy
    def forward(self, probs: np.ndarray, true_classes: np.ndarray) -> float:
        # Calculate number of samples in batch
        batch_size = probs.shape[0]
        
        # np.arange(size) creates an array of [0, 1, 2, ..., size-1]
        # Creates indexing of [image index, true class] in order to track probability of the true class being selected
        # So, each entry in probs look like [[0, 0.3], [1, 0.02]...] where the first input of each subarray is the index and the second input is the probability of the correct class being chosen
        true_probs = probs[np.arange(batch_size), true_classes]
        
        # Take log
        log_probs = np.log(true_probs)
        
        # Compute average loss across batch via negative mean
        loss = -np.mean(log_probs)
        
        return loss
    
X = np.random.randn(1,784) # Fake input data for testing -> 1 row x 784 cols

layer1 = Dense(784, 128) # Layers are hardcoded
z1 = layer1.forward(X) # Computes matrix-vector mutiplication between output weights matrix and input vector

relu = ReLU() # Initialize a ReLU vector
z1_relu = relu.forward(z1) # Apply ReLU function to z1 and store at z1_relu

layer2 = Dense(128, 10) # Create second Dense layer from z1_relu
logits = layer2.forward(z1_relu) # Create output layer from second Dense layer

print("Input shape:", X.shape)

print("Layer 1 W shape:", layer1.W.shape)

print("Layer 1 b shape:", layer1.b.shape)

print("z1 shape:", z1.shape)

print("a1/ReLU shape:", z1_relu.shape)

print("Layer 2 W shape:", layer2.W.shape)

print("Logits shape:", logits.shape)

print(logits)