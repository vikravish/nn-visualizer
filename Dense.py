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
        self.weights = np.random.randn(input_size, output_size) * 0.01
        
        # Create a vector of biases, one per output neuron
        self.biases = np.zeros((1, output_size))
    
    # Forward propagation method -> moving data through the layer
    # X is the layer/vector of neurons
    # Takes in previous layer as input, returns output layer 
    def forward(self, input) -> np.ndarray:
        # Store input layer (X) locally in order for accessing during backpropagation
        self.input = input
        
        # Return next layer via matrix multiplication (@ operator) between + bias
        return input @ self.weights + self.biases
    
    def backward(self, delta_pre_activation: np.ndarray) -> np.ndarray:
        
        # Calculate size of input batch
        batch_size = self.input.shape[0]
        
        # Calculate changes in weights
        # Use transpose in order to comply with dimensions for backpropagation
        self.delta_weights = self.input.T @ delta_pre_activation
        
        # Calculate changes in biases
        # Accumulates gradiant for each bias, iterates over each batch
        self.delta_biases = np.sum(delta_pre_activation, axis=0, keepdims=True)
        
        # Computes the effects of a change in input (not necessarily first layer input)
        delta_input = delta_pre_activation @ self.weights.T
        
        return delta_input

# ReLU class in order to apply ReLU activation function to a Dense activation vector
# Incorporating ReLU avoids linearity, therefore making the use of multiple layers meaningful
# A "ReLU" is a vector in which the activations undergo ReLU transformation
class ReLU:
    def forward(self, pre_activation: np.ndarray) -> np.ndarray:
        self.pre_activation = pre_activation
        # Returns an augmented pre-activation in which any pre-activation < 0 gets treated as 0
        return np.maximum(0,pre_activation)
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        # Don't overwrite original gradient matrix
        pre_gradient = gradient.copy()
        
        # Set the gradient of any pre_activation < 0 to 0 (Main purpose of ReLU)
        pre_gradient[self.pre_activation <= 0] = 0
        
        return pre_gradient

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
        sum_exp = np.sum(exp_values, axis=1, keepdims=True)
        
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
        # So, each entry in probs look like [[0, 0.3], [1, 0.02]...] where the first input of each subarray is the index of the layer in the batch and the second input is the probability of the correct class being chosen
        true_probs = probs[np.arange(batch_size), true_classes]
        
        # Take log
        log_probs = np.log(true_probs)
        true_probs = np.clip(probs, 1e-9, 1 - 1e-9) # prevents log(0) from diverging to -infinity
        
        # Compute average loss across batch via negative mean
        loss = -np.mean(log_probs)
        
        return loss

    def backward(self, probs:np.ndarray, true_classes: np.ndarray) -> np.ndarray:
        # Calculate number of samples in batch
        batch_size = probs.shape[0]
        
        # One-hot encode probability vector
        # np.zeros_like() creates matrix with matching shape filled with 0s
        one_hot = np.zeros_like(probs)
        # Set the index of the true class for each batch to 1 (to signal which index is the true class)
        one_hot[np.arange(batch_size), true_classes] = 1
        
        # Compute gradient
        '''
        Gradient measures how sensitive the change in loss is in respect to a single value
        For the SoftMax + CrossEntropy combo alone:
        Gradient involves calculating the difference, for each neuron, between the actual vs desired probability (0 for non-target classification, 1 for target classification)
        probs contains probability information for each neuron of the output layer.
        The probability difference for each input in batch gets summed and divided by # batches to get the average difference in actual vs desired probability.
        THAT is the gradient of the entire batch.
        '''
        layer_gradient = (probs - one_hot)
        batch_gradient = layer_gradient / batch_size
        
        return batch_gradient
    
class SGD:
    # Stochastic Gradient Descent is the process of actually changing the weights.
    # You don't change the weights within backpropagation methods themselves, as it would mess up wanting to use a different metric to change weights
    def __init__(self, learning_rate: float):
        # Set learning rate
        self.learning_rate = learning_rate
    
    def step(self, layers: list) -> None:
        # Loop through each dense layer that we want to update
        for layer in layers:
            # Only want to update layers that have weights (ie not the first layer)
            if hasattr(layer, "weights"):
                # Shift weights opposite to direction of gradient
                layer.weights -= self.learning_rate * layer.delta_weights
                # Shift biases opposite to direction of gradient
                layer.biases -= self.learning_rate * layer.delta_biases
    
# Fake batch of 4 MNIST-like images
inputs = np.random.randn(4, 784)

# Fake correct labels
labels = np.array([1, 3, 0, 5])

dense1 = Dense(784, 128)
relu1 = ReLU()
dense2 = Dense(128, 10)
softmax = Softmax()
loss_fn = Loss()
optimizer = SGD(learning_rate=0.01)

n_epochs = 100
for epoch in range(n_epochs):
    pre_activation1 = dense1.forward(inputs)
    activation1 = relu1.forward(pre_activation1)
    logits = dense2.forward(activation1)
    probabilities = softmax.forward(logits)
    loss = loss_fn.forward(probabilities, labels)

    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, loss: {loss}")

    grad_logits = loss_fn.backward(probabilities, labels)
    grad_activation1 = dense2.backward(grad_logits)
    grad_pre_activation1 = relu1.backward(grad_activation1)
    dense1.backward(grad_pre_activation1)

    optimizer.step([dense1, dense2])