import numpy as np
class Dense:
    
    '''
    Initialize a Dense layer: a layer that is fully connected to a previous layer)
    Note: Input layers are NOT Dense layers!
    '''
    # Input size is size of previous layer
    # Output size is size of this layer
    def __init__(self, input_size, output_size):
        # Note: np.something((rows, cols)) creates a matrix of "something"-s
        # I.e. np.random.randn(rows, cols) creates a matrix of random numbers around a normal distribution
        # I.e. np.zeros((rows, cols)) creates a matrix of zeros
        
        '''
        Create a matrix of random weights
        Weights are randomized in order to break symmetry.
        This is important so that weights don't get updated in tandem, because that would cause all of them to receive similar updates'''
        # Input size = rows, output size = cols because of matrix multiplication ([m x n] * [n x o] = [m x o])
        # Weights belong to the layer they are producing outputs FOR (so they come before the node)
        # Columns represent nodes/neurons, rows represent weights
        self.weights = np.random.randn(input_size, output_size) * 0.01
        
        'Create a vector of biases, one per output neuron'
        # Weights start as randomized, biases start as 0 because weights create enough randomization
        # Biases act as a normalization to allow processing of zero-inputs. Biases remain constant throughout each neuron itself, whereas weights vary per individual connection.
        self.biases = np.zeros((1, output_size))
    
    'Forward propagation method -> moving data through the layer'
    # X is the layer/vector of neurons
    # Takes in previous layer as input, returns output layer 
    def forward(self, input) -> np.ndarray:
        # Store input layer (X) locally in order for accessing during backpropagation
        self.input = input
        
        # Return next layer via matrix multiplication (@ operator) between input * weight + bias
        # Z = W*A + b
        return input @ self.weights + self.biases
    
    def backward(self, delta_pre_activation: np.ndarray) -> np.ndarray:
        
        # Calculate size of input batch
        # batch_size = self.input.shape[0]
        
        # Calculate changes in weights
        # Use transpose in order to comply with dimensions for backpropagation
        self.delta_weights = self.input.T @ delta_pre_activation
        
        # Calculate changes in biases
        # Accumulates gradient for each bias, iterates over each batch
        self.delta_biases = np.sum(delta_pre_activation, axis=0, keepdims=True)
        
        # Computes the effects of a change in input (not necessarily first layer input)
        delta_input = delta_pre_activation @ self.weights.T
        
        return delta_input

'''ReLU class in order to apply ReLU activation function to a Dense activation vector'''
# Incorporating ReLU avoids linearity, therefore making the use of multiple layers meaningful
# In English that means the activation function determines whether or not an input meets a certain threshold to pass to the next layer.
# Without an activation function, you would just be making repeated linear combinations which is just linear regression in a trench coat.
# A "ReLU" is a vector in which the activations undergo ReLU transformation
# There are multiple different kinds of activation functions, ReLU is just one of them and is very basic.
class ReLU:
    def forward(self, pre_activation: np.ndarray) -> np.ndarray:
        self.pre_activation = pre_activation
        # Returns an augmented pre-activation in which any pre-activation < 0 gets treated as 0 
        # (This is what ReLU is in essence)
        'Return a matrix of just values > 0, with 0s otherwise'
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
    - Subtracting that from each logit (pre-softmaxed value) in the row (to avoid large numbers when exponentating)
    - Exponentiating each value via e^x where x is the original value after subtracting
    - Dividing that by the sum of each softmaxed value to normalize, resulting in a probability [0,1]
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
    '''
    Takes in the softmaxed probabilities and true classifications for each digit
    Outputs total loss using cross-entropy
    '''
    def forward(self, probs: np.ndarray, true_classes: np.ndarray) -> float:
        # Calculate number of samples in batch
        batch_size = probs.shape[0]
        
        # np.arange(size) creates an array of [0, 1, 2, ..., size-1]
        # Creates indexing of [image index, true class] in order to track probability of the true class being selected
        # So, each entry in probs look like [[0, 0.3], [1, 0.02]...] where the first input of each subarray is the index of the layer in the batch and the second input is the probability of the correct class being chosen
        true_probs = probs[np.arange(batch_size), true_classes]
        
        # Take log
        true_probs = np.clip(true_probs, 1e-9, 1 - 1e-9) # prevents log(0) from diverging to -infinity
        log_probs = np.log(true_probs)

        
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
        For the SoftMax + CrossEntropy combo ALONE:
        Gradient involves calculating the difference, for each neuron, between the actual vs desired probability (0 for non-target classification, 1 for target classification)
        probs contains probability information for each neuron of the output layer.
        The probability difference for each input in batch gets summed and divided by # batches to get the average difference in actual vs desired probability.
        THAT is the gradient of the entire batch.
        '''
        layer_gradients = (probs - one_hot) # Computes the gradients for each example, resulting in a matrix of difference per node, per example, per batch
        batch_gradients = layer_gradients / batch_size # Divide each individual gradient by size of batch to calculate specific loss relative to the batch
        
        return batch_gradients
    
class SGD:
    # Stochastic Gradient Descent is the process of actually changing the weights.
    # You don't change the weights within backpropagation methods themselves, as it would mess up wanting to use a different metric to change weights
    def __init__(self, learning_rate: float):
        # Set learning rate
        self.learning_rate = learning_rate
    
    def step(self, layers: list) -> None:
        # Loop through each dense layer that we want to update
        for layer in layers:
            # Only want to update layers that have weights (i.e. not the first layer)
            if hasattr(layer, "weights"):
                # Shift weights opposite to direction of gradient
                layer.weights -= self.learning_rate * layer.delta_weights
                # Shift biases opposite to direction of gradient
                layer.biases -= self.learning_rate * layer.delta_biases

class MLP:
    '''Initialize full Multi-layer Perceptron using exiting classes'''
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Create first dense layer
        self.dense1 = Dense(input_size, hidden_size)
        
        # Initialize ReLU()
        self.relu1 = ReLU()
        
        # Create second dense layer
        self.dense2 = Dense(hidden_size, output_size)
        
        # Create softmax
        self.softmax = Softmax()
        
        # Create loss function
        self.loss_fn = Loss()
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Pass inputs through first Dense
        self.pre_activation1 = self.dense1.forward(inputs)
        
        # Apply ReLU to first Dense layer's output
        self.activation1 = self.relu1.forward(self.pre_activation1)
        
        # Pass ReLU output through second Dense
        self.logits = self.dense2.forward(self.activation1)
        
        # Apply softmax to second Dense output to produce probabilities
        self.probabilities = self.softmax.forward(self.logits)
        
        return self.probabilities
    
    def compute_loss(self, probabilities: np.ndarray, true_classes: np.ndarray) -> float:
        # Compare predications vs true classification labels to produce gradient
        return self.loss_fn.forward(probabilities, true_classes)
    
    def backward(self, probabilities: np.ndarray, true_classes: np.ndarray) -> None:
        # Start from Softmax + Cross-Entropy Loss and work backwards
        gradient_logits = self.loss_fn.backward(probabilities, true_classes)
        
        # Backpropagate through second Dense layer
        gradient_activation1 = self.dense2.backward(gradient_logits)
        
        # Apply ReLU to backpropagation cycle
        gradient_pre_activation1 = self.relu1.backward(gradient_activation1)
        
        # Backpropagate through first Dense Layer
        
        self.dense1.backward(gradient_pre_activation1)

    def update(self, learning_rate: float) -> None:
        # Update first Dense layer parameters
        self.dense1.weights -= learning_rate * self.dense1.delta_weights
        self.dense1.biases -= learning_rate * self.dense1.delta_biases

        # Update second Dense layer parameters
        self.dense2.weights -= learning_rate * self.dense2.delta_weights
        self.dense2.biases -= learning_rate * self.dense2.delta_biases

# Fake batch of n MNIST-like images (784 pixels)
inputs = np.random.randn(10, 784)

# Fake correct labels (one digit 0–9 per image in the batch)
labels = np.random.randint(0, 10, size=inputs.shape[0])

# Initialize model
num_classes = 10
model = MLP(input_size=inputs.shape[1], hidden_size=128, output_size=num_classes)

# Set number of epochs (number of learning cycles/forward-backward passes)
n_epochs = 100
learning_rate = 0.01

for epoch in range(n_epochs):
    probabilities = model.forward(inputs)
    loss = model.compute_loss(probabilities, labels)
    model.backward(probabilities, labels)
    model.update(learning_rate=learning_rate)

    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, loss: {loss}")