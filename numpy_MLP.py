import numpy as np
import json
from sklearn.datasets import fetch_openml

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

# Tracker for visualization
class MetricsTracker:
    # Track the status of each epoch
    def __init__(self):
        self.history = {
            "config": {},
            "epochs": []
        }
    
    # Append config parameters to tracker: batch size, input size, hidden layer size, output size, learning rate, #epochs
    def log_config(self, config):
        self.history["config"] = config
    
    # Calculate gradient norm
    def calculate_layer_gradient_norm(self, layer):
        # Compute gradient norm to measure gradient descent between each epoch
        # We use the norm because every connection has a different weight and bias
        norms = {}
        norms["weights"] = float(np.linalg.norm(layer.delta_weights))
        norms["biases"] = float(np.linalg.norm(layer.delta_biases))
        return norms
    
    # Calculate gradient norms
    def calculate_model_gradient_norms(self, model):
        gradient_norms = {}
        layers = {"dense1" : model.dense1, "dense2" : model.dense2}
        for layer_name, layer in layers.items():
            if hasattr(layer, "weights") and hasattr(layer, "biases"):
                gradient_norms[layer_name] = self.calculate_layer_gradient_norm(layer)
        return gradient_norms
    
    # Calculate accuracy
    def calculate_accuracy(self, predictions, labels):
        predicted_classes = np.argmax(predictions, axis=1)
        return float(np.mean(predicted_classes==labels))
    
    # Log index, loss, accuracy, and gradient norms for each epoch
    def log_epoch(self, epoch, loss, accuracy, gradient_norms):
        epoch_data = {
            "epoch": int(epoch),
            "loss": float(loss),
            "accuracy": float(accuracy),
            "gradient_norms": gradient_norms
        }
        self.history["epochs"].append(epoch_data)
    
    # History getter
    def get_history(self):
        return self.history
    
    # Safe to json for frontend
    def save_json(self, filepath):
        with open(filepath, "w") as file:
            json.dump(self.history, file, indent=4)

# Trainer class in order to allow for customizations within training
# Will also handle minibatching, training/testing split, etc
# We minibatch in order to increase the amount of updates and to add noise to improve the generalization of the trainer
# Without mini batching, you only update once per epoch (for each image). WITH minibatching, you update #images/batch size times per epoch.
# Without batches, for 100 epochs you have 100 updates. With batch size of 32, you have 3200 updates.
class Trainer:
    def __init__(self, model, tracker):
        self.model = model
        self.tracker = tracker

    # Helper to average gradient norms across batch
    def average_gradient_norms(self, batch_gradient_norms):
        averaged = {}
        for batch_norms in batch_gradient_norms:
            for layer_name, norms in batch_norms.items():
                if layer_name not in averaged:
                    averaged[layer_name] = {"weights": [], "biases": []}
                averaged[layer_name]["weights"].append(norms["weights"])
                averaged[layer_name]["biases"].append(norms["biases"])
        for layer_name in averaged:
            averaged[layer_name]["weights"] = float(np.mean(averaged[layer_name]["weights"]))
            averaged[layer_name]["biases"] = float(np.mean(averaged[layer_name]["biases"]))
        return averaged
    
    def train(self, inputs, labels, num_epochs, learning_rate, batch_size):
        num_samples = inputs.shape[0]
        
        for epoch in range(num_epochs):
            # Shuffle entries in dataset for every epoch in case of clumping
            indices = np.random.permutation(num_samples)
            shuffled_inputs = inputs[indices]
            shuffled_labels = labels[indices]
            
            # Initialize epoch-specific data
            epoch_losses = []
            epoch_accuracies = []
            epoch_gradient_norms = []
            # Train with batch
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_inputs = shuffled_inputs[start:end]
                batch_labels = shuffled_labels[start:end]
                probabilities = self.model.forward(batch_inputs)
                loss = self.model.compute_loss(probabilities, batch_labels)
                self.model.backward(probabilities, batch_labels)
                accuracy = self.tracker.calculate_accuracy(probabilities, batch_labels)
                gradient_norms = self.tracker.calculate_model_gradient_norms(self.model)
                
                # Append epoch information
                epoch_losses.append(loss)
                epoch_accuracies.append(accuracy)
                epoch_gradient_norms.append(gradient_norms)
                self.model.update(learning_rate=learning_rate)

            # Tally averages for epoch
            avg_loss = float(np.mean(epoch_losses))
            avg_accuracy = float(np.mean(epoch_accuracies))
            avg_gradient_norms = self.average_gradient_norms(epoch_gradient_norms)
            
            self.tracker.log_epoch(epoch=epoch, loss=avg_loss, accuracy=avg_accuracy, gradient_norms=avg_gradient_norms)
            if epoch == 0 or (epoch) % 10 == 9:
                print(f"Training Epoch {epoch}/{num_epochs-1}, Training Loss: {avg_loss}, Training Accuracy: {avg_accuracy}")
        return self.tracker.get_history()
    
    # Testing implementation against training results
    def test(self, inputs, labels):
        probabilities = self.model.forward(inputs)
        loss = self.model.compute_loss(probabilities, labels)
        accuracy = self.tracker.calculate_accuracy(probabilities, labels)
        test_results = {"loss": float(loss), "accuracy": float(accuracy)}
        return test_results
        
                
'''# Fake batch of n MNIST-like images (784 pixels)
inputs = np.random.randn(10, 784)

# Fake correct labels (one digit 0–9 per image in the batch)
labels = np.random.randint(0, 10, size=inputs.shape[0])'''

# Download mnist dataset, assign
mnist = fetch_openml("mnist_784", version=1)
inputs = mnist.data.to_numpy()
labels = mnist.target.astype(int).to_numpy()

# Adjust train/test parameters
inputs = inputs/255
inputs = inputs[:1200]
labels = labels[:1200]
training_inputs = inputs[:1000]
training_labels = labels[:1000]
testing_inputs = inputs[1000:]
testing_labels = labels[1000:]

# Initialize model
num_classifications = 10
hidden_size = 128
model = MLP(input_size=inputs.shape[1], hidden_size=hidden_size, output_size=num_classifications)

# Set number of epochs (number of learning cycles/forward-backward passes)
num_epochs = 500
learning_rate = 0.01
batch_size = 48

# Initialize tracker
tracker = MetricsTracker()

# Initialize trainer
trainer = Trainer(model, tracker)

# Add config to tracker
# Append config parameters to tracker: batch size, input size, hidden layer size, output size, learning rate, batch size, #epochs
config = {
    "Total Inputs": inputs.shape[0],
    "Batch Size": batch_size,
    "Input Size": inputs.shape[1],
    "Hidden Layer Size": hidden_size,
    "Output Size": num_classifications,
    "Learning Rate": learning_rate,
    "# Epochs": num_epochs
    }
tracker.log_config(config)

trainer.train(training_inputs, training_labels, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size)

test_results = trainer.test(testing_inputs, testing_labels)
print(f"Testing Loss: {test_results["loss"]}, Testing Accuracy: {test_results["accuracy"]}")