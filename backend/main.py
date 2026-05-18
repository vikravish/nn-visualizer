from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
from sklearn.datasets import fetch_openml

# Import your classes
from numpy_MLP import (
    MLP,
    MetricsTracker,
    Trainer
)

app = FastAPI()

# Allow frontend requests from Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache mnist dataset

mnist_cache = None


def load_mnist():
    global mnist_cache

    if mnist_cache is None:
        mnist = fetch_openml("mnist_784", version=1)

        inputs = mnist.data.to_numpy() / 255.0
        labels = mnist.target.astype(int).to_numpy()

        mnist_cache = (inputs, labels)

    return mnist_cache


# Training Config Class

class TrainingConfig(BaseModel):
    hidden_size: int
    epochs: int
    learning_rate: float
    batch_size: int
    training_size: int

# Root endpoint

@app.get("/")
def home():
    return {
        "message": "NN Visualizer backend running"
    }


# Training endpoint

@app.post("/train")
def train_model(config: TrainingConfig):

    # Load cached MNIST
    inputs, labels = load_mnist()

    # Reduce dataset size
    inputs = inputs[:config.training_size]
    labels = labels[:config.training_size]

    # Train/test split
    split_index = int(config.training_size * 0.8)

    train_inputs = inputs[:split_index]
    train_labels = labels[:split_index]

    test_inputs = inputs[split_index:]
    test_labels = labels[split_index:]

    # Initialize model
    model = MLP(
        input_size=784,
        hidden_size=config.hidden_size,
        output_size=10
    )

    # Initialize tracker
    tracker = MetricsTracker()

    tracker.log_config({
        "training_size": config.training_size,
        "hidden_size": config.hidden_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size
    })

    # Initialize trainer
    trainer = Trainer(model, tracker)

    # Train model
    history = trainer.train(
        train_inputs,
        train_labels,
        num_epochs=config.epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size
    )

    # Evaluate on test set
    test_results = trainer.evaluate(
        test_inputs,
        test_labels
    )

    # Return everything frontend needs
    return {
        "history": history,
        "test_results": test_results
    }