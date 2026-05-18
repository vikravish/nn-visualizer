import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.datasets import fetch_openml

from numpy_MLP import MLP, MetricsTracker, Trainer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)

mnist_cache = None


def load_mnist():
    global mnist_cache
    if mnist_cache is None:
        mnist = fetch_openml("mnist_784", version=1)
        inputs = mnist.data.to_numpy() / 255.0
        labels = mnist.target.astype(int).to_numpy()
        mnist_cache = (inputs, labels)
    return mnist_cache


@app.on_event("startup")
def preload_mnist():
    load_mnist()


def compute_confusion_matrix(probs, true_labels):
    predicted = np.argmax(probs, axis=1)
    matrix = np.zeros((10, 10), dtype=int)
    np.add.at(matrix, (true_labels, predicted), 1)
    return matrix.tolist()


def find_misclassified(test_inputs, test_labels, probs, max_samples=20):
    predicted = np.argmax(probs, axis=1)
    wrong = np.where(predicted != test_labels)[0][:max_samples]
    return [
        {
            "pixels": test_inputs[i].tolist(),
            "true_label": int(test_labels[i]),
            "predicted_label": int(predicted[i]),
            "confidence": float(probs[i][predicted[i]]),
        }
        for i in wrong
    ]


class TrainingConfig(BaseModel):
    hidden_size: int
    epochs: int
    learning_rate: float
    batch_size: int
    training_size: int


@app.get("/")
def home():
    return {"message": "NN Visualizer backend running"}


@app.post("/train")
def train_model(config: TrainingConfig):
    inputs, labels = load_mnist()

    inputs = inputs[:config.training_size]
    labels = labels[:config.training_size]

    split_index = int(config.training_size * 0.8)
    train_inputs = inputs[:split_index]
    train_labels = labels[:split_index]
    test_inputs = inputs[split_index:]
    test_labels = labels[split_index:]

    model = MLP(input_size=784, hidden_size=config.hidden_size, output_size=10)

    tracker = MetricsTracker()
    tracker.log_config({
        "training_size": config.training_size,
        "hidden_size": config.hidden_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
    })

    trainer = Trainer(model, tracker)

    def event_generator():
        for epoch_data in trainer.train_stream(
            train_inputs,
            train_labels,
            num_epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
        ):
            yield f"data: {json.dumps(epoch_data)}\n\n"

        probs = model.forward(test_inputs)
        test_loss = model.compute_loss(probs, test_labels)
        test_accuracy = tracker.calculate_accuracy(probs, test_labels)

        complete_event = {
            "type": "complete",
            "test_results": {
                "loss": float(test_loss),
                "accuracy": float(test_accuracy),
            },
            "confusion_matrix": compute_confusion_matrix(probs, test_labels),
            "misclassified": find_misclassified(test_inputs, test_labels, probs),
        }
        yield f"data: {json.dumps(complete_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
