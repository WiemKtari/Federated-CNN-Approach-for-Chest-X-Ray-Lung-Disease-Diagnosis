# server.py
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
from flwr.common import parameters_to_ndarrays

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from model import build_model  # must be identical to client model

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# -------------------------------
# Preprocess (same as client)
# -------------------------------
def preprocess(img, label):
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0
    return img, label

def prepare(ds):
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

# -------------------------------
# Load Server Test Set
# -------------------------------
def load_server_testset():
    ds = tf.keras.utils.image_dataset_from_directory(
        "TEST_DATA",
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return prepare(ds)

# -------------------------------
# Evaluate Final Model
# -------------------------------
def evaluate_global_model(weights):
    print("\n===== FINAL GLOBAL MODEL EVALUATION =====\n")

    test_ds = load_server_testset()

    model = build_model()
    model.set_weights(weights)

    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\nServer Accuracy = {acc:.4f}")

    y_true, y_pred, y_scores = [], [], []

    for x, y in test_ds:
        p = model.predict(x)
        y_scores.extend(p.flatten())
        y_pred.extend((p > 0.5).astype("int").flatten())
        y_true.extend(y.numpy().flatten())

    y_true, y_pred, y_scores = map(np.array, [y_true, y_pred, y_scores])

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Global Model")
    plt.legend()
    plt.show()

# -------------------------------
# Custom Strategy with Early Stopping
# -------------------------------
class EarlyStoppingFedAvg(FedAvg):
    def __init__(self, min_delta=0.001, **kwargs):
        super().__init__(**kwargs)
        self.min_delta = min_delta
        self.last_loss = None
        self.stop_training = False
        self.final_weights = None
        self.loss_history = []

    def evaluate(self, rnd, parameters):
        """Evaluate global model loss on server dataset after each round."""
        print(f"\n--- SERVER EVALUATION ROUND {rnd} ---")

        model = build_model()
        model.set_weights(parameters_to_ndarrays(parameters))

        test_ds = load_server_testset()
        loss, acc = model.evaluate(test_ds, verbose=0)

        print(f"Server Loss: {loss:.6f} | Accuracy: {acc:.4f}")
        self.loss_history.append(loss)

        # Check convergence
        if self.last_loss is not None:
            improvement = abs(self.last_loss - loss)

            if improvement < self.min_delta:
                print("\n### EARLY STOPPING TRIGGERED ###")
                print(f"Loss improvement {improvement:.6f} < min_delta {self.min_delta}")
                self.stop_training = True

        self.last_loss = loss
        return float(loss), {"accuracy": float(acc)}

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)

        if aggregated is not None:
            params, _ = aggregated
            self.final_weights = parameters_to_ndarrays(params)

        return aggregated

# -------------------------------
# Main Server Loop
# -------------------------------
def main(port: int = 8080):
    strategy = EarlyStoppingFedAvg(
        min_delta=0.001,         
        fraction_fit=1.0,
        min_fit_clients=4,
        min_available_clients=4,
    )

    print(f"Starting Flower server on 0.0.0.0:{port}")

    # Run exactly 5 rounds
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    # Final evaluation at the end of training
    evaluate_global_model(strategy.final_weights)

    # Plot server loss curve
    plt.figure()
    plt.plot(strategy.loss_history, marker='o')
    plt.xlabel("Round")
    plt.ylabel("Server Loss")
    plt.title("Global Model Loss per Round")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    import fire
    fire.Fire(main)
