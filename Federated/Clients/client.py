# client.py
import os
import flwr as fl
import tensorflow as tf
import numpy as np
from model import build_model

# ===============================
# Configurations
# ===============================
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# These are passed from Docker or command line
TRAIN_DIR = os.environ.get("TRAIN_DIR", "data/train")
TEST_DIR  = os.environ.get("TEST_DIR", "data/test")
SERVER_ADDRESS = "192.168.100.196:8080"

# ===============================

# Data Augmentation (same as notebook)
# ===============================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomTranslation(0.02, 0.02),
    tf.keras.layers.RandomContrast(0.08),
])


# ===============================
# Preprocess function (same)
# ===============================
def preprocess(img, label):
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0  # MobileNetV2 scaling
    return img, label


# ===============================
# prepare() from notebook
# ===============================
def prepare(ds, augment=False):
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


# ===============================
# Load datasets (your notebook code)
# ===============================
def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=42
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Apply your prepare() + augmentation
    train_ds = prepare(train_ds, augment=True)
    val_ds = prepare(val_ds, augment=False)
    test_ds = prepare(test_ds, augment=False)

    return train_ds, val_ds, test_ds


# ===============================
# Class Weight Calculation (same)
# ===============================
def compute_class_weights():
    normal_count = len(os.listdir(os.path.join(TRAIN_DIR, "Normal")))
    malade_count = len(os.listdir(os.path.join(TRAIN_DIR, "Malade")))

    total = normal_count + malade_count

    class_weight = {
        0: total / (2 * normal_count),
        1: total / (2 * malade_count)
    }

    return class_weight


# ===============================
# Flower Client
# ===============================
class Client(fl.client.NumPyClient):

    def __init__(self, model, train_ds, val_ds, test_ds, class_weight):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.class_weight = class_weight

        # Count samples (correct FL behaviour)
        self.num_examples = sum(1 for _ in train_ds.unbatch())

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Receive model from server
        self.model.set_weights(parameters)

        # Train for a few epochs per round
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=3,  # You can tune this
            class_weight=self.class_weight,
            verbose=1
        )

        # Return updated weights
        return self.model.get_weights(), self.num_examples, {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.test_ds, verbose=0)
        return float(loss), self.num_examples, {"loss":float(loss),"accuracy": float(acc)}


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_datasets()
    class_weight = compute_class_weights()

    model = build_model()

    client = Client(model, train_ds, val_ds, test_ds, class_weight)

    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=client
    )
    # ========================================================
    #   ÉVALUATION FINALE APRÈS TOUTES LES ROUNDS FL
    # ========================================================

    print("\n[INFO] Évaluation finale du modèle global...\n")

    final_model = client.model

    # --- Loss & Accuracy ---
    loss, acc = final_model.evaluate(test_ds, verbose=1)
    print(f"\nFinal Test Loss : {loss:.4f}")
    print(f"Final Test Accuracy : {acc:.4f}")

    # --- Prédictions complètes ---
    y_true = []
    y_pred = []

    for batch, labels in test_ds:
        preds = final_model.predict(batch, verbose=0)
        preds = (preds >= 0.5).astype(int).reshape(-1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # --- Metrics + Classification Report ---
    from sklearn.metrics import confusion_matrix, classification_report

    print("\n============= Classification Report =============")
    print(classification_report(y_true, y_pred, digits=4))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)

    print("\n============= Confusion Matrix =============")
    print(cm)

    # --- Courbe du Loss (dernier entraînement local) ---
    import matplotlib.pyplot as plt

    if hasattr(final_model, "history") and final_model.history:
        history = final_model.history

        plt.figure(figsize=(6, 4))
        plt.plot(history.history["loss"], label="train_loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="val_loss")

        plt.title("Courbe du Loss (dernier round local)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    else:
        print("\n[WARNING] Aucun historique d'entraînement disponible pour tracer la courbe du loss.")
