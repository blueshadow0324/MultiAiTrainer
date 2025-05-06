import os
import json
import requests
import socket
import tensorflow as tf

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP

NODE_ID = socket.gethostname()
COORDINATOR_URL = "http://192.168.1.143:8000"

# Register with coordinator
response = requests.post(f"{COORDINATOR_URL}/register", json={
    "node_id": NODE_ID,
    "node_ip": get_local_ip()
})
tf_config = response.json()

if "error" in tf_config:
    raise Exception(tf_config["error"])

os.environ["TF_CONFIG"] = json.dumps(tf_config)

# Fetch training config
training_config = requests.get(f"{COORDINATOR_URL}/config").json()

model_type = training_config["model"]
dataset_name = training_config["dataset"]
epochs = training_config["epochs"]
batch_size = training_config["batch_size"]
learning_rate = training_config["learning_rate"]

def build_model(model_type, learning_rate):
    if model_type == "dense_mnist":
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    else:
        raise ValueError("Unknown model type")

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['accuracy']
    )
    return model

def load_dataset(name):
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 784).astype("float32") / 255
        x_test = x_test.reshape(-1, 784).astype("float32") / 255
        return (x_train, y_train), (x_test, y_test)
    else:
        raise ValueError("Unknown dataset")

strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = build_model(model_type, learning_rate)
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    if tf_config["task"]["index"] == 0:
        loss, acc = model.evaluate(x_test, y_test)
        print(f"Chief - Test accuracy: {acc}")
