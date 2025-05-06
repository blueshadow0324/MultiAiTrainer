import os
import json
import requests
import socket
import tensorflow as tf

# Your unique ID (e.g., device serial number or UUID)
NODE_ID = socket.gethostname()  # Or something unique per client
COORDINATOR_URL = "http://192.168.1.143:8000"

# Step 1: Get local IP
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP

# Step 2: Register with coordinator
response = requests.post(f"{COORDINATOR_URL}/register", json={
    "node_id": NODE_ID,
    "node_ip": get_local_ip()
})
tf_config = response.json()

if "error" in tf_config:
    raise Exception(tf_config["error"])

os.environ["TF_CONFIG"] = json.dumps(tf_config)

# Step 3: Distributed Training
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255

model.fit(x_train, y_train, epochs=5, batch_size=64)

# Evaluate (only chief logs it)
if tf_config["task"]["index"] == 0:
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Chief - Test accuracy: {acc}")
