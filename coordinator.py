from flask import Flask, request, jsonify
from threading import Lock
import json
import os

app = Flask(__name__)

BASE_PORT = 12345
MAX_WORKERS = 20
CONFIG_PATH = "training_config.json"

workers = []
node_id_map = {}
lock = Lock()

@app.route("/register", methods=["POST"])
def register_node():
    data = request.get_json()
    node_id = data.get("node_id")
    node_ip = data.get("node_ip")

    if not node_id or not node_ip:
        return jsonify({"error": "Missing node_id or node_ip"}), 400

    with lock:
        if node_id in node_id_map:
            index = node_id_map[node_id]
        elif len(workers) < MAX_WORKERS:
            address = f"{node_ip}:{BASE_PORT}"
            index = len(workers)
            workers.append(address)
            node_id_map[node_id] = index
        else:
            return jsonify({"error": "Cluster full"}), 403

        tf_config = {
            "cluster": {"worker": workers},
            "task": {"type": "worker", "index": index}
        }
        return jsonify(tf_config)

@app.route("/config", methods=["GET"])
def get_config():
    if not os.path.exists(CONFIG_PATH):
        return jsonify({"error": "No config set"}), 404
    with open(CONFIG_PATH, "r") as f:
        return jsonify(json.load(f))

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"nodes": workers, "node_map": node_id_map})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)