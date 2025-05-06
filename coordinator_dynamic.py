from flask import Flask, request, jsonify
from threading import Lock

app = Flask(__name__)

# Configurable
BASE_PORT = 12345
MAX_WORKERS = 10  # Adjust as needed

# Internal state
workers = []  # e.g., ["192.168.1.10:12345", ...]
node_id_map = {}  # Maps node_id to index
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
            port = BASE_PORT
            address = f"{node_ip}:{port}"
            index = len(workers)
            workers.append(address)
            node_id_map[node_id] = index
        else:
            return jsonify({"error": "Cluster full"}), 403

        # Build TF_CONFIG
        tf_config = {
            "cluster": {
                "worker": workers
            },
            "task": {"type": "worker", "index": index}
        }
        return jsonify(tf_config)

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "total_nodes": len(workers),
        "nodes": workers,
        "node_map": node_id_map
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
