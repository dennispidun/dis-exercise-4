import json
import os

tf_config = {
    "cluster": {"worker": ["10.128.0.4:12345", "10.128.0.5:12345"]},
    "task": {"type": "worker", "index": 0},
}

os.environ['TF_CONFIG'] = tf_config

print(json.dumps(tf_config))
