import json
import sys

# Sample information replace with real data later
json.dump(
    {
        "problems": [
            {"benchmark": "A", "dataset": "d1", "tags": ["dense", "test"]},
            {"benchmark": "A", "dataset": "d2", "tags": ["tensor", "trace"]},
        ],
        "series": {
            "sparse": [[0, 1.0], [1, 2.0]],
            "numpy": [[1, 1.0], [0, 3.0]],
        },
        "xMax": 10,
    },
    sys.stdout,
)
