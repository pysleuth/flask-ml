from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import os
import numpy as np
import uuid

app = Flask(__name__)

expected = {
    "cylinder": {"min": 3, "max": 8},
    "displacement": {"min": 68.0, "max": 455.0},
    "horsepower": {"min": 46.0, "max": 230.0},
    "weight": {"min": 1613, "max": 5140},
    "acceleration": {"min": 8.0, "max": 24.8},
    "year": {"min": 70, "max": 82},
    "origin": {"min": 1, "max": 3},
}

model = load_model("mpg_model")

@app.route("/api", methods=["POST"])
def mpg_prediction():
    content = request.json
    errors = []
    for name in content:
        if name in expected:
            _min = expected[name]["min"]
            _max = expected[name]["max"]
            value = content[name]
            if value < _min or value > _max:
                errors.append(
                    f"Out of bounds: {name}: {value} ({_min}, {_max})"
                )
        else:
            errors.append(f"Unexpected field: {name}")
    
    for name in expected:
        if name not in content:
            errors.append(f"Missing value: {name}")
    
    if len(errors) == 0:
        x = np.array(
            [[content[name] for name in expected]]
        )

        prediction = model.predict(x)
        mpg = float(prediction[0])
        response = {
            "id": str(uuid.uuid4()), "mpg": mpg, "errors": errors
        }
    else:
        response = {
            "id": str(uuid.uuid4()), "errors": errors
        }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
    