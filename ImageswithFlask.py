from flask import Flask, request, jsonify, url_for, render_template
import uuid
import os

from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNet
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions

ALLOWED_EXTENSION = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3

def allowed_file(filename):
    return (
        ("." in filename) and (filename.rsplit(".", 1)[1] in ALLOWED_EXTENSION)
    )

app = Flask(__name__)
model = MobileNet(weights="imagenet", include_top=True)

@app.route('/index')
def index():
    return render_template("ImageML.html")

@app.route("/api/image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return render_template(
            "ImageML.html",
            prediction="No posted image. Please send image!"
        )
    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "ImageML.html", prediction="You did not select an image!"
        )
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(f"***{filename}")
        x = []
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(BytesIO(file.read()))
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = model.predict(x)
        lst = decode_predictions(pred, top=3)

        items = [
            {"name": item[1], "prob": float(item[2])}
            for item in lst[0]
        ]

        response = {"pred": items}
        return render_template(
            "ImageML.html",
            prediction=f"Top predictions: {response}"
        )
    else:
        return render_template(
            "ImageML.html", prediction="Invalid File extension"
        )

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
