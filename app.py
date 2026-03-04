from flask import Flask, render_template, request
from model import predict
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    image_path = None

    if request.method == "POST":

        file = request.files["file"]

        if file:

            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            prediction, confidence = predict(image_path)

    return render_template("index.html",
                       prediction=prediction,
                       confidence=confidence,
                       image=image_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
