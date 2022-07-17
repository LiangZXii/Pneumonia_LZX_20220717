from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from skimage import io
from keras.models import load_model
from PIL import Image #use PIL
import numpy as np
import cv2

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def init():
    if request.method == 'POST':
        file = request.files['file']
        print("File Received")
        filename = secure_filename(file.filename)
        print(filename)
        # Open the image form working directory
        image = Image.open(file)
        model = load_model("Pneumonia_model")
        print("Model Loaded")
        img = np.asarray(image)
        img.resize((150,150,3), refcheck=False)
        print("2")
        img = np.asarray(img, dtype="float32") #need to transfer to np to reshap
        print("2")
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) #rgb to reshape to 1,100,100,3
        print("2")
        pred=model.predict(img)
        print("2")
        return(render_template("index.html", result=str(pred)))
    else:
        return(render_template("index.html", result="WAITING"))
if __name__ == "__main__":
    app.run()

