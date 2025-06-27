from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("healthy_vs_rotten.h5")
classes = ['Coccidiosis', 'Healthy', 'Salmonella', 'New Castle Disease']

# 🧠 Prediction function
def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)[0]
    return classes[np.argmax(pred)]

# 🏠 Home page
@app.route('/')
def index():
    return render_template('index.html')

# 🔍 Prediction page (GET for form, POST for file upload)
@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400
        path = os.path.join('static/uploads', file.filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file.save(path)
        result = predict(path)
        return render_template('predict.html', prediction=result, img_path=path)
    return render_template('predict.html')

# 📩 Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
