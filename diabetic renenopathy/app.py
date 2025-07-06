from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2
from flask import send_from_directory
import matplotlib.pyplot as plt
import tensorflow as tf
import uuid

app = Flask(__name__)
model = load_model('model/saved_model_retinopathy.h5')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img_path, alpha=0.4):
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original_img = cv2.imread(original_img_path)
    original_img = cv2.resize(original_img, (224, 224))

    superimposed_img = cv2.addWeighted(heatmap_color, alpha, original_img, 1 - alpha, 0)
    
    # Save Grad-CAM image
    gradcam_filename = f"gradcam_{uuid.uuid4().hex}.jpg"
    gradcam_path = os.path.join('uploads', gradcam_filename)
    cv2.imwrite(gradcam_path, superimposed_img)
    return gradcam_filename

def preprocess_retinal_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (224, 224))

    green_channel = img[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)
    enhanced_img = cv2.merge([enhanced, enhanced, enhanced])

    img_array = np.expand_dims(enhanced_img, axis=0).astype(np.float32) / 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle sample image selection OR file upload
    if 'sample_path' in request.form:
        sample_path = request.form['sample_path']
        img = cv2.imread(sample_path)
        filename = os.path.basename(sample_path)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, img)
    elif 'file' in request.files:
        file = request.files['file']
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = cv2.imread(filepath)
    else:
        return "No file uploaded or sample selected"

    # Preprocess the image
    img_array = preprocess_retinal_image(filepath)

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    result = "Diabetic Retinopathy Detected" if prediction >= 0.5 else "Normal Eye"

    # Get last conv layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    # Generate Grad-CAM and save image
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    gradcam_filename = overlay_heatmap(heatmap, filepath)

    return render_template(
        'index.html',
        result=result,
        image=filename,
        gradcam_image=gradcam_filename
    )


@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
