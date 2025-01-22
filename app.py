from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import cv2
import numpy as np
from PIL import Image
import pytesseract

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the fine-tuned model and tokenizer
model_path = "./gpt2-pneumonia-diagnosis"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_text(prompt, max_length=300):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, temperature=0.7, top_k=50,
                            top_p=0.95)
    return tokenizer.decode(output[0], skip_special_tokens=False)


def analyze_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to preprocess the image
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Perform text extraction
    text = pytesseract.image_to_string(threshold)

    # Basic image analysis (you may want to expand this with more sophisticated techniques)
    brightness = np.mean(gray)
    contrast = np.std(gray)

    description = f"The chest X-ray image has a brightness of {brightness:.2f} and contrast of {contrast:.2f}. "
    description += f"Text extracted from the image: {text}"

    return description


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        file = request.files['file']
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No file selected for uploading'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Analyze the uploaded image
            image_description = analyze_image(filepath)

            # Generate pneumonia report using the LLM model
            prompt = f"<image>Chest X-ray\n<description>{image_description}\n"
            report = generate_text(prompt, max_length=500)

            return jsonify({'report': report}), 200
        else:
            return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)

