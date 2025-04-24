from flask import Flask, render_template, request, redirect, url_for
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption_from_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save uploaded image
            filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            with open(filepath, "rb") as f:
                image_bytes = f.read()
            
            caption = generate_caption_from_bytes(image_bytes)
            return render_template('index.html', image_url = filename, caption=caption)

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
