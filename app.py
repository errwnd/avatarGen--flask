from flask import Flask, send_file, render_template_string
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import os

# Set the matplotlib backend to 'Agg' to avoid threading issues
plt.switch_backend('Agg')

app = Flask(__name__)

# Load model
model_path = 'Final.h5'  # Update this path
model = load_model(model_path)

@app.route('/')
def index():
    # HTML content with a button to trigger image generation
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Image Generator</title>
            <!-- Materialize CSS -->
            <link href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css" rel="stylesheet">
            <style>
                body {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #f0f0f0;
                }
                .container {
                    text-align: center;
                }
                .btn-custom {
                    margin-top: 20px;
                }
                img {
                    margin-top: 20px;
                    display: none;
                    max-width: 100%;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Generate Image</h1>
                <button class="btn waves-effect waves-light btn-custom" onclick="generateImage()">Generate</button>
                <br>
                <a id="downloadLink" class="btn waves-effect waves-light btn-custom" style="display:none;" download="generated_image.png">Download</a>
                <img id="generatedImage" src="" alt="Generated Image will appear here"/>
            </div>
            <!-- Materialize JS -->
            <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script>
            <script>
                function generateImage() {
                    fetch('/generate')
                        .then(response => response.blob())
                        .then(blob => {
                            const url = URL.createObjectURL(blob);
                            const img = document.getElementById('generatedImage');
                            img.src = url;
                            img.style.display = 'block';

                            const downloadLink = document.getElementById('downloadLink');
                            downloadLink.href = url;
                            downloadLink.style.display = 'inline-block';
                        });
                }
            </script>
        </body>
        </html>
    ''')

@app.route('/generate')
def generate():
    # Generate image
    noise = tf.random.normal([1, 100])
    generated_image = model(noise)

    # Plot the generated image
    fig, ax = plt.subplots(figsize=(3.6, 3.6))  # Set figure size to match the image dimensions
    ax.imshow(generated_image[0, :, :, :])
    ax.axis('off')

    # Save the plot to a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight', pad_inches=0)
    img_io.seek(0)
    plt.close(fig)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
