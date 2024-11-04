from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

def get_top_colors(image_path, num_colors=10):
    # Load image and convert it to RGB
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.reshape((img_rgb.shape[0] * img_rgb.shape[1], 3))
    
    # Use k-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img_rgb)
    
    # Get colors and their percentages
    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_)
    
    # Sort colors by frequency
    sorted_colors = colors[np.argsort(-counts)]
    return sorted_colors.astype(int)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save the uploaded file
        uploaded_file = request.files["image"]
        if uploaded_file.filename != "":
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
            uploaded_file.save(file_path)
            
            # Get the top 10 colors
            top_colors = get_top_colors(file_path, num_colors=10)
            return render_template("index.html", colors=top_colors, image_path=file_path)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
