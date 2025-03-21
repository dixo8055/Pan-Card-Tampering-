from flask import Flask, request, render_template, send_file
import os
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image

app = Flask(__name__)

# Config paths
app.config['INITIAL_FILE_UPLOADS'] = 'static/uploads'
app.config['EXISTING_FILE'] = 'static/original'
app.config['GENERATED_FILE'] = 'static/generated'

# Ensure directories exist
os.makedirs(app.config['INITIAL_FILE_UPLOADS'], exist_ok=True)
os.makedirs(app.config['EXISTING_FILE'], exist_ok=True)
os.makedirs(app.config['GENERATED_FILE'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    if request.method == "POST":
        file_upload = request.files['file_upload']
        filename = file_upload.filename

        if filename == '':
            return render_template("index.html", error="No file selected")

        try:
            # Resize and save uploaded image
            uploaded_image = Image.open(file_upload).resize((250, 160))
            uploaded_image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.jpg'))

            # Load and resize original image
            original_image = Image.open(os.path.join(app.config['EXISTING_FILE'], 'image.jpg')).resize((250, 160))
            original_image.save(os.path.join(app.config['EXISTING_FILE'], 'image.jpg'))

            # Read images with OpenCV
            original_image = cv2.imread(os.path.join(app.config['EXISTING_FILE'], 'image.jpg'))
            uploaded_image = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.jpg'))

            # Convert to grayscale
            original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            uploaded_gray = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2GRAY)

            # SSIM comparison
            (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
            diff = (diff * 255).astype("uint8")

            # Threshold and contours
            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # Draw contours on both images
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(uploaded_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Save processed images
            cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_original.jpg'), original_image)
            cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_uploaded.jpg'), uploaded_image)
            cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_diff.jpg'), diff)
            cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_thresh.jpg'), thresh)

            return render_template('index.html', pred=f"{round(score * 100, 2)}% correct")

        except Exception as e:
            return render_template("index.html", error=str(e))

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(app.config['GENERATED_FILE'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)