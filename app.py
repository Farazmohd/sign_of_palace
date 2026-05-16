import os
import cv2
import time
import numpy as np
import tensorflow as tf
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, render_template, request, flash, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
from descriptions import descriptions  # External class descriptions
import config  # Separate credentials for this project

# === CONFIGURATION ===
UPLOAD_FOLDER = "static/uploads"
STATIC_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'mp4', 'webm'}
NUM_FRAMES = 8
IMG_SIZE = 112

# === INITIALIZE FLASK APP ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "royal_secret_key_for_flash_messages"

# Expose random to templates for cache busting
import random
app.jinja_env.globals.update(random=random)

# Email Configuration (imported from config.py)
EMAIL_USER = config.EMAIL_USER
EMAIL_PASS = config.EMAIL_PASS
TARGET_EMAIL = config.TARGET_EMAIL

# === LOAD MODEL AND CLASS NAMES ===
model = tf.keras.models.load_model("sign_language_model_augmented.keras")
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# === UTILITY FUNCTIONS ===
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        all_frames.append(frame)
    
    cap.release()
    total_frames = len(all_frames)

    if total_frames < NUM_FRAMES:
        return None, None

    # Pick 8 equally spaced frames from the captured list
    idxs = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)
    frames = []
    
    for idx in idxs:
        frame = all_frames[idx]
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)

    if len(frames) < NUM_FRAMES:
        return None, None

    # IMAGE ANIMATION SYSTEM: Save all 8 frames as JPEGs
    frame_filenames = []
    timestamp = int(time.time())
    for idx, frame in enumerate(frames):
        fname = f"frame_{timestamp}_{idx}.jpg"
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        cv2.imwrite(fpath, frame.astype(np.uint8))
        frame_filenames.append(fname)

    frames_norm = np.array(frames, dtype=np.float32) / 255.0
    return frames_norm, frame_filenames

# === ROUTES ===

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "video" not in request.files:
            return render_template("predict.html", error="No file uploaded.")

        file = request.files["video"]
        if file.filename == "":
            return render_template("predict.html", error="No file selected.")

        if file and allowed_file(file.filename):
            # Generate unique filename using timestamp
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"gesture_{int(time.time())}.{ext}"
            
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(STATIC_FOLDER, exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            frames, frame_filenames = extract_frames(filepath)
            if frames is None:
                return render_template("predict.html", error="Video must have at least 8 frames.")

            pred = model.predict(np.expand_dims(frames, axis=0))
            class_idx = np.argmax(pred)
            predicted_class = class_names[class_idx]

            # Description from separate file
            class_key = predicted_class.lower().strip()
            description = descriptions.get(class_key, "No description available.")

            # Save middle frame
            middle_frame = frames[NUM_FRAMES // 2] * 255.0  # Unnormalize
            image_filename = filename.rsplit('.', 1)[0] + "_frame.jpg"
            image_path = os.path.join(STATIC_FOLDER, image_filename)
            cv2.imwrite(image_path, middle_frame.astype(np.uint8))

            # Store in session and redirect (PRG Pattern)
            session['prediction_result'] = {
                'prediction': predicted_class,
                'description': description,
                'image_file': image_filename,
                'frame_files': frame_filenames
            }
            return redirect(url_for('predict'))

    # GET Request: check session for results
    result = session.pop('prediction_result', None)
    if result:
        return render_template("predict.html", **result)
    
    return render_template("predict.html")

@app.route("/about-palace")
def about_palace():
    return render_template("about_palace.html")

@app.route("/about-project")
def about_project():
    return render_template("about_project.html")

@app.route('/video_feed/<filename>')
def video_feed(filename):
    mimetype = 'video/mp4' if filename.endswith('.mp4') else 'video/webm'
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename, mimetype=mimetype)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        try:
            # Create the email
            msg = MIMEMultipart()
            msg['From'] = EMAIL_USER
            msg['To'] = TARGET_EMAIL
            msg['Subject'] = f"New Message from {name} (Signs of the Palace)"

            body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
            msg.attach(MIMEText(body, 'plain'))

            # Send the email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            text = msg.as_string()
            server.sendmail(EMAIL_USER, TARGET_EMAIL, text)
            server.quit()

            return {"status": "success", "message": "Your message has been sent!"}, 200
        except Exception as e:
            print(f"Error sending email: {e}")
            return {"status": "error", "message": str(e)}, 500

    return render_template("contact.html")

# === DUMMY ROUTES TO SUPPRESS EXTERNAL POLLING LOGS ===
@app.route("/api/notifications", methods=["GET"])
def dummy_notifications():
    return {"status": "ok", "notifications": []}, 200

# === RUN APP ===
if __name__ == "__main__":
    app.run(debug=True)
