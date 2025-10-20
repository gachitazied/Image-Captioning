"""
predict_gui.py

A simple Tkinter interface to load an image, visualize it, and use the trained
multitask model (binary + multiclass) to make predictions.

Requirements:
- tensorflow
- numpy
- pillow (for image display)
- imageio

Make sure 'multitask_photo_model.keras' is in the same folder.
"""

import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import imageio
import os

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = 'multitask_photo_model.keras'
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Painting', 'Photo', 'Schematics', 'Sketch', 'Text']

# -----------------------------
# Load the trained model
# -----------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# -----------------------------
# Utility function
# -----------------------------
def preprocess_image(image_path):
    """Load an image from disk and preprocess it for model prediction."""
    img = imageio.v2.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = tf.image.resize(img, IMAGE_SIZE).numpy()
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path):
    """Run inference on one image and return both binary and multiclass predictions."""
    img = preprocess_image(image_path)
    binary_pred, multiclass_pred = model.predict(img, verbose=0)
    binary_prob = float(binary_pred.squeeze())
    multiclass_prob = multiclass_pred.squeeze()
    multiclass_label = CLASS_NAMES[int(np.argmax(multiclass_prob))]
    return binary_prob, multiclass_label, multiclass_prob

# -----------------------------
# GUI
# -----------------------------
class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classification - Photo or Not")
        self.root.geometry("600x600")
        self.root.resizable(False, False)

        self.image_label = Label(self.root, text="No image loaded", width=60, height=20, bg="gray90")
        self.image_label.pack(pady=20)

        self.result_label = Label(self.root, text="", font=("Helvetica", 12), fg="black")
        self.result_label.pack(pady=10)

        self.load_button = Button(self.root, text="Open Image", command=self.load_image, width=20, height=2)
        self.load_button.pack(pady=5)

        self.predict_button = Button(self.root, text="Predict", command=self.run_prediction, width=20, height=2, state=tk.DISABLED)
        self.predict_button.pack(pady=5)

        self.image_path = None

    def load_image(self):
        """Let the user select an image and display it."""
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        path = filedialog.askopenfilename(title="Select an image", filetypes=filetypes)
        if not path:
            return
        self.image_path = path

        # Load and display the image
        img = Image.open(path)
        img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=tk_img, text="")
        self.image_label.image = tk_img

        # Enable prediction
        self.result_label.configure(text="")
        self.predict_button.configure(state=tk.NORMAL)

    def run_prediction(self):
        """Use the trained model to classify the selected image."""
        if not self.image_path:
            self.result_label.configure(text="Please load an image first.")
            return
        binary_prob, multiclass_label, multiclass_probs = predict_image(self.image_path)

        # Format results
        binary_text = f"Photo probability: {binary_prob:.2f}"
        is_photo = "PHOTO ✅" if binary_prob >= 0.5 else "NOT PHOTO ❌"
        top_probs = sorted(list(zip(CLASS_NAMES, multiclass_probs)), key=lambda x: x[1], reverse=True)
        top_str = "\n".join([f"{cls}: {p*100:.1f}%" for cls, p in top_probs])

        result_text = f"{is_photo}\n{binary_text}\n\nPredicted class: {multiclass_label}\n\nClass probabilities:\n{top_str}"
        self.result_label.configure(text=result_text, justify="left")

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()
