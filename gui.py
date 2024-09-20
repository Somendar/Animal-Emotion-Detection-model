import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('facial_expression_model.h5')

# Define class names
class_names = ['Happy', 'Sad', 'Hungry']

# Function to preprocess and predict image
def predict_image():
    file_paths = filedialog.askopenfilenames()
    if file_paths:
        for file_path in file_paths:
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (48, 48))  # Resize to match model's expected sizing
                img = img.astype('float32') / 255.0  # Normalize pixel values
                img = np.expand_dims(img, axis=0)  # Add batch dimension
                img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale

                # Predict using the loaded model
                predictions = model.predict(img)
                predicted_class = np.argmax(predictions)
                predicted_label = class_names[predicted_class]

                # Extract animal name from file path (assuming filename contains animal name)
                animal_name = os.path.basename(file_path).split('.')[0]

                # Display image and prediction in GUI
                display_image_with_prediction(file_path, animal_name, predicted_label)

def display_image_with_prediction(image_path, animal_name, predicted_label):
    # Create a new window for displaying the image and prediction
    window = tk.Toplevel()
    window.title(f"{animal_name} - Emotion Prediction")
    window.configure(bg='#f0f8ff')  # Set background color to a soft blue

    # Add caption for animal name
    label_caption = tk.Label(window, text=f"Emotion Prediction for {animal_name}", font=("Helvetica", 18, "bold"), bg='#f0f8ff')
    label_caption.pack(pady=10)

    # Load and display the image
    img = Image.open(image_path)
    img = img.resize((400, 400))  # Resize image for better display
    photo = ImageTk.PhotoImage(img)
    label_image = tk.Label(window, image=photo, bg='#f0f8ff')
    label_image.image = photo
    label_image.pack(pady=10)

    # Display predicted emotion
    label_prediction = tk.Label(window, text=f"{animal_name} is {predicted_label}", font=("Helvetica", 16), bg='#f0f8ff')
    label_prediction.pack(pady=10)

    # Trigger notification pop-up
    messagebox.showinfo("Emotion Detected", f"{animal_name} is {predicted_label}")

    # Button to close the window
    btn_close = tk.Button(window, text="Close", command=window.destroy, font=("Helvetica", 12), bg='#4682b4', fg='white')
    btn_close.pack(pady=10)

    # Run the window's main loop
    window.mainloop()

# Create the main application window
root = tk.Tk()
root.title("Animal Emotion Prediction")
root.geometry("500x400")
root.configure(bg='#e0f7fa')  # Light background color

# Add a title label
label_title = tk.Label(root, text="Animal Emotion Prediction System", font=("Helvetica", 20, "bold"), bg='#e0f7fa')
label_title.pack(pady=20)

# Add instructions
label_instructions = tk.Label(root, text="Upload animal images to predict their emotions.", font=("Helvetica", 14), bg='#e0f7fa')
label_instructions.pack(pady=10)

# Add a button to upload images and predict
btn_upload = tk.Button(root, text="Upload Images", command=predict_image, padx=20, pady=10, font=("Helvetica", 14), bg='#00bcd4', fg='white')
btn_upload.pack(pady=30)

# Add a footer label
label_footer = tk.Label(root, text="Developed by Somendar Das", font=("Helvetica", 10), bg='#e0f7fa')
label_footer.pack(side="bottom", pady=10)

# Run the main loop
root.mainloop()
