# Animal Emotion Detection

## Overview

The **Animal Emotion Detection** project is designed to identify the emotions of animals in images, such as **happy**, **sad**, or **hungry**. The model not only detects individual animals' emotions but can also process images containing multiple animals, marking the emotion for each one. Additionally, the system provides a notification feature that displays each animal's name and its detected emotion, making it useful for animal behavior analysis.

## Key Features

- **Single Animal Emotion Detection**: Identifies whether an animal is happy, sad, or hungry from an image.
- **Group Emotion Detection**: Analyzes images containing multiple animals, marking each animal's detected emotion.
- **Notification System**: Triggers a pop-up notification in the format: `"Animal Name + Emotion"`.
- **User-Friendly Interface**: A simple GUI allows users to upload animal images and see the predictions instantly.

## Tools and Technologies Used

- **Programming Language**: Python 3.x
- **Framework**: TensorFlow and Keras (for deep learning)
- **Frontend**: Streamlit (for GUI development)
- **Libraries**:
  - OpenCV: For image preprocessing.
  - Pillow: For image manipulation.
  - Matplotlib: For data visualization.
  - Scikit-learn: For splitting the dataset and evaluation.
  - NumPy & Pandas: For data handling and processing.

## Methodology

### 1. Data Collection
- Collected a dataset of animal images labeled with emotions: **happy**, **sad**, and **hungry**. The dataset included a variety of species to generalize the model.
  
### 2. Data Preprocessing
- **Image Resizing**: All images were resized to a standard dimension.
- **Normalization**: Pixel values were normalized for faster and more efficient model training.
- **Data Augmentation**: Techniques like flipping, rotation, and zooming were applied to increase dataset variety and prevent overfitting.

### 3. Model Architecture
- **Convolutional Neural Network (CNN)**:
  - Consists of several convolutional and pooling layers to extract important features from the images.
  - Fully connected layers at the end for classification into **happy**, **sad**, or **hungry**.
  
- **Loss Function**: Categorical Crossentropy (as it's a multi-class classification problem).
- **Optimizer**: Adam (for adaptive learning rate optimization).

### 4. Model Training
- **Dataset Split**: The dataset was split into training, validation, and test sets (80:10:10).
- **Epochs and Batch Size**: The model was trained over several epochs with a batch size of 32.
- **Evaluation**: Accuracy and loss metrics were used to evaluate the model’s performance.

### 5. Notification System
- After detecting the emotions, the system triggers a pop-up notification for each animal.


