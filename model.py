import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tqdm import tqdm

# Function to load images from the dataset
def load_images(folder_path):
    images = []
    labels = []
    emotions = os.listdir(folder_path)

    for emotion in emotions:
        emotion_path = os.path.join(folder_path, emotion)
        for filename in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (48, 48))  # Resize the image to the required size
            images.append(img)
            labels.append(emotion)

    return np.array(images), np.array(labels)

# Load images and labels
train_folder = 'train'
X_train, y_train = load_images(train_folder)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(y_train)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with progress bar
epochs = 10
batch_size = 32
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    for i in tqdm(range(0, len(X_train), batch_size)):
        X_batch = X_train[i:i + batch_size].reshape(-1, 48, 48, 1)
        y_batch = y_train_encoded[i:i + batch_size]
        model.train_on_batch(X_batch, y_batch)

# Save the model
model.save('emotion_model.h5')
