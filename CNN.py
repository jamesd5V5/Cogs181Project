import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data_path = os.getcwd()
data = pd.read_csv(f'{data_path}/features_3_sec.csv')

# Prepare labels
y = data['label']
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)  # One-hot encoding for classification

# Prepare spectrograms as input data
X = []
for filename in data['filename']:
    file_path = os.path.join(data_path, "genres_original", filename.split('.')[0], filename)
    y_audio, sr = librosa.load(file_path, sr=None)
    y_audio, _ = librosa.effects.trim(y_audio)
    S = librosa.feature.melspectrogram(y=y_audio, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    S_DB = np.expand_dims(S_DB, axis=-1)  # Add channel dimension
    X.append(S_DB)

X = np.array(X)
X = np.expand_dims(X, axis=-1)  # Ensure correct shape for CNN

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)  # Normalize per sample

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')
