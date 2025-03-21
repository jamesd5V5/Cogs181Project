#Cogs 181 Final
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import librosa
import librosa.display
import IPython.display as ipd
import warnings
warnings.filterwarnings('ignore')
import os
import sklearn.preprocessing

data_path = os.getcwd()
print(list(os.listdir(f'{data_path}/genres_original/')))

n_fft = 2048
hop_length = 512

rock_path = os.path.join(data_path, "genres_original", "rock", "rock.00016.wav")
y, sr = librosa.load(rock_path)
y, _ = librosa.effects.trim(y)
S = librosa.feature.melspectrogram(y=y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)
plt.figure(figsize=(16,6))
librosa.display.specshow(S_DB, sr=sr,hop_length=hop_length, x_axis='time',y_axis='log',cmap='magma')
plt.colorbar()
plt.title("Mel Spectrogram", fontsize = 23)
plt.savefig("MelSpectorgam")

data = pd.read_csv(f'{data_path}/features_30_sec.csv')

spike_cols = [col for col in data.columns if 'mean' in col]
corr = data[spike_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize=(18, 18))
cmap = sns.color_palette("coolwarm", as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt='.2f',vmax=1, vmin=-1, center=0, square=True, linewidths=0.7,cbar_kws={"shrink": 0.6, "orientation": "vertical"})
plt.title('Mean Feature Correlation Heatmap', fontsize=28, fontweight='bold', pad=20)
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12)
plt.grid(visible=True, linewidth=0.3)
plt.savefig("Enhanced_Corr_Heatmap.jpg", dpi=300, bbox_inches='tight')

#CNN Classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

print(data.head())

y = data['label']
X = data.drop(columns=['label', 'filename'])

le = LabelEncoder()
y = le.fit_transform(y)

scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

X_train_list, X_test_list = [], []
y_train_list, y_test_list = [], []

# Split each genre separately
for genre in data['label'].unique():
    genre_indices = data[data['label'] == genre].index
    X_genre = X.loc[genre_indices]
    y_genre = pd.Series(y[genre_indices], index=genre_indices)

    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_genre, y_genre, test_size=0.3)

    X_train_list.append(X_train_g)
    X_test_list.append(X_test_g)
    y_train_list.append(pd.Series(y_train_g, index=y_train_g.index))
    y_test_list.append(pd.Series(y_test_g, index=y_test_g.index))

X_train = pd.concat(X_train_list)
X_test = pd.concat(X_test_list)
y_train = pd.concat(y_train_list)
y_test = pd.concat(y_test_list)

overlap_indicesX = X_train.index.intersection(X_test.index)
overlap_indicesY = y_train.index.intersection(y_test.index)

from tensorflow import keras
from tensorflow.keras import layers

X_train = X_train.values.reshape(-1, X_train.shape[1], 1)
X_test = X_test.values.reshape(-1, X_test.shape[1], 1)

model = keras.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.BatchNormalization(),
    #layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(128, kernel_size=3, use_bias=False), #likes activation after batch normalization
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Conv1D(128, kernel_size=2, activation='relu'),
    #layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.6), #0.3 increase dropout, had some effect
    layers.Dense(len(np.unique(y)), activation='softmax')
])

#Lower Learning_rate 0.01, 0.005
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002), loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# Train model
#Batch size 16, 32, 64

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[early_stop])

plt.figure(figsize=(16,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Training Loss.jpg")

plt.figure(figsize=(16,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("Training Accuracy.jpg")

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.5f}")

preds = model.predict(X_test)
preds = np.argmax(preds, axis=1)

#Confusion Matrix
confusion_matr = confusion_matrix(y_test, preds)
plt.figure(figsize=(10,5))
sns.heatmap(confusion_matr, cmap="Blues", annot=True, xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"], yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"])
plt.savefig("Confusion Matrix.jpg")

import random
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv(f'{data_path}/features_30_sec.csv')
song_names = data['filename']
#719
extractor = keras.Model(inputs=model.inputs, outputs=model.get_layer('flatten').output)

cnn_features = extractor.predict(X.values.reshape(-1, X.shape[1], 1))
random_index = random.randint(0, X.shape[0] - 1)
random_song = cnn_features[random_index].reshape(1, -1)

similarities = cosine_similarity(random_song, cnn_features)

# Get top 5 most similar songs
similar_indices = np.argsort(similarities[0])[::-1][1:6]

print(f"Randomly Selected Song: {song_names.iloc[random_index]}\n")
print("Top 5 Most Similar Songs:")
for i, idx in enumerate(similar_indices, start=1):
    print(f"{i}. {song_names.iloc[idx]} - Similarity Score: {similarities[0][idx]:.4f}")