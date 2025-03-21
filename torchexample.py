#Cogs 181 Final
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# Librosa (the mother of audio files)
import librosa
import librosa.display
import IPython.display as ipd
import warnings
warnings.filterwarnings('ignore')

import os

import sklearn.preprocessing
data_path = os.getcwd()
print(list(os.listdir(f'{data_path}/genres_original/')))

#file_path = os.path.join(data_path, "genres_original", "disco", "disco.00022.wav")
#y, sr = librosa.load(file_path)

#audio_file, _ = librosa.effects.trim(y)
#print('Audio File shape:', np.shape(audio_file))

#plt.figure(figsize= (16,6))
#librosa.display.waveshow(y = audio_file, sr = sr)
#plt.title("Sound Wave Example")

n_fft = 2048
hop_length = 512
#D = np.abs(librosa.stft(audio_file, n_fft=n_fft, hop_length=hop_length))
#print('Shape of D: ', np.shape(D))

#plt.figure(figsize= (16,6))
#plt.plot(D)
#plt.show()

rock_path = os.path.join(data_path, "genres_original", "rock", "rock.00016.wav")
y, sr = librosa.load(rock_path)
y, _ = librosa.effects.trim(y)
S = librosa.feature.melspectrogram(y=y, sr=sr)
S_DB = librosa.amplitude_to_db(S, ref=np.max)
#plt.figure(figsize=(16,6))
#librosa.display.specshow(S_DB, sr=sr,hop_length=hop_length, x_axis='time',y_axis='log',cmap='cool')
#plt.colorbar()

zero_crossings = librosa.zero_crossings(y, pad=False)
print(sum(zero_crossings))

y_harm, y_perc = librosa.effects.hpss(y)
#plt.figure(figsize = (16, 6))
#plt.plot(y_harm, color = '#A300F9');
#plt.plot(y_perc, color = '#FFB100');

tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(tempo)

def sklearnNormalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

spectral_centroids = librosa.feature.spectral_centroid(y=y,sr=sr)[0]
print('Centroid:', spectral_centroids, '\n')
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
print('t:', t)

#plt.figure(figsize=(16,6))
#librosa.display.waveshow(y=y, sr=sr, alpha=0.4, color = '#A300F9')
#plt.plot(t, sklearnNormalize(spectral_centroids), color='#FFB100')

spectral_rolloff = librosa.feature.spectral_rolloff(y=y,sr=sr)[0]
#plt.figure(figsize=(16,6))
##librosa.display.waveshow(y=y, sr=sr, alpha=0.4, color = '#A300F9')
#plt.plot(t, sklearnNormalize(spectral_rolloff), color='#FFB100')

mfccs = librosa.feature.mfcc(y=y,sr=sr)
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print('Mean, Varaince: ', mfccs.mean(), ', ', mfccs.var())
#plt.figure(figsize=(16,6))
#librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool')


hop_length = 5000
chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
#plt.figure(figsize=(10,5))
#librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')

#The csv file contains all the audio features above for all audio clips
data = pd.read_csv(f'{data_path}/features_30_sec.csv')
print(data.head())

spike_cols = [col for col in data.columns if 'mean' in col]
corr = data[spike_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool_))
f,ax = plt.subplots(figsize=(10,5))
cmap = sns.diverging_palette(0,25,as_cmap=True, s=90, l=45, n=5)
sns.heatmap(corr, mask=mask, cmap=cmap,vmax=0.3,center=0,square=True,linewidths=0.5,cbar_kws={"shrink": 0.5})
#plt.xticks(fontsize=10)
#plt.yticks(fontsize=10)
#plt.savefig("Corr Heatmap.jpg")

#x = data[["label", "tempo"]]
#f,ax = plt.subplots(figsize=(10,5))
#sns.boxplot(x="label", y="tempo", data=x, palette='husl')
#plt.xlabel("Genre")
#plt.ylabel("BPM")
#plt.savefig("BPM Boxplot.jpg")

#Hello
from sklearn import preprocessing

data = data.iloc[0:, 1:]
y = data['label']
X = data.loc[:, data.columns != 'label']

#Normalize X
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns=cols)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDF = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
finalDF = pd.concat([principalDF, y], axis = 1)
print(pca.explained_variance_ratio_)

plt.figure(figsize=(10,6))
sns.scatterplot(x='principal component 1',y='principal component 2', data=finalDF,hue="label", alpha=0.7, s=100)
plt.title('PCA on Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 2", fontsize = 15)
plt.savefig("PCA Scatter.jpg")

#ML Classification
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree, plot_importance

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

data = pd.read_csv(f'{data_path}/features_3_sec.csv')
print(data.head())

y = data['label']
X = data.drop(columns=['label', 'filename'])

le = LabelEncoder()
y = le.fit_transform(y)

#normalizw X
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

X_train_list, X_test_list = [], []
y_train_list, y_test_list = [], []

# Split each genre separately
for genre in data['label'].unique():
    genre_indices = data[data['label'] == genre].index  # Get indices of this genre
    X_genre = X.loc[genre_indices]
    y_genre = pd.Series(y[genre_indices], index=genre_indices)  # Convert y to Series

    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_genre, y_genre, test_size=0.3)

    # Store in lists
    X_train_list.append(X_train_g)
    X_test_list.append(X_test_g)
    y_train_list.append(pd.Series(y_train_g, index=y_train_g.index))  # Ensure Series
    y_test_list.append(pd.Series(y_test_g, index=y_test_g.index))  # Ensure Series

# Concatenate all genre-specific splits
X_train = pd.concat(X_train_list)
X_test = pd.concat(X_test_list)
y_train = pd.concat(y_train_list)
y_test = pd.concat(y_test_list)

overlap_indicesX = X_train.index.intersection(X_test.index)
overlap_indicesY = y_train.index.intersection(y_test.index)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Reshape your data for PyTorch input
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(2)  # Add channel dimension
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(2)  # Add channel dimension
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoader for batching
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define the CNN model using PyTorch
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128 * (X_train.shape[1] // 4), 128)  # Adjust the size of the Linear layer
        self.fc2 = nn.Linear(128, len(np.unique(y)))  # Number of classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layer
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, p=0.3, train=self.training)  # Apply dropout during training
        x = self.fc2(x)
        return x

# Initialize the model
model = AudioCNN()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 30
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        running_loss += loss.item()

    # Track training loss
    train_losses.append(running_loss / len(train_loader))

    # Evaluate the model on the test data
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():  # No gradient computation during evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(correct / total)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")

# Plotting the training and test losses/accuracy
plt.figure(figsize=(10,6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.title('Test Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluating the model's performance
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, preds = torch.max(outputs, 1)

# Confusion Matrix
confusion_matr = confusion_matrix(y_test, preds.numpy())
plt.figure(figsize=(10,5))
sns.heatmap(confusion_matr, cmap="Blues", annot=True, xticklabels = le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.savefig("Confusion Matrix.jpg")
