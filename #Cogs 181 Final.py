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
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig("Corr Heatmap.jpg")

x = data[["label", "tempo"]]
f,ax = plt.subplots(figsize=(10,5))
sns.boxplot(x="label", y="tempo", data=x, palette='husl')
plt.xlabel("Genre")
plt.ylabel("BPM")
plt.savefig("BPM Boxplot.jpg")
