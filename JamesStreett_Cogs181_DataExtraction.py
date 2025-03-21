import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    y, _ = librosa.effects.trim(y)
    
    features = {
        'filename': os.path.basename(file_path),
        'length': len(y),
        'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'chroma_stft_var': np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
        'rms_mean': np.mean(librosa.feature.rms(y=y)),
        'rms_var': np.var(librosa.feature.rms(y=y)),
        'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_centroid_var': np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth_mean': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'spectral_bandwidth_var': np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'rolloff_var': np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'zero_crossing_rate_mean': np.mean(librosa.feature.zero_crossing_rate(y=y)),
        'zero_crossing_rate_var': np.var(librosa.feature.zero_crossing_rate(y=y)),
        'harmony_mean': np.mean(librosa.effects.hpss(y)[0]),
        'harmony_var': np.var(librosa.effects.hpss(y)[0]),
        'perceptr_mean': np.mean(librosa.effects.hpss(y)[1]),
        'perceptr_var': np.var(librosa.effects.hpss(y)[1]),
        'tempo': librosa.beat.tempo(y=y, sr=sr)[0]
    }
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
        features[f'mfcc{i+1}_var'] = np.var(mfccs[i])
    
    return features

data_path = "genres_original"
files = []
for root, _, filenames in os.walk(data_path):
    for file in filenames:
        if file.endswith(".wav"):
            files.append(os.path.join(root, file))

dataset = []
for file in files:
    features = extract_features(file)
    print("Completed: ", file)
    features['label'] = os.path.basename(os.path.dirname(file))
    dataset.append(features)

df = pd.DataFrame(dataset)
df.to_csv("audio_features.csv", index=False)
print("CSV file saved successfully!")