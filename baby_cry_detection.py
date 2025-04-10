# Baby Cry Detection using Random Forest
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

data = []
labels = []
base_path = "baby_cry_dataset"  # Make sure your folder contains subfolders named after labels

for label in os.listdir(base_path):
    folder = os.path.join(base_path, label)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        features = extract_features(file_path)
        data.append(features)
        labels.append(label)

df = pd.DataFrame(data)
df['label'] = labels

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, 'baby_cry_classifier.pkl')
