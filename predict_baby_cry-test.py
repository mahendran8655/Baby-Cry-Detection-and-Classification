import joblib
import librosa
import numpy as np

# Load the model
model = joblib.load('baby_cry_classifier.pkl')

# Load new audio file (change the path accordingly)
file_path = 'baby_dataset_3/hungry/0a983cd2-0078-4698-a048-99ac01eb167a-1433917038889-1.7-f-04-hu.wav'
y, sr = librosa.load(file_path, duration=5)

# Extract MFCC features (make sure it matches training format)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_processed = np.mean(mfcc.T, axis=0).reshape(1, -1)

# Make prediction
prediction = model.predict(mfcc_processed)
print("Predicted cry type:", prediction[0])
