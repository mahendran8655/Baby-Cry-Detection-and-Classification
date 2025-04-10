# Baby Cry Detection and Classification

## Overview
Classifies baby cries using audio features (MFCC) into categories like hunger, pain, tiredness, and discomfort.

## Files
- baby_cry_detection.py — Main script to train and evaluate the model
- Baby_Cry_Detection_Presentation.pptx — Project slides
- README.txt — This file

## Libraries Required
- pandas
- numpy
- librosa
- scikit-learn
- joblib

To install them:
pip install pandas numpy librosa scikit-learn joblib

## Dataset
Download: https://www.kaggle.com/datasets/paultimothymooney/baby-cry
Extract it into a folder named: `baby_cry_dataset/`

or
using this script generate_dummy_baby_cries.py, generate the dummy samples

or
You can source these free sound clips from:

Freesound.org → search "baby cry", "baby hungry", etc.


## Output
The trained model is saved as: `baby_cry_classifier.pkl`
