import sys
import librosa
import joblib
import numpy as np
import streamlit as st
from tensorflow.keras import models
from PIL import Image


model_filepath = "simple-cnn-ssv.h5"


def load_model(filepath):
    # load model
    new_model = models.load_model(filepath)
    #new_model1 = joblib.load(filepath)
	#model summary
    #new_model.summary()

    return new_model

def extract_feature(audio_file):
    ydat, samp_rate = librosa.load(audio_file)

    #check number of 2 second files in there
    nslices = int(np.floor(ydat.shape[0]/(2*samp_rate)))

    print(f" Number of 2 sec slices: {nslices}")

    feat_list = []
    len_samp = int(samp_rate*2.0)
    #create #nslice mel spectrograms
    st.spinner(text=f"Processing {nslices} 2  second chunks.")
    for i in range(nslices):
        S_mel = librosa.feature.melspectrogram(y=ydat[i*len_samp:(i+1)*len_samp], sr=samp_rate, n_mels=128)
        S_mel_db = librosa.amplitude_to_db(S_mel, ref=np.max)
        feat = (S_mel_db - S_mel_db.min())/(S_mel_db.max() - S_mel_db.min())
        feat_list.append(feat)
    return feat_list

def detect_deepfake(model_path, feat_list):
    
    model = load_model(model_path)
    pred_list = np.zeros(len(feat_list))

    for i in range(len(feat_list)):
        pred_list[i] = model.predict(np.expand_dims(feat_list[i], 0))
    
    print(f" predicted values: {pred_list}")

    pred_list[pred_list < 0.5] = 0
    pred_list[pred_list >= 0.5] = 1
    
    print(pred_list)

    if pred_list.any()  == 0:
        return 0
    else:
        return 1


def main():
    st.title("Welcome to the AudioShield")
    st.text("Audioshield is a software designed to detect deepfake audio samples")
    image = Image.open("deepfake-logo.png")
    st.image(image)


    file_uploaded = st.file_uploader("Choose the Audio File", type=["flac","wav","mp3"])

    if file_uploaded is not None:

        feature_list = extract_feature(file_uploaded)
        output = detect_deepfake(model_filepath, feature_list)
        
        if output == 0:
            result = 'Fake Audio'
        else:
            result = 'Genuine Audio'
        st.write(f"The audio file is {result}")


if __name__ == '__main__':
    main()
    
	#file_uploaded = sys.argv[1]
    #feature_list = extract_feature(file_uploaded)
    #output = detect_deepfake(model_filepath, feature_list)

    #print(outputt)
