import streamlit as st
import os
import sys
import numpy as np
import librosa
import librosa.display
import plotly.express as px
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import sound
import SessionState
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit.components.v1 as components  # Import Streamlit
import pandas as pd

def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
  
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20) 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = 862 - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 

    return mfccs


def display_results(uploaded_file, flag='uploaded'):
    if flag=='uploaded':
      audio_bytes = uploaded_file.getvalue()
    elif flag=='recorded':
        audio_bytes = open(uploaded_file, 'rb').read()
    st.subheader('Sample of the submitted audio')
    st.audio(audio_bytes, format='audio/wav')
    input_features = extract_features(uploaded_file)
    input_features = np.reshape(input_features, (*input_features.shape,1)) 
    input_features = np.reshape(input_features, (1, *input_features.shape)) 
    output = model.predict(input_features)

    healthy_prob = '{:.1%}'.format(output[0][0])
    symptomatic_prob = '{:.1%}'.format(output[0][1])
    COVID_prob = '{:.1%}'.format(output[0][2])
    d = {'Class': ['Probability'], 'COVID-19': [COVID_prob], 'Healthy': [healthy_prob], 'Symptomatic': [symptomatic_prob]}
    table = pd.DataFrame(data=d)
    table.set_index('Class', inplace=True)
    # table.reset_index(drop=True, inplace=True)
    st.table(table)    # if (healthy_prob > symptomatic_prob) and (healthy_prob > COVID_prob):
    # elif (symptomatic_prob > healthy_prob) and (COVID_prob > healthy_prob):
    #     st.subheader('you are having symptoms of COVID/other diseases')
    # else:
    #    st.subheader('oof you have COVID')

def load_model():
  model = tf.keras.models.load_model('./covid.model')
  return model

data_load_state = st.text('Loading data...')
model = load_model()

data_load_state.text("Done! (using st.cache)")

st.subheader('Please submit the cough audio')
session_state = SessionState.get(name='', path=None)

with st.form(key='uploader'):
    uploaded_file = st.file_uploader("Choose a file... (Try to keep the audio short 5-6 seconds and upload as a .wav file)")
    submit_button_upl = st.form_submit_button(label='Submit the uploaded audio')


import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 5  # Duration of recording

if st.button('Record'):
  with st.spinner(f'Recording for 5 seconds ....'):
      try:
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        write("audio1.wav", fs, myrecording)  # Save as WAV file 
        # session_state.path = sound.record()   
      except:
          pass
  st.success("Recording completed")

if st.button('Submit the recorded audio'):
    # filename = 'audio.wav'
    filename = "audio1.wav"
    display_results(filename, flag='recorded')
    os.remove(filename)

if (uploaded_file is None and submit_button_upl):
  st.subheader("Something's not right, please refresh the page and retry!")

elif uploaded_file and submit_button_upl:
    display_results(uploaded_file, flag="uploaded")
