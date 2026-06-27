import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import pickle

st.title('PREDICT CLINICAL DEATH EVENT RISK')

try:
    model = load_model('death_event_model.h5')
    history = pickle.load(open('death_event_history', 'rb'))
except Exception as e:
    st.error(f"Error loading model or history. Details: {e}")

def get_user_inputs():
    st.sidebar.header("Patient Clinical Factors")
    
    age = st.sidebar.slider('Age', 0, 100, 50)
    anaemia = st.sidebar.selectbox('Anaemia', (0, 1))
    creatinine_phosphokinase = st.sidebar.number_input('Creatinine Phosphokinase (mcg/L)', 0, 8000, 250)
    diabetes = st.sidebar.selectbox('Diabetes', (0, 1))
    ejection_fraction = st.sidebar.slider('Ejection Fraction (%)', 0, 100, 35)
    high_blood_pressure = st.sidebar.selectbox('High Blood Pressure', (0, 1))
    platelets = st.sidebar.number_input('Platelets (kiloplatelets/mL)', 0.0, 900000.0, 260000.0)
    serum_creatinine = st.sidebar.slider('Serum Creatinine (mg/dL)', 0.0, 10.0, 1.0)
    serum_sodium = st.sidebar.slider('Serum Sodium (mEq/L)', 100, 150, 135)
    sex = st.sidebar.selectbox('Sex (0=Female, 1=Male)', (0, 1))
    smoking = st.sidebar.selectbox('Smoking', (0, 1))
    time = st.sidebar.slider('Follow-up Period (Days)', 0, 300, 100)

    data = [[age, anaemia, creatinine_phosphokinase, diabetes, 
             ejection_fraction, high_blood_pressure, platelets, 
             serum_creatinine, serum_sodium, sex, smoking, time]]
             
    return tf.constant(data, dtype=tf.float32)

patient_data = get_user_inputs()
prediction = model.predict(patient_data, steps=1)
pred_rounded = [round(x[0]) for x in prediction]

st.subheader("Prediction Result")
if pred_rounded == [1]:
    st.error('**High Risk:** The model predicts a Death Event based on these clinical factors.')
else:
    st.success('**Lower Risk:** The model does not predict a Death Event.')

def plot_data():
    st.subheader("Model Training Metrics")
    
    loss_train = history['loss']
    loss_val = history['val_loss']
    epochs = range(1, len(loss_train) + 1)
    
    fig1, ax1 = plt.subplots()
    ax1.plot(epochs, loss_train, 'g', label='Training Loss')
    ax1.plot(epochs, loss_val, 'b', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    st.pyplot(fig1)
    
    acc_train = history['acc']
    acc_val = history['val_acc']
    
    fig2, ax2 = plt.subplots()
    ax2.plot(epochs, acc_train, 'g', label='Training Accuracy')
    ax2.plot(epochs, acc_val, 'b', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    st.pyplot(fig2)

plot_data()