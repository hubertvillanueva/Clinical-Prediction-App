import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT'].values.astype(np.float32)

input_dim = X.shape[1] 
X = X.values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

model = keras.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(input_dim,)),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']) 

history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

model.save('death_event_model.h5')

with open('death_event_history', 'wb') as file:
    pickle.dump(history.history, file)