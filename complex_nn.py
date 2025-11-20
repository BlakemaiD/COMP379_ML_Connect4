import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
rows = [1, 2, 3, 4, 5, 6]
board_columns = [col + str(row) for col in columns for row in rows]

column_names = board_columns + ['Class']

df = pd.read_csv(
        'connect+4/connect-4.data',  # The uncompressed file name
        header=None,       # No header row in the file
        names=column_names # Assign the generated column names
    )
#load
np_array = df.to_numpy()
X = np_array[:, :-1]
y = np_array[:, -1]

one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = one_hot_encoder.fit_transform(X)
y_reshaped = y.reshape(-1,1)
y_encoded = one_hot_encoder.fit_transform(y_reshaped)

#keras model

network = models.Sequential()
network.add(layers.Dense(42, activation='relu', input_shape=(126,)))
network.add(layers.Dense(42, activation='relu'))
network.add(layers.Dense(42, activation='relu'))
network.add(layers.Dense(42, activation='relu'))
network.add(layers.Dense(3, activation='softmax'))

network.summary()

network.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'recall', 'precision'])

#60/20/20
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=.2)

network.fit(X_train, y_train, epochs=40, batch_size=128, validation_split=.25)

results = network.evaluate(X_test, y_test, batch_size=128)
print("test loss, accuracy, recall, and precision", results)
#50/25/25
network = models.Sequential()
network.add(layers.Dense(42, activation='relu', input_shape=(126,)))
network.add(layers.Dense(42, activation='relu'))
network.add(layers.Dense(42, activation='relu'))
network.add(layers.Dense(42, activation='relu'))
network.add(layers.Dense(3, activation='softmax'))

network.summary()

network.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'recall', 'precision'])
    
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=.25)

network.fit(X_train, y_train, epochs=40, batch_size=128, validation_split=.33)

results = network.evaluate(X_test, y_test, batch_size=128)
print("test accuracy, recall, and precision", results)
#20/20/60
network = models.Sequential()
network.add(layers.Dense(42, activation='relu', input_shape=(126,)))
network.add(layers.Dense(42, activation='relu'))
network.add(layers.Dense(42, activation='relu'))
network.add(layers.Dense(42, activation='relu'))
network.add(layers.Dense(3, activation='softmax'))

network.summary()

network.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy', 'recall', 'precision'])
    
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=.6)

network.fit(X_train, y_train, epochs=40, batch_size=128, validation_split=.5)

results = network.evaluate(X_test, y_test, batch_size=128)
print("test accuracy, recall, and precision", results)