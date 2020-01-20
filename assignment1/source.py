import keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the data

df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv').values

# Separate label and data

data = df.drop(['default.payment.next.month'], axis=1)
data = data.values

labels = df['default.payment.next.month']
labels = labels.values

# Convert the test set to a numpy array as the train and validation data

test = np.array(test)

# Preprocess data using the same scaler 

scaler = StandardScaler()
scaler.fit(data)
x_train = scaler.transform(data)
test = scaler.transform(test)


# Build the deep network

dims = x_train.shape[1] # = 23

model = Sequential()
model.add(Dense(dims, activation='relu', input_shape=(dims,)))
model.add(Dense(dims, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Compute weights

weights = compute_class_weight('balanced', np.unique(labels), labels)
weights = dict(enumerate(weights))

# Train the network

n_epochs = 20
size = 256

history = model.fit(x_train, labels, epochs=n_epochs, batch_size=size, shuffle=True, validation_split=0.3)

# Define a plot function to check accuracy and loss values

def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, network_history.history['loss'])
    plt.plot(x_plot, network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, network_history.history['accuracy'])
    plt.plot(x_plot, network_history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

# Show the plots

x_plot = list(range(1,n_epochs+1))
plot_history(history)

# Predict labels on test data

y_test = model.predict_classes(test)

# Now save the predictions to a txt file

np.savetxt("Luca_Gandolfi_score1.txt", y_test, fmt="%s")

'''
The code here is commented since no better results were achieved using PCA.
Anyway, i don't delete this to show that i've tried it too.

# Let's now apply PCA to check if we can obtain better results

pca = PCA(.95) # 95% of the variance is retained

pca.fit(x_train)
pca_train = pca.transform(x_train)
pca_test = pca.transform(test)

# Build the network

dims = pca_train.shape[1]

model = Sequential()
model.add(Dense(dims, activation='relu', input_shape=(dims,)))
model.add(Dense(dims, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train the network

n_epochs = 20
size = 256

history = model.fit(pca_train, labels, epochs=n_epochs, batch_size=size, shuffle=True, validation_split=0.3)

x_plot = list(range(1,n_epochs+1))
plot_history(history)

'''

