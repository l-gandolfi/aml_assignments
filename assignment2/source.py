import numpy as np
import pandas as pd

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.utils import to_categorical, np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, average_precision_score, precision_recall_curve


from itertools import cycle
from collections import Counter


# load the data
x_train = pickle.load(open("x_train.obj","rb"))
x_test = pickle.load(open("x_test.obj","rb"))
y_train = pickle.load(open("y_train.obj","rb"))

# reshape in [0,1] values
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# split train and validation
x_train_AE, x_val_AE, y_train_AE, y_val_AE = train_test_split(x_train, y_train, test_size=0.2,random_state=42)

'''
Autoencoder
'''

# create layers
encoding_dim = 32 
input_img = Input(shape=(784,))

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)


# fit the model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

fBestModel = 'best_model.h5' 
best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)

history = autoencoder.fit(x_train_AE, x_train_AE, epochs=200, batch_size=64, shuffle=True, validation_data=(x_val_AE, x_val_AE), callbacks=[best_model])

autoencoder.load_weights('best_model.h5')

# plot the loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(200)
plt.figure()
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# define the encoder model
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim, ))

# define the decoder model
decoder_layer1 = autoencoder.layers[-3]
decoder_layer2 = autoencoder.layers[-2]
decoder_layer3 = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))

# encode and decode the test set - we will also use this later.
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# plot original vs reconstructed input
n = 12  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Now i've obtained a feature extraction tool, i can use my encoder to obtain a new train
# and use this train to predict

encoded_input = encoder.predict(x_train)
print(encoded_input.shape)
#np.savetxt("encoded_input.csv", encoded_input, delimiter=",")
#np.savetxt("encoded_test.csv", encoded_imgs, delimiter=",")
x_test = encoded_imgs

# plot a t-sne representation
def tsne_plot(x1, y1):
	print('Building t-SNE...')
	tsne = TSNE(n_components=2, random_state=0)
	X_t = tsne.fit_transform(x1)
	plt.figure(figsize=(12, 8))
	plt.scatter(X_t[np.where(y1 == 16), 0], X_t[np.where(y1 == 16), 1], marker='o', color='g', 	linewidth='1', alpha=0.8, label='P')
	plt.scatter(X_t[np.where(y1 == 17), 0], X_t[np.where(y1 == 17), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='Q')
	plt.scatter(X_t[np.where(y1 == 18), 0], X_t[np.where(y1 == 18), 1], marker='o', color='b', linewidth='1', alpha=0.8, label='R')
	plt.scatter(X_t[np.where(y1 == 19), 0], X_t[np.where(y1 == 19), 1], marker='o', color='black', linewidth='1', alpha=0.8, label='S')
	plt.scatter(X_t[np.where(y1 == 20), 0], X_t[np.where(y1 == 20), 1], marker='o', color='grey', linewidth='1', alpha=0.8, label='T')
	plt.scatter(X_t[np.where(y1 == 21), 0], X_t[np.where(y1 == 21), 1], marker='o', color='gold', linewidth='1', alpha=0.8, label='U')
	plt.scatter(X_t[np.where(y1 == 22), 0], X_t[np.where(y1 == 22), 1], marker='o', color='rosybrown', linewidth='1', alpha=0.8, label='V')
	plt.scatter(X_t[np.where(y1 == 23), 0], X_t[np.where(y1 == 23), 1], marker='o', color='olivedrab', linewidth='1', alpha=0.8, label='W')
	plt.scatter(X_t[np.where(y1 == 24), 0], X_t[np.where(y1 == 24), 1], marker='o', color='deepskyblue', linewidth='1', alpha=0.8, label='X')
	plt.scatter(X_t[np.where(y1 == 25), 0], X_t[np.where(y1 == 25), 1], marker='o', color='orange', linewidth='1', alpha=0.8, label='Y')
	plt.scatter(X_t[np.where(y1 == 26), 0], X_t[np.where(y1 == 26), 1], marker='o', color='salmon', linewidth='1', alpha=0.8, label='Z')
	
	plt.legend(loc='best')
	plt.show()

tsne_plot(encoded_input, y_train)

'''
Now i'll build a new FFW NN to classify the x_train extracted with AE
'''
# preprocess the labels
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_train = np_utils.to_categorical(y_train)

# get a train and validation set from our low dimensional train set
x_train, x_val, y_train, y_val = train_test_split(encoded_input, y_train, test_size=0.2, random_state=42)

# create the model
predicter = Sequential()
predicter.add(Dense(32, activation='relu', input_shape=(32,)))
predicter.add(Dense(64, activation='relu'))
predicter.add(Dropout(0.2))
predicter.add(Dense(11, activation='softmax'))

predicter.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the network
n_epochs = 70
b_size = 128
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = predicter.fit(x_train, y_train, epochs=n_epochs, batch_size=b_size, validation_data=(x_val, y_val), callbacks=[early_stop])

n_epochs = early_stop.stopped_epoch + 1

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

# Save labels for validation data
y_score = predicter.predict(x_val)

# Let's check precision-recall curve

# For each class
precision = dict()
recall = dict()
average_precision = dict()
n_classes = 11
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_val[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_val[:, i], y_score[:, i])

# Use a micro-average metric
precision["micro"], recall["micro"], _ = precision_recall_curve(y_val.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_val, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'salmon', 'gold', 'olivedrab', 'deepskyblue'])

# all this code above is to plot precision-recall-f1 for every class
plt.figure(figsize=(10, 12))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

plt.show()

# now we can predict
predicted_classes = predicter.predict_classes(x_test)

# We have to rename our classes to the original values
predicted_classes = np.where(predicted_classes==0, 16, predicted_classes)
predicted_classes = np.where(predicted_classes==1, 17, predicted_classes)
predicted_classes = np.where(predicted_classes==2, 18, predicted_classes)
predicted_classes = np.where(predicted_classes==3, 19, predicted_classes)
predicted_classes = np.where(predicted_classes==4, 20, predicted_classes)
predicted_classes = np.where(predicted_classes==5, 21, predicted_classes)
predicted_classes = np.where(predicted_classes==6, 22, predicted_classes)
predicted_classes = np.where(predicted_classes==7, 23, predicted_classes)
predicted_classes = np.where(predicted_classes==8, 24, predicted_classes)
predicted_classes = np.where(predicted_classes==9, 25, predicted_classes)
predicted_classes = np.where(predicted_classes==10, 26, predicted_classes)

#np.savetxt("Luca_Gandolfi_807485_score2.txt", predicted_classes, fmt="%s")

# print results
print(Counter(predicted_classes))

'''
Create a new network to take in input the original data
'''
# load again the data
x_train = pickle.load(open("x_train.obj","rb"))
x_test = pickle.load(open("x_test.obj","rb"))
y_train = pickle.load(open("y_train.obj","rb"))

# reshape in [0,1] values
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# labels 
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_train = np_utils.to_categorical(y_train)

# split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# create the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(125, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(11, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the network
n_epochs = 50
b_size = 64
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=b_size, validation_data=(x_val, y_val), callbacks=[early_stop], verbose=2)

n_epochs = early_stop.stopped_epoch + 1

# Show the plots
x_plot = list(range(1,n_epochs+1))
plot_history(history)

# Let's check precision-recall curve

# Save labels for validation data
y_score = model.predict(x_val)

# For each class
precision = dict()
recall = dict()
average_precision = dict()
n_classes = 11
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_val[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_val[:, i], y_score[:, i])

# Use a micro-average metric
precision["micro"], recall["micro"], _ = precision_recall_curve(y_val.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_val, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'salmon', 'gold', 'olivedrab', 'deepskyblue'])

# all this code above is to plot precision-recall-f1 for every class
plt.figure(figsize=(10, 12))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

plt.show()

# now we can predict
predicted_classes = model.predict_classes(x_test)

# We have to rename our classes to the original values
predicted_classes = np.where(predicted_classes==0, 16, predicted_classes)
predicted_classes = np.where(predicted_classes==1, 17, predicted_classes)
predicted_classes = np.where(predicted_classes==2, 18, predicted_classes)
predicted_classes = np.where(predicted_classes==3, 19, predicted_classes)
predicted_classes = np.where(predicted_classes==4, 20, predicted_classes)
predicted_classes = np.where(predicted_classes==5, 21, predicted_classes)
predicted_classes = np.where(predicted_classes==6, 22, predicted_classes)
predicted_classes = np.where(predicted_classes==7, 23, predicted_classes)
predicted_classes = np.where(predicted_classes==8, 24, predicted_classes)
predicted_classes = np.where(predicted_classes==9, 25, predicted_classes)
predicted_classes = np.where(predicted_classes==10, 26, predicted_classes)

np.savetxt("Luca_Gandolfi_807485_score2.txt", predicted_classes, fmt="%s")

# print results
print(Counter(predicted_classes))
