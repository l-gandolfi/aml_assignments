import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# define the convolutional neural network
def conv(model):
	# first convolutional block
	model.add(Conv2D(filters = 8, kernel_size=(3, 3), strides=(1, 1),
		         activation='relu', input_shape=(28,28,1)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.BatchNormalization())

	# second convolutional block
	model.add(Conv2D(filters = 16, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

	# third convolutional block
	model.add(Conv2D(filters = 4, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Dropout(0.25))

	# prepare data for the fully-connected layers
	model.add(Flatten())

	# fully-connected layers
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(num_classes, activation='softmax'))
	
	return model

# define plot function
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

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# pre-process
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# reshape to have 1 channel b&w
x_train = x_train.reshape(len(x_train),28,28,1)
x_test = x_test.reshape(len(y_test),28,28,1)

# binarizate the labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# split train into train and validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# build the model
model = Sequential()
model = conv(model)

print(model.summary())

# fit parameters
batch_size = 64
epochs = 35
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
opt = keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
opt2 = 'adadelta'

'''
# per usare la seguente funzione conviene alzare il learning_rate di adam a 0.01
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
'''

# loss function and optimizer
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])

# fit the model
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_val, y_val), callbacks=[early_stop])

n_epochs = early_stop.stopped_epoch + 1

# Show the plots
x_plot = list(range(1,n_epochs+1))
plot_history(history)

# evaluate results
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(classification_report(pred, np.argmax(y_test, axis=1)))
