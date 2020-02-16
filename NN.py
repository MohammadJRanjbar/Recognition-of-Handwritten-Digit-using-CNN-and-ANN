from keras.datasets.mnist import load_data
import keras 
from keras.models import Sequential 
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Activation , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD , Adam
from sklearn.model_selection import train_test_split
from keras.losses import categorical_crossentropy,binary_crossentropy
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
(train_digits, train_labels), (test_digits, test_labels) = load_data()
image_height = train_digits.shape[1]  
image_width = train_digits.shape[2]
num_channels = 1 
train_data = np.reshape(train_digits, (train_digits.shape[0], image_height, image_width, num_channels))
test_data = np.reshape(test_digits, (test_digits.shape[0],image_height, image_width, num_channels))
train_data = train_data.astype('float32') / 255.
test_data = test_data.astype('float32') / 255.
num_classes = 10
train_labels_cat = to_categorical(train_labels,num_classes)
test_labels_cat = to_categorical(test_labels,num_classes)
model=Sequential()
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
filepath='weights0.{epoch:02d}-{val_loss:.2f}.hdf5'
CB=keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_cat, epochs=50, batch_size=64,validation_split=0.2 , callbacks=[CB])
model.summary()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label = 'val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
from keras.models import load_model
model3=load_model('weights0.03-1.41.hdf5')
test_loss, test_acc = model3.evaluate(test_data,  test_labels_cat, verbose=2)
print(test_loss, test_acc)
train_loss, train_acc = model3.evaluate(train_data,  train_labels_cat, verbose=2)
print(train_loss, train_acc)