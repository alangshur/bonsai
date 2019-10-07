from keras.datasets import mnist
from keras.models import Sequential
import keras.layers as layers
import matplotlib.pyplot as plt
import keras

# setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
xh, xw = x_train[0].shape

# pre-process data
x_train = x_train.reshape(x_train.shape[0], xh, xw, 1)
x_test = x_test.reshape(x_test.shape[0], xh, xw, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
input_shape = (xh, xw, 1)

# convert to one-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# build CNN model
model = Sequential()
model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(num_classes, activation='softmax'))

# compile model for training
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

#model training
model_log = model.fit(x_train, y_train,
    batch_size=128,
    epochs=8,
    verbose=True,
    validation_data=(x_test, y_test)
)

# plot result
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(model_log.history['accuracy'])
plt.plot(model_log.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
plt.show()

# save model
model_digit_json = model.to_json()
with open("mnist\model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
model.save_weights("mnist\model_digit.h5")