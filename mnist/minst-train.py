# import ML libraries
from keras.datasets import mnist
from keras.models import Sequential
import keras.layers as layers
import matplotlib.pyplot as plt
import keras
import matplotlib.pyplot as plt
from keras import datasets, layers, models

# setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# pre-process images
x_train /= 255.0
x_test /= 255.0

# convert to one-hot encoding
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("Processed training data shape: ", x_train.shape)
print("Processed test data shape: ", x_test.shape)

def create_nn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    return model

def evaluate(model, batch_size=32, epochs=10):
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=True)
    loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    print()
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')

model = create_dense([256, 128, 64, 32])
evaluate(model)
