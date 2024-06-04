import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0  # Normalize and reshape input
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train_cat = to_categorical(y_train)  # Convert labels to one-hot encoding
y_test_cat = to_categorical(y_test)

# Define and compile the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(x_train, y_train_cat, epochs=4, batch_size=64, validation_data=(x_test, y_test_cat))

# Evaluate the CNN model accuracy
cnn_predictions = cnn_model.predict(x_test)
cnn_accuracy = accuracy_score(y_test, np.argmax(cnn_predictions, axis=-1))
print("CNN Accuracy:", cnn_accuracy)
