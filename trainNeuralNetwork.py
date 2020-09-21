import json
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

DATA_PATH = "data.json"

LEARNING_RATE = 0.0001

EPOCHS = 50

BATCH_SIZE = 32

SAVED_MODEL_PATH = "model.h5"

NUM_OF_KEYWOARDS = 12

def load_dataset(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(fp)
        
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return X, y

def get_data_splits(data_path, test_size = 0.1, validation_size = 0.1):

    X, y = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, 
                                                        test_size = validation_size)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test                                                                                        

def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):

    model = keras.Sequential()

    # layer 1
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", 
                                input_shape = input_shape, 
                                kernel_regularizer=keras.regularizers.l2(LEARNING_RATE)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"))

    # layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", 
                                kernel_regularizer=keras.regularizers.l2(LEARNING_RATE)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"))

    # layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu", 
                                kernel_regularizer=keras.regularizers.l2(LEARNING_RATE)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(NUM_OF_KEYWOARDS, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer, loss=error, metrics=["accuracy"])

    model.summary()

    return model



def main():

    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape, LEARNING_RATE)

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size = BATCH_SIZE, 
                validation_data=(X_validation, y_validation))

    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, test accuracy: {test_accuracy}")

    model.save(SAVED_MODEL_PATH)

if __name__ == "__main__":
    main()