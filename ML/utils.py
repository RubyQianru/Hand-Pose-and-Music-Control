import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import json

def load_data(explore=True, random_seed=1):
    # Load JSON data from the file
    with open("data.json", "r") as file:
        json_data = json.load(file)

    # Extract xs and ys from each dictionary
    x_data = [sample['xs'] for sample in json_data['data']]
    temp = [sample['ys'] for sample in json_data['data']]

    # Assuming you want to convert 'r' to 1 and other values to 0
    y_data = []
    for label in temp:
        if label['0'] == 'h':
            y_data.append(0)
        elif label['0'] == 'r':
            y_data.append(1)
        elif label['0'] == 'y':
            y_data.append(2)
        elif label['0'] == 't':
            y_data.append(3)

    x_data = np.array([list(sample.values()) for sample in x_data])

    # shuffle data 
    if random_seed is not None:
        np.random.seed(random_seed)
    
    shuffle_indices = np.random.permutation(len(x_data))
    x_data = x_data[shuffle_indices]
    y_data = np.array(y_data)[shuffle_indices]

    split_idx = int(len(x_data) * 0.8)
    x_train, y_train = x_data[:split_idx], y_data[:split_idx]
    x_test, y_test = x_data[split_idx:], y_data[split_idx:]

    print(y_data)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def preprocess_data(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(-1, 63).astype("float32") / 505
    x_test = x_test.reshape(-1, 63).astype("float32") / 505

    y_train = y_train.astype("float32") 
    y_test = y_test.astype("float32") 

    val_size = int(len(x_train) * 0.1)

    x_train, x_val = x_train[:-val_size], x_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def build_model():

    model = keras.Sequential([
        layers.Dense(32, activation="relu"),
        layers.Dense(4, activation="softmax")
    ])

    model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    
    return model


def train_model(model, x_train, y_train, x_val, y_val, epochs=5, batch_size=32):
    # train the model using train and validation sets
    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(x_val, y_val))
    return history


def plot_loss(history):
    # plot the training and validation loss side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # plot the training and validation loss
    ax[0].plot(history.history['loss'], label='train')
    ax[0].plot(history.history['val_loss'], label='val')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # plot the training and validation accuracy
    ax[1].plot(history.history['accuracy'], label='train')
    ax[1].plot(history.history['val_accuracy'], label='val')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()


def test_model(model, x_test, y_test):

    predictions = model.predict(x_test)
    print(predictions)
    y_pred = np.argmax(predictions, axis=1)

    test_loss, test_acc = model.evaluate(x_test, y_test)

    

    return test_acc, y_pred


