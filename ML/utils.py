import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import json

def load_data(file_paths):

    all_x_data = []
    all_y_data = []
    
    for file_path in file_paths:
        with open(file_path, "r") as file:
            json_data = json.load(file)
        
        x_data = [sample['xs'] for sample in json_data['data']]
        x_data = [[coord for dict_ in entry for coord in (dict_['x'], dict_['y'], dict_['z'])] for entry in x_data]
        all_x_data.extend(x_data)  

        y_data = []
        for sample in json_data['data']:
            label = sample['ys']['0']
            if label == 'o':
                y_data.append(1)
            elif label == 'n':
                y_data.append(0)
        all_y_data.extend(y_data)  

    all_x_data = np.array(all_x_data)
    all_y_data = np.array(all_y_data)

    return all_x_data, all_y_data


def shuffle_data(x_data, y_data, random_seed=2):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    shuffle_indices = np.random.permutation(len(x_data))
    x_data = x_data[shuffle_indices]
    y_data = y_data[shuffle_indices]

    return x_data, y_data


def split_data(x_data, y_data, split_idx):

    x_train, y_train = x_data[:split_idx], y_data[:split_idx]
    x_test, y_test = x_data[split_idx:], y_data[split_idx:]

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def preprocess_data(x_train, y_train, x_test, y_test):

    x_train = x_train.reshape(-1, 63)
    x_test = x_test.reshape(-1, 63)

    val_size = int(len(x_train) * 0.1)

    x_train, x_val = x_train[:-val_size], x_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def build_baseline(input_shape=63):

    model = keras.models.Sequential([

        # Dense layer with 64 units
        layers.Dense(64, input_shape = (input_shape,), activation='relu'),  
        layers.Dense(32, activation='relu'),  
        layers.Dense(2, activation="relu"),  

        # Sigmoid output layer 
        layers.Dense(1, activation='sigmoid')  
    ])

    model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
    
    return model

def build_model(input_shape=63):

    model = keras.models.Sequential([

        # Dense layer with 64 units
        layers.Dense(64, input_shape = (input_shape,), activation='relu'),  
        layers.Dense(10, activation="relu"),  

        # Sigmoid output layer 
        layers.Dense(1, activation='sigmoid')  
    ])

    model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
    
    return model


def train_model(model, x_train, y_train, x_val, y_val, epochs=5, batch_size=32, class_weights=None):
    # train the model using train and validation sets
    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(x_val, y_val),
                        class_weight=class_weights)
    return history, model

def classify_handpose(model, hand_features):

    predictions = model.predict(hand_features)
    y_pred = np.where(predictions < 0.5, 0, 1)
    y_pred = np.squeeze(y_pred, axis=1)

    return y_pred


def test_model(model, x_test, y_test):
    # Get model predictions
    y_pred = model.predict(x_test)

    y_pred_labels = (y_pred > 0.5).astype(int)

    acc = np.mean(y_pred_labels.flatten() == y_test.flatten())  

    return acc, y_pred_labels


# Auxiliary functions below =====================================

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

def explore_data(X_train, y_train, y_test, y_val):

    # Class names
    class_names = [0, 1]

    # Plot the distribution of classes in the training, validation, and test sets
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    # Plot the distribution of classes in the training set
    train_class_counts = np.bincount(y_train)
    ax[0].bar(range(len(class_names)), train_class_counts)
    ax[0].set_xticks(range(len(class_names)))
    ax[0].set_xticklabels(class_names, rotation=45)
    ax[0].set_title('Training set')

    # Plot the distribution of classes in the test set
    test_class_counts = np.bincount(y_val)
    ax[1].bar(range(len(class_names)), test_class_counts)
    ax[1].set_xticks(range(len(class_names)))
    ax[1].set_xticklabels(class_names, rotation=45)
    ax[1].set_title('Val set')

    # Plot the distribution of classes in the test set
    test_class_counts = np.bincount(y_test)
    ax[2].bar(range(len(class_names)), test_class_counts)
    ax[2].set_xticks(range(len(class_names)))
    ax[2].set_xticklabels(class_names, rotation=45)
    ax[2].set_title('Test set')

    plt.show()

