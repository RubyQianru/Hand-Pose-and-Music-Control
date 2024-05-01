import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import json

def load_data(file_paths):
    """
    Load and preprocess data from a list of JSON files containing features and labels.

    Returns
    -------
    all_x_data : np.ndarray
        A numpy array containing the extracted and processed features from all provided files.
        Each feature set is a flattened list of coordinates (x, y, z) from the JSON data structure.
    all_y_data : np.ndarray
        A numpy array containing all labels associated with the features. Labels are converted
        from 'o' and 'n' in the JSON files to binary values 1 and 0, respectively.
    """

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
    """
    Shuffle the data arrays in unison while maintaining the correlation between features and labels.

    Parameters
    ----------
    x_data : np.ndarray
        A numpy array containing the features to be shuffled.
    y_data : np.ndarray
        A numpy array containing the labels corresponding to the features in x_data.
    random_seed : int, optional
        An integer seed for the random number generator to ensure reproducibility of the shuffle.
        Default is 2.

    Returns
    -------
    x_data : np.ndarray
        The shuffled numpy array of features.
    y_data : np.ndarray
        The shuffled numpy array of labels, maintaining correspondence with x_data.
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    shuffle_indices = np.random.permutation(len(x_data))
    x_data = x_data[shuffle_indices]
    y_data = y_data[shuffle_indices]

    return x_data, y_data


def split_data(x_data, y_data, split_idx):
    """
    Split the data into training and testing sets based on a specified index.

    Parameters
    ----------
    x_data : np.ndarray
        A numpy array containing the features of the entire dataset.
    y_data : np.ndarray
        A numpy array containing the labels of the entire dataset, corresponding to the features in x_data.
    split_idx : int
        The index at which the data is split into training and testing sets. Features and labels before this index
        are assigned to the training set, and those after to the testing set.

    Returns
    -------
    x_train : np.ndarray
        A numpy array containing the features for the training set.
    y_train : np.ndarray
        A numpy array containing the labels for the training set, corresponding to x_train.
    x_test : np.ndarray
        A numpy array containing the features for the testing set.
    y_test : np.ndarray
        A numpy array containing the labels for the testing set, corresponding to x_test.
    """

    x_train, y_train = x_data[:split_idx], y_data[:split_idx]
    x_test, y_test = x_data[split_idx:], y_data[split_idx:]

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def preprocess_data(x_train, y_train, x_test, y_test):
    """
    Reshape and partition the data into training, validation, and testing sets.

    Parameters
    ----------
    x_train : np.ndarray
        A numpy array containing the features for the training set.
    y_train : np.ndarray
        A numpy array containing the labels for the training set, corresponding to x_train.
    x_test : np.ndarray
        A numpy array containing the features for the testing set.
    y_test : np.ndarray
        A numpy array containing the labels for the testing set, corresponding to x_test.

    Returns
    -------
    x_train : np.ndarray
        A numpy array of reshaped features for the training set.
    y_train : np.ndarray
        A numpy array of labels for the training set.
    x_val : np.ndarray
        A numpy array of reshaped features for the validation set extracted from the original training set.
    y_val : np.ndarray
        A numpy array of labels for the validation set, corresponding to x_val.
    x_test : np.ndarray
        A numpy array of reshaped features for the testing set.
    y_test : np.ndarray
        A numpy array of labels for the testing set.
    """

    x_train = x_train.reshape(-1, 63)
    x_test = x_test.reshape(-1, 63)

    val_size = int(len(x_train) * 0.1)

    x_train, x_val = x_train[:-val_size], x_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def build_baseline(input_shape=63):
    """
    Build a baseline neural network model for binary classification.

    Parameters
    ----------
    input_shape : int, optional
        The shape (number of features) of the input data. Default is 63.
        This parameter specifies the number of features in the input dataset.

    Returns
    -------
    model : keras.models.Sequential
        A Keras sequential model object configured for binary classification. 
        The model consists of a series of Dense layers with ReLU activations, 
        ending with a single unit output layer with a sigmoid activation function 
        to output probabilities indicative of class membership.
        The model is compiled with the Adam optimizer, binary cross-entropy as the
        loss function, and tracks accuracy as a performance metric.
    """

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
    """
    Construct a neural network model for binary classification with a simpler architecture.

    Parameters
    ----------
    input_shape : int, optional
        The number of input features. Default is 63.
        This parameter specifies the size of the input data expected by the model.

    Returns
    -------
    model : keras.models.Sequential
        A Keras sequential model that is configured for binary classification tasks.
        The architecture includes a sequence of Dense layers with ReLU activations 
        and a final output layer with a sigmoid activation. The model is compiled 
        with the Adam optimizer, uses binary cross-entropy as the loss function, and 
        measures accuracy as the performance metric.
    """

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
    """
    Train a given Keras model using the provided training data and validate it using the provided validation data.

    Parameters
    ----------
    model : keras.models.Sequential
        The Keras model to be trained.
    x_train : np.ndarray
        The input features for training the model.
    y_train : np.ndarray
        The target labels for training the model.
    x_val : np.ndarray
        The input features for validating the model.
    y_val : np.ndarray
        The target labels for validating the model.
    epochs : int, optional
        The number of epochs to train the model for. Default is 5.
    batch_size : int, optional
        The batch size to use during training. Default is 32.
    class_weights : dict, optional
        A dictionary mapping class indices to a weight for the class, to balance the training data during model training. Default is None.

    Returns
    -------
    history : keras.callbacks.History
        The training history object containing the details of the training process, such as loss and accuracy for each epoch.
    model : keras.models.Sequential
        The trained Keras model after completing the training cycles.
    """

    # train the model using train and validation sets
    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(x_val, y_val),
                        class_weight=class_weights)
    return history, model

def classify_handpose(model, hand_features):
    """
    Classify hand poses as one of the classes (0 or 1) based on the given features using a trained model.
    This function is designed to handle individual or few samples, making it suitable for real-time applications.

    Parameters
    ----------
    model : keras.engine.training.Model
        The trained model used for classification of hand poses.
    hand_features : np.ndarray
        A numpy array containing the features of the hand poses to classify. The features should be
        preprocessed appropriately to match the input requirements of the model.

    Returns
    -------
    np.ndarray
        A numpy array containing the predicted labels (0 or 1) for the given hand poses. '0' might
        represent a specific gesture or non-target class, and '1' might represent another specific gesture
        or target class, depending on the model training.
    """

    predictions = model.predict(hand_features)
    y_pred = np.where(predictions < 0.5, 0, 1)
    y_pred = np.squeeze(y_pred, axis=1)

    return y_pred


def test_model(model, x_test, y_test):
    """
    Test a trained model using the provided testing data set to evaluate its accuracy.

    Parameters
    ----------
    model : keras.engine.training.Model
        The trained Keras model to be evaluated.
    x_test : np.ndarray
        The input features for the testing set.
    y_test : np.ndarray
        The true target labels against which the model predictions will be compared.

    Returns
    -------
    acc : float
        The accuracy of the model on the testing set, calculated as the percentage of 
        correctly predicted labels.
    y_pred_labels : np.ndarray
        A numpy array containing the predicted labels (0 or 1) for the testing set. 
    """
    
    # Get model predictions
    y_pred = model.predict(x_test)

    y_pred_labels = (y_pred > 0.5).astype(int)
    acc = np.mean(y_pred_labels.flatten() == y_test.flatten())  

    return acc, y_pred_labels


# Auxiliary functions below sourced from DL4M Homework 1 
# https://github.com/dl4m/homework-1-RubyQianru
# =====================================

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

