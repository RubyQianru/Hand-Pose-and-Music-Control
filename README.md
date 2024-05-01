# DL4M-Hand-Pose-and-Music-Control

A music constroller triggered by a specific hand pose, using Hand pose recognition model trained based upon TensorFlow MediaPipe Hand Pose Detection Model. Reference: [Hand Pose Detection](https://github.com/tensorflow/tfjs-models/blob/master/hand-pose-detection/README.md) 

## Data Collection

We are planning to use TensorFlow.js MediaPipe Hand Pose Detection model as the model our transfer learning model is basing upon. With the knowledge of 63 landmarks of hand pose, our model will have a better understanding of features of hand poses, and where users’ hands are located at. For the model training, we will use the 63 landmarks as the data structure of our training samples. For the model usage, we will run the TensorFlow.js MediaPipe Hand Pose model (V2), and use its output as our model input to retrieve predictions.

<img src="https://github.com/RubyQianru/Hand-Pose-and-Music-Control/assets/142470034/d0ac8c39-7418-4905-b1eb-e3a61b3aac32" width="400">

Previous model:

<img src="https://github.com/RubyQianru/Hand-Pose-and-Music-Generation/assets/142470034/c0cceb46-7b63-4d43-805a-b2c23f869fa9" width="400">

Updated model:

<img src="https://github.com/RubyQianru/Hand-Pose-and-Music-Generation/assets/142470034/4fcccc6c-89fe-45a6-8c37-d78f5a6855fe" width="400">


### Steps
1. Download the folder "Data-Collector" from [link](https://github.com/RubyQianru/DL4M-Hand-Pose-and-Music-Generation/tree/main/Data-Collector)
2. Open the folder on VS Code.
3. On VS Code, download plugin "Live Server"


<img width="300" alt="截屏2024-04-17 18 32 22" src="https://github.com/RubyQianru/DL4M-Hand-Pose-and-Music-Generation/assets/142470034/1f401603-c6c4-4962-9b75-403e68109712">


4. Click on "Go Live" on the bottom right of VS Code.


<img width="300" alt="截屏2024-04-17 18 33 26" src="https://github.com/RubyQianru/DL4M-Hand-Pose-and-Music-Generation/assets/142470034/249429fe-98be-4edf-9ab7-99d5e61c2c6f">


5. Make sure the web app is running on Chrome browser. Right click and select "Inspect".
6. When the panel pops up, select "console" to go to console.


<img width="300" alt="截屏2024-04-17 18 37 47" src="https://github.com/RubyQianru/DL4M-Hand-Pose-and-Music-Generation/assets/142470034/d7331ad2-e013-4164-ac7e-7429b1bc56de">


7. Follow the instruction on the web app. Make sure to press key  "s" to save and download the JSON dataset. It might takes 30 seconds for the download to initiate.

9. Refresh the web app anytime you want to collect a new label or run into any issues.

## Dataset

You may access the dataset through the link to [data](https://drive.google.com/file/d/1j5LG9KK0rGxBih69Itpd6Z-gzOUAGMG7/view?usp=share_link)

<img width="300" alt="截屏2024-04-29 16 15 57" src="https://github.com/RubyQianru/Hand-Pose-and-Music-Control/assets/142470034/8e393506-2379-4a8a-8054-ea4707f1da74">

## Model 

The first model is a Keras sequential model object representing the built model. The model architecture consists of four Dense layers: the first dense layer of size 64 with relu activation, the second dense layer of size 32 with relu activation, the third dense layer of size 2 with relu activation, followed by an output layer with a single unit and sigmoid activation function. The model is compiled with the binary cross-entropy loss function, an adam optimizer, and the accuracy metric.

<img width="300" alt="截屏2024-04-30 22 12 42" src="https://github.com/RubyQianru/Hand-Pose-and-Music-Control/assets/142470034/f48c258a-5d1d-4870-aa38-a369bca0e505">

The second model is a Keras sequential model object representing the built model. The model architecture consists of three Dense layers: the first dense layer of size 64 with relu activation, the second dense layer of size 10 with relu activation, followed by an output layer with a single unit and sigmoid activation function. The model is compiled with the binary cross-entropy loss function, an adam optimizer, and the accuracy metric.

<img width="300" alt="截屏2024-04-30 22 12 25" src="https://github.com/RubyQianru/Hand-Pose-and-Music-Control/assets/142470034/a6c98c7f-a37d-448c-aacf-8f924522ea37">

### Requirements

Give instructions on setting up the environment.

```
conda env create -f environment.yml
conda activate hotdog
```

### Training the Model

```python
model = u.build_model()

# Adjust class weights based on the dataset balance
class_weights = {0: 1, # o
                 1: 1} # n

history, model = u.train_model(model, x_train, y_train, x_val, y_val, 
                      epochs=30, batch_size=64, class_weights=class_weights)

u.plot_loss(history)
```


### Using the Model

```python
# Load demo data from the folder
file_paths = ["../Data/demo.json"]
x_data, y_data = u.load_data(file_paths) 
```

```python
# Load model from the folder
from tensorflow.keras.models import load_model
model = load_model('Handpose-Recognition.h5')
```

```python
import numpy as np

# Classify handpose
# sample random handpose from the demo dataset
idx = np.random.randint(0, len(y_data), size=1)[0] 
handpose_features = x_data[idx][np.newaxis, :] 
print('This handpose is not the target handpose.' if y_data[idx]==0 else 
      'This handpose is the target handpose.')

# Use your model to make predictions
y_pred = u.classify_handpose(model, handpose_features)

if y_pred[0] == 1:
    print('The model says this handpose is the target handpose.')
else:
    print('The model says this handpose is not the target handpose.')
```

<img width="300" alt="截屏2024-04-24 18 54 23" src="https://github.com/RubyQianru/final-project-example/assets/142470034/4830b139-ae45-4b60-84c1-669fc4667675">

## References


