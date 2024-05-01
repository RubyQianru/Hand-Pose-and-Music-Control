# DL4M-Hand-Pose-and-Music-Control

A music constroller triggered by a specific hand pose, using Hand pose recognition model trained based upon TensorFlow MediaPipe Hand Pose Detection Model. Reference: [Hand Pose Detection](https://github.com/tensorflow/tfjs-models/blob/master/hand-pose-detection/README.md) 

## Data Collection

We are planning to use TensorFlow.js MediaPipe Hand Pose Detection model as the model our transfer learning model is basing upon. With the knowledge of 63 landmarks of hand pose, our model will have a better understanding of features of hand poses, and where users’ hands are located at. For the model training, we will use the 63 landmarks as the data structure of our training samples. For the model usage, we will run the TensorFlow.js MediaPipe Hand Pose Detection model, and use its output as our model input to retrieve predictions.

<img src="https://github.com/RubyQianru/Hand-Pose-and-Music-Control/assets/142470034/d0ac8c39-7418-4905-b1eb-e3a61b3aac32" width="400">

Previous model:

<img src="https://github.com/RubyQianru/Hand-Pose-and-Music-Generation/assets/142470034/c0cceb46-7b63-4d43-805a-b2c23f869fa9" width="400">

Updated model:

<img src="https://github.com/RubyQianru/Hand-Pose-and-Music-Generation/assets/142470034/4fcccc6c-89fe-45a6-8c37-d78f5a6855fe" width="400">


### Steps
1. Download the folder "Data-Collector" from [link](https://github.com/RubyQianru/DL4M-Hand-Pose-and-Music-Generation/tree/main/Data-Collector)
2. Open the folder on VS Code.
3. On VS Code, download plugin "Live Server"


<img width="400" alt="截屏2024-04-17 18 32 22" src="https://github.com/RubyQianru/DL4M-Hand-Pose-and-Music-Generation/assets/142470034/1f401603-c6c4-4962-9b75-403e68109712">


4. Click on "Go Live" on the bottom right of VS Code.


<img width="400" alt="截屏2024-04-17 18 33 26" src="https://github.com/RubyQianru/DL4M-Hand-Pose-and-Music-Generation/assets/142470034/249429fe-98be-4edf-9ab7-99d5e61c2c6f">


5. Make sure the web app is running on Chrome browser. Right click and select "Inspect".
6. When the panel pops up, select "console" to go to console.


<img width="400" alt="截屏2024-04-17 18 37 47" src="https://github.com/RubyQianru/DL4M-Hand-Pose-and-Music-Generation/assets/142470034/d7331ad2-e013-4164-ac7e-7429b1bc56de">


7. Follow the instruction on the web app. Make sure to press key  "s" to save and download the JSON dataset. It might takes 30 seconds for the download to initiate.

9. Refresh the web app anytime if you run into any issues (like errors in console).

## Dataset

You may also access the dataset through the link to [data](https://drive.google.com/file/d/1j5LG9KK0rGxBih69Itpd6Z-gzOUAGMG7/view?usp=share_link)

<img width="400" alt="截屏2024-04-29 16 15 57" src="https://github.com/RubyQianru/Hand-Pose-and-Music-Control/assets/142470034/8e393506-2379-4a8a-8054-ea4707f1da74">

## Model 

The first model is a Keras sequential model object representing the built model. The model architecture consists of four Dense layers: the first dense layer of size 64 with relu activation, the second dense layer of size 32 with relu activation, the third dense layer of size 2 with relu activation, followed by an output layer with a single unit and sigmoid activation function. The model is compiled with the binary cross-entropy loss function, an adam optimizer, and the accuracy metric.

<img width="400" alt="截屏2024-04-30 22 12 42" src="https://github.com/RubyQianru/Hand-Pose-and-Music-Control/assets/142470034/f48c258a-5d1d-4870-aa38-a369bca0e505">

The second model is a Keras sequential model object representing the built model. The model architecture consists of three Dense layers: the first dense layer of size 64 with relu activation, the second dense layer of size 10 with relu activation, followed by an output layer with a single unit and sigmoid activation function. The model is compiled with the binary cross-entropy loss function, an adam optimizer, and the accuracy metric.

<img width="400" alt="截屏2024-04-30 22 12 25" src="https://github.com/RubyQianru/Hand-Pose-and-Music-Control/assets/142470034/a6c98c7f-a37d-448c-aacf-8f924522ea37">

### Requirements

Instructions on setting up the environment:

```
conda env create -f environment.yml
conda activate dl4m-final
```

### Training the Model

Instructions on training the model:

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

Instructions on using the model: 

1. Go to demo.ipynb. 
2. You may use the demo.json we provided inside /Data folder, or you may use our Data Collector to collect your own hand pose data. If you are collectng your own data, make sure to rename the JSON file to "data.json", put it inside /Data folder, and run the following code.
3. You may refer to the **Data Collection** instructions above for a step-by-step runthrough of how to correctly use Data Collector. 

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

<img width="400" alt="截屏2024-04-24 18 54 23" src="https://github.com/RubyQianru/final-project-example/assets/142470034/4830b139-ae45-4b60-84c1-669fc4667675">

## Music Control

Similar to theremin, the Max Patch would allow users to control the pitch of the sound by changing hand positions of the camera, and controlling the volume by sliding your hand left and right. The Max patch will take coordinates from the hand gesture capturing model tensorflow js file in real time.

### Timbre Control

Not only pitch, but also the wavetable can be controlled by the patch; using deep learning model to recognize the hand gesture of the sound, (N)in order to change the type of sound the instrument is going to trigger.

### Realtime Data Transmission

Max Msp internally support .js javascript code, using external compiler to edit the code and have it run within Max environment. Using this function, the coordinate matrix and hand gesture will be imported real time into Max Msp environment to control the instrument.

Create a server environment using Websocket that receives our model output ( prediction result, x, y axis of the hand pose) and transmit the data to outlets in MAX MSP.

### Prototype

https://github.com/RubyQianru/Hand-Pose-and-Music-Control/assets/142470034/423a7bc3-8e7c-43e0-9487-77e9e9e0b5a9

## Future Plan & Usage

1. Classification of multiple handposes.
2. Design and development of Node.js server leveraging websocket to bridge our frontend hand pose classification with our instrument in Max. 

## References

1. Transfer learning model is originally based on [TensorFlow Hand Pose Detection Model](https://github.com/tensorflow/tfjs-models/tree/master/hand-pose-detection)
2. Model report and testing code is based on [DL4M Homework 1](https://github.com/dl4m/homework-1-RubyQianru)

