# DL4M-Hand-Pose-and-Music-Generation

A music constroller triggered by a specific hand pose, using Hand pose recognition model trained based upon TensorFlow MediaPipe Hand Pose Detection Model. Reference: [Hand Pose Detection](https://github.com/tensorflow/tfjs-models/blob/master/hand-pose-detection/README.md)

## Data Collection

We are planning to use TensorFlow.js MediaPipe Hand Pose Detection model as the model our transfer learning model is basing upon. With the knowledge of 63 landmarks of hand pose, our model will have a better understanding of features of hand poses, and where users’ hands are located at. For the model training, we will use the 63 landmarks as the data structure of our training samples. For the model usage, we will run the TensorFlow.js MediaPipe Hand Pose model (V2), and use its output as our model input to retrieve predictions.

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


7. Follow the instruction on the web app. Make sure to copy the JSON output from the console, and paste it into a local JSON file.


<img width="300" alt="截屏2024-04-17 18 41 08" src="https://github.com/RubyQianru/DL4M-Hand-Pose-and-Music-Generation/assets/142470034/b83f4ada-c44c-4afe-8b77-77d8ed4b60cb">


9. Refresh the web app anytime you want to collect a new label or run into any issues.

## Model Training
