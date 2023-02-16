Human Intention Prediction from Multimodal Features
======

Course project of Machine Perception and Learning class, Winter 2022/23 at Universität Stuttgart.

### Introduction

This study presents a model for predicting human intention using multi-modal human behavior features, based on 1D convolutional neural network. The model was developed and evaluated using the MoGaze dataset, with two sets of features, human pose and eye-gaze. The main goal was to compare the performance of the model with baseline prediction and to show the effectiveness of each possible application of the same dataset across multiple models. The study is motivated by the importance of predicting human intention in various fields, such as human-computer interaction, autonomous driving, and VR/AR applications. The aim of the study is to build a model that utilizes multi-modal features to predict intention and select the optimal one through evaluation.

### Group member

Sena Tarpan, University of Stuttgart, M.Sc. Autonomous Systems in Computer Science

Kuan-Yu Lin, University of Stuttgart, M.Sc. Computational Linguistics 

### Data

This study utilized the MoGaze dataset, which includes human motion data, including full-body motion and eye-gaze data, to develop a model for predicting human intention. The dataset was comprised of recordings from six participants and was used to create a training dataset, with additional recordings from a single participant used to create a test dataset. The Pose_Gaze_Intention class was used for dataset preprocessing, which loads the pose, gaze, and intention data for each subject and action, concatenates the data, and samples sequences of length seq_len with a stride of 1. The resulting sequences are downsampled by taking every 10th sample to reduce the size of the data. The loaded and preprocessed data is returned as a PyTorch dataset object for training and testing the machine learning model.

### Setting and running the model

1. Download the dataset at [here](https://drive.google.com/file/d/1_bAl7rxzc-u1-JOEDV2JXJwm1OyXJ8lZ/view).
2. Set "data_dir" in `src/config.py`
3. Uncomment one of the five models in the main.py. Make sure other four models are commented. If you want to see the resulting animation, please uncomment the lines in `src/utils.py`.
4. Install the requirement of our project 
```
python3 -m pip install -r requirements.txt --user
```
5. Train the model and get the result.
```
python3 main.py
```
