Human Intention Prediction from Multimodal Features
======

Course project of machine perception and learning, Winter 2022/23 at Universität Stuttgart.

### Introduction

Human intention prediction can be used many areas, for example: eye tracking, autonomous driving, VR/AR applications ...

Our goal in this project is develop Multimodel Learning method to predict human intention from 2 input modalities 

are eye gaze and human pose. Because, there is a correlation between them to predict intention.

### Group member

Sena Tarpan, University of Stuttgart, M.Sc. Autonomous Systems in Computer Science

Kuan-Yu Lin, University of Stuttgart, M.Sc. Computational Linguistics 

### Data

Input = Human Pose and eye-gaze data

Output = Intention

In our dataset, we have records of 6 people. 

Our training dataset consists of person 1, person 2, person 4, person 5 and person 6 records.

Our test dataset consists of person 7 records.

### Setting and running the model

1. Download the dataset at [here](https://drive.google.com/file/d/1_bAl7rxzc-u1-JOEDV2JXJwm1OyXJ8lZ/view).
2. Set "data_dir" in src/config.py
3. Uncomment one of the five models in the main.py. Make sure other four models are commented.
4. Install the requirement of our project 
```
python3 -m pip install -r requirements.txt --user
```
5. Train the model and get the result.
```
python3 main.py
```
