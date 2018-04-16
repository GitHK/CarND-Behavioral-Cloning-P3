# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering
angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without
leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car
can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the
convolution neural network. The file shows the pipeline I used for
training and validating the model, and it contains some comments to
explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My approach was to start from a well tested network, which gave good
results. The `Nvidia network`. Look at the `make_model` function in
code (lines 78 to 97).

The original model was extended with an additional convolutional layer
composed of 128 filters with 3x3 kernels. After the frist five
convolutional layers a dropout layer with probability 50% was added.

The network is also responsible for data normalization (keras lambda
layer) and for cropping ( using keras cropping2d layer) the upper 50
and lower 20 lines from the image (first two operations done on images).

RELU activation function was was kept as in the original model.

#### 2. Attempts to reduce overfitting in the model


To avoid overfitting dropout layers were placed after the first five
convolutional layers.

The model was tested by running it through the simulator and ensuring
that the vehicle could stay on the track.

#### 3. Model parameter tuning

Adam optimizer was used for learning rate tuning.

Another parameter called `steering_correction` was tuned manually. The
dataset was also composed of images coming from the side cameras, thus
compensating for the camera view angle was necessary.


#### 4. Appropriate training data

Training data was chosen by driving the vehicle round the first track
in clockwise and counter clockwise directions once. This was used
to teach the car to stay in the center of the road.

Additional training data was also registered by driving the vehicle on
the second track. Thi was used to teach the care to make sharper turns.
Without this data, the car would not steer well enough.

An undesired characteristic inherited from the second track is to drive
on the left side of the track when going straight. This was introduced
accidentally by me. I had driven the car too close to the left border
of the track, before sharp turns.

As mentioned above, training data also includes data from the left and
right side cameras on the car.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

At first I tried the LeNet model. Results were mediocre and not event
with better data improvements were noticeable.

A good thing to do was to augment existing data to enhance results. I
added data augmentation (flipped the image and multiplied the steering
angle by -1). Also normalization and image cropping were added. These
three combined gave a good boost to driving performance. Even with the
LeNet model in place, better results were visible.

After some reading on the internet I decided that my goto network
architecture would be the [Nvidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
model. It was a good start. Smooth driving was provided, and almost
kept the car on track.

The way to optimize from this point was to acquire better data. I had
previously tried, keyboard input, PS4 controller input and mouse.
The combination between mouse and keyboard gave the best results. It
permitted to create smooth steering angles during turns. Data was
collected from both tracks.

In the end an additional convolutional layer was added. And tne next
step was adding dropout layers between all convolutional layers.

The result was smooth driving around the first track.

#### 2. Final Model Architecture

Below a summary of the model as provided by Keras' `model.summary`
function is shownn. The model architecture can be visualized in code
at lines 78 to 97, inside the `make_model` function.


|Layer (type)                       | Output Shape          | Param #   |
|-----------------------------------|-----------------------|-----------|
|lambda_1 (Lambda)                  | (None, 160, 320, 3)   | 0         |
|cropping2d_1 (Cropping2D)          | (None, 90, 320, 3)    | 0         |
|convolution2d_1 (Convolution2D)    | (None, 43, 158, 24)   | 1824      |
|dropout_1 (Dropout)                | (None, 43, 158, 24)   | 0         |
|convolution2d_2 (Convolution2D)    | (None, 20, 77, 36)    | 21636     |
|dropout_2 (Dropout)                | (None, 20, 77, 36)    | 0         |
|convolution2d_3 (Convolution2D)    | (None, 8, 37, 48)     | 43248     |
|dropout_3 (Dropout)                | (None, 8, 37, 48)     | 0         |
|convolution2d_4 (Convolution2D)    | (None, 6, 35, 64)     | 27712     |
|dropout_4 (Dropout)                | (None, 6, 35, 64)     | 0         |
|convolution2d_5 (Convolution2D)    | (None, 4, 33, 64)     | 36928     |
|dropout_5 (Dropout)                | (None, 4, 33, 64)     | 0         |
|convolution2d_6 (Convolution2D)    | (None, 2, 31, 128)    | 73856     |
|flatten_1 (Flatten)                | (None, 7936)          | 0         |
|dense_1 (Dense)                    | (None, 100)           | 793700    |
|dense_2 (Dense)                    | (None, 50)            | 5050      |
|dense_3 (Dense)                    | (None, 10)            | 510       |
|dense_4 (Dense)                    | (None, 1)             | 11        |

    Total params: 1,004,475
    Trainable params: 1,004,475
    Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process


To inherit a good driving behaviour the car was driven one lap clockwise
and one lap counterclockwise on the first track. The car was kept for
most of the time in the center of the lane, even during curves.

![img1.jpg]

As the car was laking in turning ability, I've recorded some data on the
second track; an entire lap. The driving was not precise, I made hard
turns in the attempt to improve turning rate on the first track.

![img2.jpg]

After the data collection process I ended up with 16764 data points.
This data was preprocessed by flipping the images and multiplying the
steering angle by -1.
The data also includes the right and left cameras. The total amount of
data points after argumentation was 33528.

The data set was split 80% - 20% between training and validation sets.
Data was randomly shuffled. Also to help out the training process a
generator was used to load up data.

The ideal number of epochs turned out to be 3. The model was neither
underfitting no overfitting. An adam optimizer was used to avoid manual
tuning of the learning rates.

#### 4. Final conclusions

I think something is lacking in my approach. A lot of data was provided
and the model is capable of driving on its own, but it is capable of
driving only on a training track. The track from which most of the data
is coming.

I would like to try something else: acquiring data the second track and
teach the car how to drive on the first without showing it any data
from it.

This could provide a more reliable model for autonomous driving.