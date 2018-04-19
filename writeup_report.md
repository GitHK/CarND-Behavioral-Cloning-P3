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

Extra files:

* video.mp4 contains `model.h5` autonomous mode video **model
track 1**
* video_track2.mp4 contains  `model_track2.h5` autonomous mode video
**model track 2**
* track1.zip training dataset track1
* track2.zip training dataset track2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car
can be driven autonomously around the track by executing

For the first track:

```sh
python drive.py model.h5
```

For the second track:

```sh
python drive.py model_track2.h5
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
code (lines 77 to 99).

The original model was extended with an additional convolutional layer
composed of 128 filters with 3x3 kernels. After each of the frist five
convolutional layers a dropout layer with probability 20% was added.

The network is also responsible for data normalization (keras lambda
layer) and for cropping ( using keras cropping2d layer) the upper 50
and lower 20 lines from the image (first two operations done on images).

RELU activation function was was kept as in the original model. For the
dense layers `tanh` activaiton function was used. It avoided overfiting
thus providing a better result.

#### 2. Attempts to reduce overfitting in the model

To avoid overfitting dropout layers were placed after the first five
convolutional layers.
Activation layers were also added after each fully connected layer.

The model was tested by running it through the simulator and ensuring
that the vehicle could stay on the track.

#### 3. Model parameter tuning

Adam optimizer was used for learning rate tuning.

Another parameter called `steering_correction` was tuned manually. The
dataset was also composed of images coming from the side cameras, thus
compensating for the camera view angle was necessary.


#### 4. Appropriate training data

My approach was to gather up as little data as possible. I used a trial
and error approach.

At first I registered a couple of seconds of driving straight,
driving and through curves on the first track.

Next I observed where the car was failing and registered additional
data to compensate and keep it on the road (in some cases this
was a long process).

In a couple of iterations I had the car driving on the firs track.

For the second track, the model used on the first track seemed to do
great except in the U turns and on a couple of ramps. With 4 keypoint
samples, and some other minor corrective samples,the model was able
drive a complete lap around the second track.

The final result was a model capable of driving around the second track.

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
permitted to create smooth steering angles during turns.

In the end an additional convolutional layer was added. And tne next
step was adding dropout layers between all convolutional layers. Extra
activation layers were added after each fully connected layer.

The result: the car obtained smooth driving with carefully acquired data,
on both tracks.

#### 2. Final Model Architecture

Below a summary of the model as provided by Keras' `model.summary`
function is shownn. The model architecture can be visualized in code
at lines 78 to 97, inside the `make_model` function.

|Layer (type)                     | Output Shape            | Param #  |
|---------------------------------|-------------------------|----------|
|lambda_1 (Lambda)                |(None, 160, 320, 3)      |0         |
|cropping2d_1 (Cropping2D)        |(None, 90, 320, 3)       |0         |
|onvolution2d_1 (Convolution2D)   |(None, 43, 158, 24)      |1824      |
|dropout_1 (Dropout)              |(None, 43, 158, 24)      |0         |
|convolution2d_2 (Convolution2D)  |(None, 20, 77, 36)       |21636     |
|dropout_2 (Dropout)              |(None, 20, 77, 36)       |0         |
|convolution2d_3 (Convolution2D)  |(None, 8, 37, 48)        |43248     |
|dropout_3 (Dropout)              |(None, 8, 37, 48)        |0         |
|convolution2d_4 (Convolution2D)  |(None, 6, 35, 64)        |27712     |
|dropout_4 (Dropout)              |(None, 6, 35, 64)        |0         |
|convolution2d_5 (Convolution2D)  |(None, 4, 33, 64)        |36928     |
|dropout_5 (Dropout)              |(None, 4, 33, 64)        |0         |
|convolution2d_6 (Convolution2D)  |(None, 2, 31, 128)       |73856     |
|flatten_1 (Flatten)              |(None, 7936)             |0         |
|dense_1 (Dense)                  |(None, 100)              |793700    |
|dense_2 (Dense)                  |(None, 50)               |5050      |
|dense_3 (Dense)                  |(None, 10)               |510       |
|dense_4 (Dense)                  |(None, 1)                |11        |

    Total params: 1,004,475
    Trainable params: 1,004,475
    Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process

In order to complete a full lap around both track, I used the same
model for each track with slightly different datasets. The one used
for the second track contains the first and has extra data coming from
the second track.

Data was acquired in small sessions of a couple of seconds at most, in
key points arround the first track. Where problems occured, more data
was acquired. This helped to design a final solution for track one.

Data from the second track was only acquired form key points where the
previously trained model performed bad, thus to drive on the second
track data from the first track was used in conjunction with data from
some of the parts of the second track where the model performed very bad,
the U turns and a couple of ramps.

Note that hhe second model was not capable of driving a full lap around
the first track.

The acquired data was preprocessed by flipping the images and multiplying
the steering angle by -1. The data also includes the right and left
cameras.

###### Model track 1

This model has **558** lines in the csv file, which correspond to a
total of 1674 raw images. With image augmentation the final amount
of images used were `3348`.

###### Model track 2

This model has **558** lines in the csv file, which correspond to a
total of 3015 raw images. With image augmentation the final amount
of images used were `6030`.

Samples can be easily visualized in the **track1.zip** and **track2.zip**
archives, In total 23 MB of compressed data was used for the first
model, and an extra 13 MB of compressed data was used for the second
model.

The data set was split 80% - 20% between training and validation sets.
Data was randomly shuffled. Also to help out the training process a
generator was used to load up data.

The ideal number of epochs turned out to be 3. An adam optimizer was
used to avoid manual tuning of the learning rates.

As seen in the diagrams below both models seem to be fitted well enough.

###### Model track 1 loss diagram

![loss_diagram_model_track1.png]

###### Model track 2 loss diagram

![loss_diagram_model_track2.png]



#### 4. Final conclusions

I was surprised by the quality and the ability of the network to drive
the car around the second track, especially with such a small
about of data, and most of it coming from the first track.

In previous attempts I ended up with 10 times more data and had awful
results.

A next step would be to train the model to drive on both tracks, and
not to extend it to be able to drive on the second but loose the ability
to drive on the first track.