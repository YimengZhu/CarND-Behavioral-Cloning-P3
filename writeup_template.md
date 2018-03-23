# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"
[image3]: ./examples/left_2016_12_01_13_46_38_947.jpg "left camera image"
[image4]: ./examples/center_2016_12_01_13_46_38_947.jpg "center camera image"
[image5]: ./examples/right_2016_12_01_13_46_38_947.jpg "right camera image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I choosed [the NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) that is sugessted from the lecture with following layers.

My model consists of 3 convolution layers with 5x5 filter sizes and depths between 24, 36 and 48. After those layers there are 2 convolution layrs with 3x3 filter sizes and depths of 64. (model.py lines 82-90).

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 80). 

To reduce the unnecessary information in the images I used a cropping layer to crop the top and bottom aera of the raw images (code line 81).

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py lines 95). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 21). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 110).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the sample data from the Udacity. As I found the data was not equally distributed in terms of the steering angles. There are almost 70% of the data representing the straight driving, that have the steering angle little than 0.05. So I choosed part of the data (80%) to ensure the data is relative fairer distributed, so that I can gain a better model.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the Nvidia network architecture and the training data offered by Udacity to train a good model, that has a small loss on both training set and cross validation set.

My first step was to use a convolution neural network model that is the same as Nvidia Network.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model and between the first and secound dense layer there is a dropout layer with dropout probility of 0.7 will be added.

Then I retrain the model again and found the gap between training loss and validation loss became smaller.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

- Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Flatten layer
- Fully connected: neurons: 100, activation: RELU
- Drop out (0.5)
- Fully connected: neurons:  50, activation: RELU
- Fully connected: neurons:  10, activation: RELU
- Fully connected: neurons:   1 (output)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I used the sample data from udacity lecture. As the majority (80%) of the data consists of the straight driving, (with steering angle less than 0.05), I used 20% of them so that the data set can be balanced. 


For every position I also used left and right cameras of the car. For the left camera image I added 0.2 to the steering angle whereas for the right camera image I substracted 0.2 from the original angle. (model.py line 38, 44).

Here is the example of three camera images from left, center and right at the same sample time.

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped images and angles thinking that this would make the data more generalized.

After the collection process, I had 3291 number of data points. I then preprocessed this data by normalize the data and convert it from BGR color space to RGB color space, as the driver.py read the image into RGB color space (model.py line 42).


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 50 as evidenced by that the validation loss decreases slowly after 50 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
