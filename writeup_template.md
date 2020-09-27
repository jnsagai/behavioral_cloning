# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center Image"
[image2]: ./examples/left_side.jpg "Left Drifted Image"
[image3]: ./examples/recovery_1.jpg "Recovery Image 1"
[image4]: ./examples/recovery_1.jpg "Recovery Image 2"
[image5]: ./examples/recovery_1.jpg "Recovery Image 3"
[image6]: ./examples/recovery_1.jpg "Recovery Image 4"
[image7]: ./examples/recovery_1.jpg "Recovery Image 5"
[image8]: ./examples/recovery_1.jpg "Recovery Image 6"
[image9]: ./examples/recovery_1.jpg "Recovery Image 7"
[image10]: ./examples/recovery_1.jpg "Recovery Image 8"
[image11]: ./examples/recovery_1.jpg "Recovery Image 9"
[image12]: ./examples/recovery_1.jpg "Recovery Image 10"
[image13]: ./examples/recovery_1.jpg "Recovery Image 11"
[image14]: ./examples/normal.jpg "Normal Image"
[image15]: ./examples/flipped.jpg "Flipped Image"
[image16]: ./examples/left.jpg "Left Image"
[image17]: ./examples/right.jpg "Right Image"
[image18]: ./examples/histogram.png "Histogram"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes for first 3 layers and 3x3 for the next 2 layers, and depths between 32 and 64 (clone.py lines 80-94) 

The model includes RELU layers to introduce nonlinearity (code line 83-87), and the data is normalized in the model using a Keras lambda layer (code line 81). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (clone.py lines 90 and 92). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 76-77). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 96).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
I drove the car for both clockwise and counter-clockwise.
In summary the following strategy was used:
 - Five laps of center lane driving in clockwise
 - Three laps of center lane driving in counter-clockwise
 - one lap of recovery driving from the sides
 - one lap focusing on driving smoothly around curves

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with something really simple like a model for linear regression with just one flatten layer. The goal here was just to understand better the pipeline that would be necessary to sucessfully get a model with acceptable perfomance.
The training data was gathered by just recording one single lap on track one.

In order to optimize the model training and also increase the performace, I normatize all the inputs images and applied a small deviation (clone.py line 81) and also cropped the image to just get the most relavant part: the road (clone.py line 82).

After that I evolve my model to a CNN ([LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) to be more specific, where the results had a significant improvement, and the vehicle as able to stay on the road for the most part of the circuit. However, I noticed that the car was biased to the left side of the road.
Then I decided to increase my training dataset by recording the car running in sereval laps ( clockwise, counter-clockwise, recoverying from the sides and driving smoothly around curves ). With this new training set I finally got a result where the car could complete an entire lap without touching the side lanes.

The performance was already acceptable, but the car moving was still oscilaty. Then I decided to implement the same model defined by [NVIDEA End to End Learning for Self-Driving Cars paper](End to End Learning for Self-Driving Cars). And, in order to combat the overfitting, I modified the model and and one dropout layer between each full connected layer.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road in a well-smooth manner.

#### 2. Final Model Architecture

The final model architecture (clone.py lines 80-94) consisted of a convolution neural network with the following layers and layer sizes:
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							| 
| Convolution 5x5     	| 2x2 stride, filter depth 24 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, filter depth 36 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, filter depth 48 	|
| RELU					|												|
| Convolution 3x3     	| 2x2 stride, filter depth 64 	|
| RELU					|												|
| Convolution 3x3     	| 2x2 stride, filter depth 64 	|
| RELU					|												|
| Fully connected		| 100 neurons        									|
| Dropout		| 20%       									|
| RELU					|												|
| Fully connected		| 50 neurons        									|
| Dropout		| 20%       									|
| RELU					|												|
| Fully connected		| 10 neurons        									|
| RELU					|												|
| MSE				| Mean Squared Error        									|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded five laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center if it drift back to the side. These images show what a recovery looks like starting from left side :

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

I used the left and right camera images to increase my dataset and also help my model to learn how to recovery its path to the center of the road. Here is an example of left, center and right camera images:

![alt text][image16]
![alt text][image1]
![alt text][image17]

To augment the data set, I also flipped images thinking that this would help my model to generalize better, and also I would double my training data set. For example, here is an image that has then been flipped:

![alt text][image14]
![alt text][image15]

After the collection process, I had 85092 number of data points. I then preprocessed this data by normalizing then and cropping the images for just capturing the road, as explained in the previous session.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by the Outputting Training and Validation Loss Metrics in the image below. I used an adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image18]
