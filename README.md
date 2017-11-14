# Traffic Sign Recognition Classifier
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[test1]: ./test_images/speed_limit_50_photo.jpg "Speed limit"
[test2]: ./test_images/speed_limit_50.jpg "Speed limit"

[imgDataset]: ./report_images/exampledata.png "Example"
[imgAugmented]: ./report_images/augmented.png "Augemented image"

[label_dist1]: ./report_images/labels1.png "Distribution of labels"
[label_dist2]: ./report_images/labels2.png "Distribution of labels after augmentation"


### Dependencies
This notebook requires:
- Python 3.3 or above
- Tensorflow 1.3.0 or above
- OpenCV
- Pandas

The easiest way to get started is using the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) and upgrading/ installing the above packages. The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The notebook should be run using GPUs. If local GPUs are not available consider using a web service e.g. AWS. On AWS, there is a Community AMI "Udacity self-driving car" which can be used after upgrading to a newer version of tensorflow.

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

In the following 15 random images of the dataset are displayed.
![alt text][imgDataset]


####2. Exploratory visualization of the dataset.

![alt text][label_dist1]

The labels are not equality distributed over the dataset. The data set consists to 5.8% of the traffic sign "speed limit (50km/h)" but only to 0.5% of the traffic sign "speed limit (20km/h)". All over traffic signs are contained in the between interval.
Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed over the labels. The traffic sign "speed limit (50km/h)" is the most common sign while the traffic sign "speed limit (20km/h)" is the least common one. Hence, the model might get biased towards frequent labels. This could be checked for example with the recall and precision metrics (but is not done here). Adding more images with less frequent labels can circumvent this potential biases and has a side effect that the model is less likely to overfit to the data due to the larger dataset.


### Design and Test a Model Architecture

####1. Data augmentation
The dataset was augmentated by randomly modified copies of the original images. Tensorflow's own data augementaion functions were used to this end.

Function1
Function2

Every label was augemented by at least xx images until the label contained at least xx images.
After augementaion, the training data contained ... images. The labels are distributed more equally now.

Here is an example of a randomly modified image (right) in comparison to the original image (left):

![alt text][imgAugemented]

####1. Image preprocessing

The image were converted to grayscale and then normalized using
'(data - np.array(data).mean()) / np.array(data).std()'

Here is an example of a traffic sign image before and after preprocessing.

![alt text][imgPreprocessed]

I used grayscaling to reduce the dimensionality of the data. However I considered adding the red and blue channels after thresholding since the traffic signs consist mainly of the colours black, white, red and blue.
The normalization is necessary so that the weights can be similar for every image and that the model does not get biased towards particularly "bright" images.

####2. Model architecture 
![LeNet Architecture](examples/lenet.png)
Source: Yan LeCun

The LeNet Convolutional network was adapted to the traffic sign recognition problem by changing the output classes to 43 and adding dropout layers in the fully connected layers in order to prevent overfitting. The resulting model architecture is the following:

| Layer         		 |     Description	         | Input | Output | Parameters |
|:----------------------:|:-------------------------:|:-------: | :--------:|:----------:|
| Convolution 5x5        | 1x1 stride, valid padding | 32x32x1  | 28x28x6   | 582  |
| Activation			 | ReLU					     |          |           |      |
| Max pooling	      	 | 2x2 stride				 | 28x28x6  | 14x14x6   |      |
| Convolution 5x5        | 1x1 stride, valid padding | 14x14x6  | 10x10x12  |      |
| Activation			 | ReLU					     |          |           |      |
| Max pooling	      	 | 2x2 stride				 | 10x10x12 | 5x5x12    |      |
| Flatten                | from 3D to 1D             | 5x5x12   | 400       |      |
| Fully connected		 |          				 | 400      | 200       |      |
| Activation			 | ReLU					     |          |           |      |
| Dropout                | Probability: 75%          |          |           |      |          
| Fully connected		 |          				 | 200      | 84        |      |
| Activation			 | ReLU					     |          |           |      |
| Dropout                | Probability: 75%          |          |           |      |
| Fully connected(Logits)|							 | 84       | 43        |      |
| Softmax                |                           |          |           |      |
|:----------------------:|:-------------------------:|:-------: | :--------:|:----------:|
| Total                  |                           |          |           |  ???    |


I decided to use the basic LeNet model because my own modifications (inception layers, deeper networks,...) did not improve the accuracy significantly. Hence, I prefered using a simple model and rather finish the project than searching for the perfect solution. The LeNet architecture is already suited to the image dimensions and thus needs only few adaptions. Since more output classes (43 instead of 10) are present, a deeper or more complex convolutional network (e.g. based on AlexNet) would be better suited.


####3. Model training
The model was trained with gradient descent. After every epoch the loass of the cross entropy is calculated and then minimized using the Adam optimizer.
The following hyperparameters were used for training as they showed the best trade-off between overfitting (dropout,...), oscillating accuracy (large learning rate, small batch number,...) and training time (number of epochs,...). 

| Hyperparameter | Value  | 
|:--------------:|:------:|
| Learning rate  | 0.0008 |
| Dropout	     | 0.75   | 
| Batch size     | 512    |
| Epochs         | 30     |


####4. Model validation
e.g. training accuracy >> validation accuracy)

After finding the right steps for image augmentation and preprocessing, only the hyperparameters were tuned. I tried several combinations and checked the behavior of the model for
- overfitting e.g. if training accuracy >> validation accuracy >> test accuracy -> increase dropout
- oscillation of the accuracy -> decrease learning rate, increase batch size
- saturation of the accuracy -> increase learning rate, increase number of epochs
- training time -> decrease number of epochs, decrease batch size (more parallelisation)


My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?


### Test the Model on New Images

####1. German traffic signs from the internet
The test images are a mixture of photos and pictograms at different cropings and perspectives. Also modified versions of the original traffic signs are present. I selected the following 15 traffic signs:

![alt text][test1] ![alt text][test2] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Short overview table of the traffic signes

|Image| Difficulty|
|:--:|:--:|
|image|comment|

The first image is a photo "Speed limit (50km/h)" and the second image is pictogram of the same sign. This is the sign type which is most present in the training set (5.8%) and it should be easy for the model to predict them correctly.

The third image is a no pasing.

The fourth image is a speed limit (20km/h) which is the least present sign type in the training set.

The fifth image is a photo of a "no entry" sign which was modified in a funny way. Such modifications could be caused by obscured sight in reality.


####2. Predictions of test images
Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| Probability|
|:---------------------:|:---------------------------------------------:| :---:|
| Stop Sign      		| Stop sign   									| |
| U-turn     			| U-turn 										||
| Yield					| Yield											||
| 100 km/h	      		| Bumpy Road					 				||
| Slippery Road			| Slippery Road      							||


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
was not done yet


### Appendix: Calculation of model parameters
The dimensions of every layer can be caluclated according to the following equations:

Input size = input_width x input_height, x input_depth (here: 32x32x3)

Variables:
* Input size $H_{in}$ x $W_{in}$ x $D_{in}$
* Output size $H_{out}$ x $W_{out}$ x $D_{out}$
* Filter size $H_{filter}$ x $W_{filter}$ x $N_{filter}$
* Stride $S$
* Padding $P$ ('VALID':0, 'SAME':?, 'ZERO':?)

(Width and hights are symmetric here and thus only the caluclation of the width is shown in the following.)

1. Convolution Layer
    * Dimensions:
    \begin{equation}
        W_{out} = \dfrac{W_{in} − W_{filter} + 2P}{S} + 1
    \end{equation}
    \begin{equation}
        D_{out} = N_{filter}
    \end{equation}

    * Number of Parameters in this layer (parameter sharing!):
    \begin{align}
        NP_{layer} &= NP_{filter} \cdot N_{filter} + NP_{bias} \\
                    &=  (W_{filter} \cdot H_{filter} \cdot N_{filter} + 1) \cdot D_{out}
    \end{align}

2. Pooling Layer
Pooling is as a convolution with a single filter. However, the filter is not defined by weights but by chosing the maximum/average/... . Hence, the same equation holds.

    \begin{equation}
        W_{out} = \dfrac{W_{in} − W_{filter} + 2P}{S} + 1
    \end{equation}
    \begin{equation}
        D_{out} = D_{in}
    \end{equation}

3. Fully Connected Layer

    Total parameters:
    Number of Parameters $NP$
    \begin{align}
        NP_{layer} &= NP_{filter} \cdot N_{output,neurons} + NP_{bias} \\
                    &=  (W_{filter} \cdot H_{filter} \cdot N_{filter} + 1) \cdot W_{out} \cdot H_{out} \cdot D_{out}
    \end{align}

