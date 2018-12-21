## Vehicle Detection and Tracking

---
**The goals / steps of this project are the following:**

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
    * Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
    * Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: output_images/20_sample_car_images.png
[image2]: output_images/20_sample_non_car_images.png
[image3]: output_images/sample_hog_images.png
[image4]: output_images/test_accu_vs_predict_time_1000_samples.png
[image5]: output_images/test_images_vis.png
[image6]: output_images/cars_found.png
[image7]: output_images/roi.png
[image8]: output_images/400_500_1.png
[image9]: output_images/400_500_1-25.png
[image10]: output_images/400_500_1-5.png
[image11]: output_images/400_464_1.png
[image12]: output_images/416_480_1.png
[image13]: output_images/400_496_1-5.png
[image14]: output_images/432_528_2.png
[image15]: output_images/432_528_1-5.png
[image16]: output_images/400_528_2.png
[image18]: output_images/400_528_1-75.png
[image19]: output_images/432_560_2.png
[image20]: output_images/400_596_3-5.png
[image21]: output_images/464_660_3-5.png
[image22]: output_images/final_frame_parameters.png
[image23]: output_images/heat_mapped_1.png
[image24]: output_images/heat_map_thresholded_1.png
[image25]:output_images/bounding_boxes.png
[image26]:output_images/processed_imgs_vis.png


[video1]: project_video_out.mp4

---
## 1. Files Submitted

The required files can be referenced at :
* [Jupyter Notebook with Code](vehicle_detection_and_tracking.ipynb)
* [HTML output of the code](vehicle_detection_and_tracking.html)
* [Helpers.py](helpers.py)
* [Project Output Video](project_video_out.mp4)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I first loaded the provided data set for vehicles and non-vehicles and visualized some random sample images from both classes as shown below:
* **8792 Car Images**
* **8968  Non-Car Images were loaded.**

**Visualizing Sample Car Images**

![alt text][image1]

**Visualizing Sample Non-Car Images**

![alt text][image2]


The code for extracting HOG features is defined by the method `get_hog_features` and is contained in the cell titled "**Function to convert image to Histogram of Oriented Gradients (HOG)**" I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:

![alt text][image3]

Next, in the section titled "**Parameter Exploration using Design of Experiments**", I defined the following:
* HOG feature extraction parameters for exploration
* Their value ranges to be explored
* Possible parameter combinations for experimentation

Using `pyDOE` Design of Experiments library function -  `fullfact()` I derived all the possible configurations using the chosen parameters and their chosen values. These configurations were then explored using the method `parameter_exploration()` wherein various features were extracted as per the input configuration being looped through.
The `extract_features()` method was used to extract features mainly using 3 methods:
* `bin_spatial()`
* `color_hist()`
* `get_hog_features()`

These feature sets are combined and a label vector is defined (1 for cars, 0 for non-cars). The features and labels are then shuffled and split into training and test sets in preparation to be fed to a **SVM Classifier**. I ran these configurations only a sample set of 1000 images each of car and non-cars. This was to save time that would have been otherwise spent on training so many configurations. Also, the point of this trial run was more of a comparative run so results should be independent of the sample size. 
**The table below documents the 192 different parameter combinations that I explored and their results.**

Colorspace|HOG Channel|Orientations|Pixels/Cell|Cells/Block|Feature Compute Secs|Feature Vector Length|Training Secs|Test Accuracy|Predict Secs
---|---|---|---|---|---|---|---|---|---
RGB|0|10|12|2|7.122|3808|5.1|0.938|0.0002
HSV|0|10|12|2|4.707|3808|1.73|0.938|0.00018
LUV|0|10|12|2|4.672|3808|2.32|0.932|0.00017
HLS|0|10|12|2|5.144|3808|1.61|0.938|0.00016
YUV|0|10|12|2|4.617|3808|3.07|0.95|0.00015
YCrCb|0|10|12|2|4.532|3808|3.25|0.94|0.00013
RGB|1|10|12|2|4.497|3808|4.91|0.922|0.00016
HSV|1|10|12|2|4.556|3808|1.54|0.928|0.00018
LUV|1|10|12|2|4.797|3808|3.36|0.96|0.00024
HLS|1|10|12|2|4.721|3808|2.68|0.955|0.00013
YUV|1|10|12|2|4.684|3808|2.98|0.92|0.00016
YCrCb|1|10|12|2|4.714|3808|2.3|0.955|0.00014
RGB|2|10|12|2|4.459|3808|4.44|0.922|0.00015
HSV|2|10|12|2|4.556|3808|1.42|0.938|0.00015
LUV|2|10|12|2|4.814|3808|2.14|0.932|0.00018
HLS|2|10|12|2|4.674|3808|2.36|0.945|0.00017
YUV|2|10|12|2|4.714|3808|1.33|0.932|0.00023
YCrCb|2|10|12|2|4.745|3808|2.85|0.938|0.00023
RGB|ALL|10|12|2|9.631|5088|7.15|0.928|0.00023
HSV|ALL|10|12|2|9.993|5088|1.44|0.95|0.0002
LUV|ALL|10|12|2|9.771|5088|0.95|0.968|0.00027
HLS|ALL|10|12|2|9.664|5088|1.47|0.955|0.00019
YUV|ALL|10|12|2|9.69|5088|3.16|0.97|0.00025
YCrCb|ALL|10|12|2|9.592|5088|5.98|0.982|0.00025
RGB|0|12|12|2|4.637|3936|5.28|0.935|0.00022
HSV|0|12|12|2|4.903|3936|2.09|0.94|0.00017
LUV|0|12|12|2|4.884|3936|3.4|0.962|0.00021
HLS|0|12|12|2|5.064|3936|1.36|0.94|0.0002
YUV|0|12|12|2|4.714|3936|4.48|0.938|0.00018
YCrCb|0|12|12|2|4.692|3936|4.57|0.91|0.00019
RGB|1|12|12|2|4.49|3936|4.7|0.932|0.00013
HSV|1|12|12|2|4.665|3936|2.28|0.952|0.00015
LUV|1|12|12|2|4.762|3936|1.64|0.948|0.00018
HLS|1|12|12|2|4.65|3936|2.29|0.945|0.00016
YUV|1|12|12|2|4.765|3936|2.62|0.938|0.00018
YCrCb|1|12|12|2|4.568|3936|3.95|0.942|0.00018
RGB|2|12|12|2|4.542|3936|4.84|0.925|0.00023
HSV|2|12|12|2|4.732|3936|2.24|0.962|0.0002
LUV|2|12|12|2|4.8|3936|2.05|0.945|0.00017
HLS|2|12|12|2|4.844|3936|1.84|0.94|0.00019
YUV|2|12|12|2|4.638|3936|4.2|0.95|0.00017
YCrCb|2|12|12|2|4.668|3936|2.99|0.925|0.00018
RGB|ALL|12|12|2|9.382|5472|7.09|0.932|0.00024
HSV|ALL|12|12|2|9.592|5472|1.22|0.965|0.00024
LUV|ALL|12|12|2|9.676|5472|1.2|0.97|0.00024
HLS|ALL|12|12|2|9.511|5472|1.48|0.965|0.0002
YUV|ALL|12|12|2|9.469|5472|5.27|0.965|0.00021
YCrCb|ALL|12|12|2|9.387|5472|1.86|0.985|0.00022
RGB|0|10|16|2|4.338|3528|4.34|0.92|0.00016
HSV|0|10|16|2|4.331|3528|1.33|0.905|0.00013
LUV|0|10|16|2|4.293|3528|3.16|0.942|0.00019
HLS|0|10|16|2|4.402|3528|3.03|0.928|0.00013
YUV|0|10|16|2|4.216|3528|3.39|0.925|0.00015
YCrCb|0|10|16|2|4.407|3528|4.46|0.918|0.00015
RGB|1|10|16|2|4.202|3528|4.3|0.945|0.00012
HSV|1|10|16|2|4.344|3528|2.18|0.95|0.00016
LUV|1|10|16|2|4.303|3528|3.35|0.958|0.00015
HLS|1|10|16|2|4.129|3528|1.8|0.935|0.00013
YUV|1|10|16|2|4.105|3528|3.11|0.925|0.00014
YCrCb|1|10|16|2|4.226|3528|3.48|0.94|0.00013
RGB|2|10|16|2|4.057|3528|3.86|0.93|0.00016
HSV|2|10|16|2|4.196|3528|2.47|0.965|0.00014
LUV|2|10|16|2|4.271|3528|2.62|0.93|0.00012
HLS|2|10|16|2|4.238|3528|2.0|0.94|0.00014
YUV|2|10|16|2|4.18|3528|3.84|0.962|0.00012
YCrCb|2|10|16|2|4.351|3528|3.27|0.915|0.00012
RGB|ALL|10|16|2|7.957|4248|5.25|0.942|0.00017
HSV|ALL|10|16|2|8.378|4248|0.92|0.968|0.00015
LUV|ALL|10|16|2|8.269|4248|1.9|0.952|0.00014
HLS|ALL|10|16|2|8.248|4248|1.52|0.94|0.00016
YUV|ALL|10|16|2|8.181|4248|3.39|0.975|0.00014
YCrCb|ALL|10|16|2|8.163|4248|3.51|0.972|0.0002
RGB|0|12|16|2|4.196|3600|4.27|0.93|0.00014
HSV|0|12|16|2|4.313|3600|2.11|0.965|0.00012
LUV|0|12|16|2|4.27|3600|2.64|0.918|0.00014
HLS|0|12|16|2|4.376|3600|1.29|0.932|0.00014
YUV|0|12|16|2|4.31|3600|3.22|0.94|0.00016
YCrCb|0|12|16|2|4.319|3600|3.71|0.922|0.00016
RGB|1|12|16|2|4.22|3600|4.25|0.938|0.00012
HSV|1|12|16|2|4.344|3600|2.69|0.958|0.00014
LUV|1|12|16|2|4.375|3600|2.3|0.94|0.00016
HLS|1|12|16|2|4.23|3600|3.08|0.94|0.00013
YUV|1|12|16|2|4.228|3600|4.35|0.922|0.00013
YCrCb|1|12|16|2|4.268|3600|3.65|0.958|0.00013
RGB|2|12|16|2|4.179|3600|4.02|0.915|0.00012
HSV|2|12|16|2|4.195|3600|1.87|0.945|0.00015
LUV|2|12|16|2|4.348|3600|2.53|0.945|0.00013
HLS|2|12|16|2|4.297|3600|1.72|0.955|0.00017
YUV|2|12|16|2|4.231|3600|3.78|0.94|0.00014
YCrCb|2|12|16|2|4.31|3600|4.0|0.915|0.00013
RGB|ALL|12|16|2|8.542|4464|4.96|0.945|0.00016
HSV|ALL|12|16|2|8.373|4464|0.86|0.958|0.00019
LUV|ALL|12|16|2|8.515|4464|2.13|0.982|0.00016
HLS|ALL|12|16|2|8.418|4464|1.93|0.932|0.00015
YUV|ALL|12|16|2|8.366|4464|4.25|0.975|0.00016
YCrCb|ALL|12|16|2|8.338|4464|1.83|0.965|0.00015
RGB|0|10|12|3|4.127|3978|4.76|0.928|0.00016
HSV|0|10|12|3|4.233|3978|2.29|0.962|0.00015
LUV|0|10|12|3|4.233|3978|4.5|0.962|0.00015
HLS|0|10|12|3|4.242|3978|1.47|0.945|0.0002
YUV|0|10|12|3|4.151|3978|4.19|0.905|0.00016
YCrCb|0|10|12|3|4.14|3978|4.87|0.952|0.00015
RGB|1|10|12|3|4.094|3978|4.96|0.942|0.00015
HSV|1|10|12|3|4.229|3978|5.36|0.948|0.00016
LUV|1|10|12|3|4.224|3978|1.48|0.962|0.00015
HLS|1|10|12|3|4.107|3978|3.21|0.92|0.00019
YUV|1|10|12|3|4.148|3978|4.14|0.928|0.00014
YCrCb|1|10|12|3|4.147|3978|3.76|0.965|0.00015
RGB|2|10|12|3|4.119|3978|4.82|0.932|0.00015
HSV|2|10|12|3|4.111|3978|3.27|0.96|0.00014
LUV|2|10|12|3|4.202|3978|3.24|0.932|0.00014
HLS|2|10|12|3|4.165|3978|2.52|0.93|0.00014
YUV|2|10|12|3|4.195|3978|1.57|0.962|0.00014
YCrCb|2|10|12|3|4.203|3978|4.47|0.935|0.00015
RGB|ALL|10|12|3|8.108|5598|6.92|0.94|0.00023
HSV|ALL|10|12|3|8.258|5598|1.04|0.952|0.00019
LUV|ALL|10|12|3|8.152|5598|1.15|0.97|0.00019
HLS|ALL|10|12|3|8.201|5598|1.26|0.94|0.00019
YUV|ALL|10|12|3|8.194|5598|2.84|0.965|0.00021
YCrCb|ALL|10|12|3|8.257|5598|5.65|0.975|0.0002
RGB|0|12|12|3|4.123|4140|4.95|0.952|0.00015
HSV|0|12|12|3|4.365|4140|2.12|0.95|0.00019
LUV|0|12|12|3|4.295|4140|2.95|0.952|0.00018
HLS|0|12|12|3|4.354|4140|2.87|0.938|0.00018
YUV|0|12|12|3|4.147|4140|3.67|0.935|0.00016
YCrCb|0|12|12|3|4.095|4140|5.15|0.922|0.00016
RGB|1|12|12|3|4.135|4140|5.19|0.945|0.00014
HSV|1|12|12|3|4.24|4140|1.64|0.96|0.00016
LUV|1|12|12|3|4.312|4140|1.78|0.955|0.00017
HLS|1|12|12|3|4.156|4140|4.45|0.938|0.00017
YUV|1|12|12|3|4.167|4140|2.47|0.918|0.00019
YCrCb|1|12|12|3|4.175|4140|3.69|0.948|0.00015
RGB|2|12|12|3|4.17|4140|5.15|0.92|0.00014
HSV|2|12|12|3|4.245|4140|4.09|0.94|0.00018
LUV|2|12|12|3|4.313|4140|2.71|0.95|0.00015
HLS|2|12|12|3|4.33|4140|2.77|0.938|0.00014
YUV|2|12|12|3|4.302|4140|1.04|0.95|0.00015
YCrCb|2|12|12|3|4.177|4140|2.01|0.932|0.00014
RGB|ALL|12|12|3|8.07|6084|4.59|0.952|0.00022
HSV|ALL|12|12|3|8.407|6084|1.38|0.978|0.00022
LUV|ALL|12|12|3|8.355|6084|5.97|0.982|0.0002
HLS|ALL|12|12|3|8.241|6084|1.49|0.965|0.00019
YUV|ALL|12|12|3|8.217|6084|1.83|0.965|0.00019
YCrCb|ALL|12|12|3|8.11|6084|5.84|0.945|0.0002
RGB|0|10|16|3|3.933|3528|3.96|0.932|0.00015
HSV|0|10|16|3|4.01|3528|5.08|0.95|0.00014
LUV|0|10|16|3|4.139|3528|2.62|0.942|0.00014
HLS|0|10|16|3|4.027|3528|2.03|0.928|0.00014
YUV|0|10|16|3|3.983|3528|3.81|0.928|0.00012
YCrCb|0|10|16|3|3.963|3528|3.49|0.92|0.00013
RGB|1|10|16|3|3.847|3528|4.27|0.958|0.00016
HSV|1|10|16|3|4.004|3528|2.14|0.925|0.00015
LUV|1|10|16|3|4.068|3528|2.57|0.95|0.00013
HLS|1|10|16|3|4.083|3528|3.18|0.96|0.00013
YUV|1|10|16|3|3.992|3528|4.88|0.945|0.00013
YCrCb|1|10|16|3|3.969|3528|3.64|0.925|0.00013
RGB|2|10|16|3|3.929|3528|4.54|0.945|0.00016
HSV|2|10|16|3|4.035|3528|3.08|0.942|0.00015
LUV|2|10|16|3|4.381|3528|3.62|0.905|0.00016
HLS|2|10|16|3|4.082|3528|4.57|0.91|0.00014
YUV|2|10|16|3|3.99|3528|2.53|0.95|0.00012
YCrCb|2|10|16|3|3.899|3528|3.75|0.92|0.00018
RGB|ALL|10|16|3|7.432|4248|4.98|0.938|0.00015
HSV|ALL|10|16|3|7.618|4248|1.06|0.955|0.0002
LUV|ALL|10|16|3|7.557|4248|4.4|0.97|0.00018
HLS|ALL|10|16|3|7.551|4248|1.67|0.94|0.00017
YUV|ALL|10|16|3|7.463|4248|4.39|0.972|0.00021
YCrCb|ALL|10|16|3|7.399|4248|3.16|0.96|0.00017
RGB|0|12|16|3|4.058|3600|4.7|0.915|0.00019
HSV|0|12|16|3|4.18|3600|2.31|0.938|0.00015
LUV|0|12|16|3|4.146|3600|2.45|0.945|0.00013
HLS|0|12|16|3|4.152|3600|2.06|0.92|0.00018
YUV|0|12|16|3|3.998|3600|3.56|0.942|0.00015
YCrCb|0|12|16|3|4.142|3600|3.92|0.928|0.00019
RGB|1|12|16|3|4.034|3600|4.34|0.942|0.00014
HSV|1|12|16|3|4.355|3600|2.62|0.932|0.00015
LUV|1|12|16|3|4.273|3600|3.95|0.968|0.00015
HLS|1|12|16|3|4.107|3600|2.49|0.94|0.00016
YUV|1|12|16|3|4.219|3600|3.55|0.918|0.00014
YCrCb|1|12|16|3|3.999|3600|3.31|0.955|0.00015
RGB|2|12|16|3|3.991|3600|4.4|0.95|0.00014
HSV|2|12|16|3|4.027|3600|1.96|0.938|0.00016
LUV|2|12|16|3|4.106|3600|2.37|0.932|0.00016
HLS|2|12|16|3|4.195|3600|3.38|0.928|0.00014
YUV|2|12|16|3|4.003|3600|3.63|0.96|0.00017
YCrCb|2|12|16|3|4.027|3600|3.94|0.935|0.00015
RGB|ALL|12|16|3|7.663|4464|4.38|0.95|0.00021
HSV|ALL|12|16|3|7.966|4464|1.13|0.968|0.00017
LUV|ALL|12|16|3|7.762|4464|1.02|0.972|0.00018
HLS|ALL|12|16|3|7.694|4464|1.64|0.952|0.00018
YUV|ALL|12|16|3|7.894|4464|4.78|0.975|0.00018
YCrCb|ALL|12|16|3|8.019|4464|5.15|0.972|0.00021


#### 2. Explain how you settled on your final choice of HOG parameters.

I settled on my final choice of HOG parameters based upon the performance of the SVM classifier produced using them. I considered not only the accuracy with which the classifier made predictions on the test dataset, but also the speed at which the classifier is able to make predictions (Tested the prediction time for 20 sample images). There is a balance to be struck between accuracy and speed of the classifier, and my strategy was to bias toward speed first, and achieve as close to real-time predictions as possible, and then pursue accuracy if the detection pipeline were not to perform satisfactorily. The data above is plotted in a scatter graph below to visualize how spread out the results are of the various combinations of the paramters, which signifies how crucial their tuning becomes in classification. 

![alt text][image4]

Hence I shortlisted the candidates that attained more than 98% Test Accuracy as marked in the graph above and as outlined in the table below:

Colorspace|HOG Channel|Orientations|Pixels/Cell|Cells/Block|Feature Compute Secs|Feature Vector Length|Training Secs|Test Accuracy|Predict Secs
---|---|---|---|---|---|---|---|---|---
LUV|ALL|12|16|2|8.515|4464|2.13|0.982|0.00016
LUV|ALL|12|12|3|8.355|6084|5.97|0.982|0.0002
YCrCb|ALL|12|12|2|9.387|5472|1.86|0.985|0.00022
YCrCb|ALL|10|12|2|9.592|5088|5.98|0.982|0.00025

I decided to further shortlist and chose the 1st and 3rd configuration as they had highest Test Accuracy and lowest Training and Prediction time. I then decided to run them again through the `parameter_exploration()` method but this time on complete dataset to make a more informed decision. The results looked like this:

Colorspace|HOG Channel|Orientations|Pixels/Cell|Cells/Block|Feature Compute Secs|Feature Vector Length|Training Secs|Test Accuracy|Predict Secs
---|---|---|---|---|---|---|---|---|---
LUV|ALL|12|16|2|95.384|4464|24.35|0.98|0.00198
YCrCb|ALL|12|12|2|102.318|5472|51.71|0.9797|0.00658

**The final choice was obvious due to its incredibly fast Training and Prediction times at 98% Accuracy**

Colorspace|Spatial Features|Spatial Size|Hist Features|Hist Bins|Hog Feature|HOG Channel|Orientations|Pixels/Cell|Cells/Block
---|---|---|---|---|---|---|---|---|---
LUV|True|(32, 32)|True|32|True|ALL|12|16|2



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the section titled "**Parameter Exploration using Design of Experiments**" I used a method `parameter_exploration()` within which I trained a linear SVM with the default classifier parameters and using above mentioned selected HOG feature parameters, `spatial_size = (32,32)` and `hist_bins = 32` and was able to achieve a test accuracy of 98%. 
>**The data for trial runs was stored in pickle files in this [FOLDER](data) and the final model is contained in this [FILE](data/doe_results_3.p)**

---
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

**Loading and Visualizing Test Images**
![alt text][image5]

In the section titled "**Classifier to Detect Cars in an Image**" I adapted the method `find_cars()` from the lesson materials. The method combines HOG feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image (or a selected portion of it) and then these full-image features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

**I first identified the area in which the cars can possibly appear. Shown below:**
![alt text][image7]

**I then explored several configurations of window sizes and positions, with various overlaps in the X and Y directions. The following images show the configurations of some of the search windows I tried for small (1x, 1.25x), medium (1.5x, 1.75x, 2x), and large (3x, 3.5x) windows:**
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image21]


The final algorithm calls `find_cars()` for each window scale and the rectangles returned from each method call are aggregated. In previous implementations smaller (0.5,0.75) scales were explored but found to return too many false positives, and originally the window overlap was set to 50% in both X and Y directions, but an overlap of 75% in the Y direction (yet still 50% in the X direction) produced more redundant true positive detections, which were preferable given the heatmap strategy described below. Additionally, only an appropriate vertical range of the image is considered for each window size (e.g. smaller range for smaller scales) to reduce the chance for false positives in areas where cars at that scale are unlikely to appear. The final implementation proves to be robust enough to reliably detect vehicles while maintaining a high speed of execution.

```python
frame_param = [(400,464,1),(416,480,1),(400,496,1.5),(432,528,1.5),
               (400,528,2),(432,560,2),(400,596,3.5),(464,660,3.5)]
```
               
The image below shows the rectangles returned by `find_cars()` drawn onto one of the test images in the final implementation. Notice that there are several positive predictions on each of the near-field cars, and one positive prediction on a car in the oncoming lane.

![alt text][image22]

Because a true positive is typically accompanied by several positive detections, while false positives are typically accompanied by only one or two detections, a combined heatmap and threshold is used to differentiate the two. The add_heat function increments the pixel value (referred to as "heat") of an all-black image the size of the original image at the location of each detection rectangle. Areas encompassed by more overlapping rectangles are assigned higher levels of heat. The following image is the resulting heatmap from the detections in the image above:

![alt text][image23]

A threshold is applied to the heatmap (in this example, with a value of 1), setting all pixels that don't exceed the threshold to zero. The result is below:

![alt text][image24]

The scipy.ndimage.measurements.label() function collects spatially contiguous areas of the heatmap and assigns each a label:

![alt text][image6]

And the final detection area is set to the extremities of each identified label using `draw_labeled_bboxes()`:

![alt text][image25]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

My extensive start with exploring 192 configurations of parameters gave me very accurate detections including small cars on the other side of the lane and that too at a very high prediction speed. The test images and vidoes gave me  almost no false positives but the actual project video had this one section near the barrier separating the two lanes which gave me almost constant false positives. This got me to explore the frame start stop position and scaling towards the middle of the images where the road vanishes in the frames. This is where my model was constantly identifying cluster on incoming cars from the other side of the lane and cars diminishing on the same the side. This cluster was forming a big bounding box and was eliminated to an extent after exploring more window search parameters. In future i would like to develop a systematic approach to figure out the optimum parameters for the window search just like i did for the HOG Parameters.

Also other optimization techniques included changes to window sizing and overlap as described above, and lowering the heatmap threshold to improve accuracy of the detection (higher threshold values tended to underestimate the size of the vehicle). 

**Here are some example image outputs from `Process Pipeline()`:**

![alt text][image26]

---
### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in test images.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Above is an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the test images. 

Also, the code for processing frames of video is contained in the cell titled "**Enhanced Process Pipeline**" and is identical to the code for processing a single image described above in **`Process Pipeline()`** , with the exception of storing the detections, returned by `find_cars()` from the previous 2 frames of video using the `prev_rects` parameter from a class called `Vehicle_Detect()`. This records heat maps from frame to frame for a given time frame. The augmented heatmap of these frames is then thresholded to better eliminate false positives as their heat map weakens out over several frames of no detection. Also the accumulated heat map helps better locate the centroid of the bounding box and its boundaries to tightly wrap the car. Rather than performing the heatmap/threshold/label steps for the current frame's detections, the detections for the past 2 frames are combined and added to the heatmap and the threshold value of 2 is applied. This value is higher than used while testing the method on test video. A lower threshold value of 1 was raising many false positives. Also, suprisingly accumulating more than 2 frames of data was creating a huge rectangle on the other side of the lane as many cars were quickly passing by from that side and a lot of heatmaps were getting accumulated close by on a larger area if the data was stored for several frames. Using only 2 frames of history negatively impacts the smoothness of plotting around the slow moving cars in the same lane.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems that I faced while implementing this project were mainly concerned with detection accuracy. Balancing the accuracy of the classifier with execution speed was crucial. Scanning around 200 windows using a classifier that achieves 98% accuracy should result in around 4 misidentified windows per frame. Of course, integrating detections from previous frames mitigates the effect of the misclassifications, but it also introduces another problem: vehicles that significantly change position from one frame to the next (e.g. oncoming traffic) will tend to escape being labeled. Producing a very high accuracy classifier and maximizing window overlap might improve the per-frame accuracy to the point that integrating detections from previous frames is unnecessary (and oncoming traffic is correctly labeled), but it would also be far from real-time without massive processing power.

The pipeline is probably most likely to fail in cases where vehicles (or the HOG features thereof) don't resemble those in the training dataset, but lighting and environmental conditions might also play a role (e.g. a white car against a white background). As stated above, oncoming cars are an issue, as well as distant cars (as mentioned earlier, smaller window scales tended to produce more false positives, but they also did not often correctly label the smaller, distant cars).

I believe that the best approach, given plenty of time to pursue it, would be to combine a very high accuracy classifier with high overlap in the search windows. The execution cost could be offset with more intelligent tracking strategies, such as:

* Determine vehicle location and speed to predict its location in subsequent frames
* Begin with expected vehicle locations and nearest (largest scale) search areas, and preclude overlap and redundant detections from smaller scale search areas to speed up execution
* Use a convolutional neural network, to preclude the sliding window search altogether

Thank you !
