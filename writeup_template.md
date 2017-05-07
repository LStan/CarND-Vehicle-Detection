##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog.png
[image2]: ./output_images/car_boxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in function `get_hog_features` (lines 16 through 28 of the file `P4lib.py`).  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. For example I first tried HSV color space. It showed high test accuracy for SVC, but in video it gave a lot of false positives. So I ended up using YCrCb color space (all channels). If I set number of HOG orientations > 9 it did not gave improvements, so as increasing the number of cells per block. So the final parameters are:  `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training the classifier is in `P4train.py`. First I extract all features in function `extract_features` (lines 5-53).  In addition to HOG features I used color histogram features(`hist_bins = 32`) and binned color features(`spatial_size = (16, 16)`). 
I normalize this features  with StandardScaler. Then I split them into training and test sets and train a LinearSVC. (lines 94-125)

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in function `get_heatmap` (lines 11 through 77 of the file `P4video.py`). 
The function only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%.
So, for base scale (==1) the function searches 64x64 windows. So, after cutting the image according `ystart` and `ystop` (top and bottom border for the search) I  resize the input image with a given scale. Then I get HOG features for the whole image. Then is for-loops HOG features are subsampled and combined with the spatial and color hist features. The features then normalized with X_scaler and used to predict presence/absence of a car and construct a heatmap.
After different experiments on the video I chose to use following windows sizes and search zones: 

* (128, 128) windows, ystart = 400, ystop = 700
* (96, 96) windows, ystart = 400, ystop = 656
* (64, 64) windows, ystart = 400, ystop = 500 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I summed all three heatmaps and applied a threshold of 2 (function `apply_threshold`, lines 78-82). Then I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. Then with function `draw_labeled_bboxes` (lines 84-97) I drew the boxes.  
Here are some example images:

![alt text][image2]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Method of combining bounding boxes is described above. The only difference for video is that I sum heatmaps from 14 consecutive frames(for that I use deque with maxlen=14) and apply threshold 4 (`process_frame` function in `P4video.py`) .


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I found this approach for vehicle detection unreliable. I had to fine tune the parameters in order it to work on the video. Even after tuning I had some false positives on the left part of the image and had to restrict window searching to the right part. So this is definitely not robust to different situations. Using more different window sizes would make it more robust but make it very slow. Probably, this problem is better be solved with CNN approach

