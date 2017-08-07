# **Advanced Lane Finding Project**

The goals of the project is to build a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. 

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in [Here](https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb)  

The chessboard sample images were used for calibrating the camera. Objectpoints stores 3d points in real world while imgpoints stors 2d points in image plane. 

The assumption is that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time when the chessboard corners in a test image are successfully identified. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

The output `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The distortion correction function is applied to the test image using `cv2.undistort()`. An example result of the sample image is
 
![](camera_cal/calibration3.jpg?raw=true "Original_Image")

![](camera_cal/test_undist.jpg?raw=true "Undistorted_Image") 

## Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The distortion correction applied using cv2.undistort to one of the test images resulted in :

![](test_images/test2.jpg?raw=true "Original_Image")

![](output_images/undistorted.jpg?raw=true "Undistorted_Image") 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

A combination of color and gradient thresholds was used to generate a binary image (thresholding functions at lines 20 through 98). The thresholding preprocessing is applied at lines 158 through 161

![](output_images/threshold.jpg?raw=true "Binary_Image_after_thresholding")

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective transformation is applied in lines 175 through 179 using `cv2.getPerspectiveTransform`. The source (`src`) and destination (`dst`) points chosen are:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

The perspective transform was verified by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![](output_images/warped.jpg?raw=true "Warped_Image") 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

A convolution was applied to the sliding window method which will maximize the number of "hot" pixels in each window using `np.convolve`. A convolution is the summation of the product of two separate signals, in our case the window template and the vertical slice of the pixel image. The window template used to slide across the image from left to right and any overlapping values are summed together, creating the convolved signal. The peak of the convolved signal is where there was the highest overlap of pixels and the most likely position for the lane marker. A smoothing factor was applied to the average to avoid lane detections jumping from frame to frame. 

Using the x and y pixel positions of the located lane line pixels, a second order polynomial curve was fitted:

f(y)=Ay^​2 + By + C 

![](output_images/template.jpg?raw=true "Template") 

![](output_images/result.jpg?raw=true "Result")


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
Radius of curvature and position of the car on the road is calculated in lines 251 through 260 in the code

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of the result on a test image:

![](output_images/tracked1.jpg?raw=true "Tracked_output")

### Pipeline (video)

Here's a video output [](./project_video.mp4)

### Discussion

The mountain roads are more challenging to detect because the curvature changes more frequently. Boundary region needs to be constrained to a much smaller area. Also need a way to distinguish carpool sign, since the signs were getting misinterpreted as edge of the right lane in challenge video.

Here's a challenge video output [](./challenge_output.mp4)
Here's a harder challenge video output [](./harder_challenge_output.mp4)