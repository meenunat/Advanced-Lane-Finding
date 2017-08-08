# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:39:24 2017

@author: meenu
"""
import numpy as np
import cv2
import glob
import pickle
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

#Camera Calibration
dist_pickle = pickle.load(open( "./camera_cal/calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"] 
dist = dist_pickle["dist"] 

#Gradient thresholding and Color thresholding functions
def abs_sobel_thresh(img, orient='x', thresh=(0,255)):
    # Apply the following steps to img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary_output
    
def color_threshold(img, sthresh=(0, 255), vthresh=(0, 255)):
    # 1) Convert to HLS color space and separate S Channel
    # 2) Apply a threshold to the S channel
    # 3) Convert to HSV color space and separate V Channel
    # 4) Apply a threshold to the V channel
    # 3) Return a binary image of combined S and V channel threshold result
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1
    
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_binary==1)&(v_binary==1)] = 1
    return binary_output

#Function to draw window areas
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output    

#Function for Identifying lane points    
def find_window_centroids(warped, window_width, window_height, margin):
    recent_centers = [] # List that stores all the values (left,right) window centroid positions for smoothing the output
    window_centroids = [] # Store the (left,right) window centroid positions per level
    smooth_factor = 15 #smoothing factor to avoid  line detections will jump around from frame to frame a bit
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))
     
    recent_centers.append(window_centroids)
    return np.average(recent_centers[-smooth_factor:],axis=0)

def process_image(img , t_type = 0):
    # window settings
    window_width = 25
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 25 # How much to slide left and right for searching
  
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    
    img_size = (img.shape[1], img.shape[0])
    # undistort the image
    img = cv2.undistort(img,mtx,dist,None,mtx)  
   
    #Process image and generate binary pixels of interests
    # Apply each of the thresholding functions
    preprocessImage = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12, 255))
    grady = abs_sobel_thresh(img, orient='y', thresh=(25, 255))
    c_binary = color_threshold(img, sthresh=(100, 255), vthresh=(50, 255))
    preprocessImage[((gradx==1) & (grady==1)) |(c_binary==1)] = 255
           
    src = np.float32(
                [[((img_size[0] / 2) - 55),((img_size[1] / 2) + 100)],
                [((img_size[0] / 6) - 10), img_size[1]],
                [((img_size[0] * 5 / 6) + 60), img_size[1]], 
                [((img_size[0] / 2) + 55), ((img_size[1] / 2 )+ 100)]])
    dst = np.float32([[(img_size[0] / 4), 0],
                [(img_size[0] / 4), img_size[1]],
                [(img_size[0] * 3 / 4), img_size[1]],
                [(img_size[0] * 3 / 4), 0]])


    # Given src and dst points, calculate the perspective transform matrix and inverse perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(preprocessImage, M, img_size, flags=cv2.INTER_LINEAR)
    #Convolution for sliding window search
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    if len(window_centroids) > 0:
    
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
        
        #points used to find the left and right lanes
        rightx = []
        leftx = []
    
        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            
    	    leftx.append(window_centroids[level][0])
    	    rightx.append(window_centroids[level][1])
         
           #Left and Right window mask  
    	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
    	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
 
       	    # Add graphic points from window mask here to total pixels found 
    	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
    
        if(t_type):    
            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
         if(t_type):      
            output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        
    #Fit the lane boundaries to the left, right, center positions
    yvals = range(0, warped.shape[0])
    res_yvals= np.arange(warped.shape[0]-(window_height/2),0,-window_height)
    
    left_fit = np.polyfit(res_yvals,leftx,2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx,np.int32)    
    
    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx,np.int32) 
    
    #Area of interest
    left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2,left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    inner_area = np.array(list(zip(np.concatenate((left_fitx - window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    
    matrix = np.zeros_like(img)    
    matrix_bkg = np.zeros_like(img)
    cv2.fillPoly(matrix,[inner_area],color=[0,255,0])
    cv2.fillPoly(matrix,[left_lane],color=[255,0,0])
    cv2.fillPoly(matrix,[right_lane],color=[0,0,255])
    cv2.fillPoly(matrix_bkg,[left_lane],color=[255,255,255])
    cv2.fillPoly(matrix_bkg,[right_lane],color=[255,255,255])
        
    warped_output = cv2.warpPerspective(matrix, Minv, img_size, flags=cv2.INTER_LINEAR)
    warped_output_bkg = cv2.warpPerspective(matrix_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)
    
    base_result = cv2.addWeighted(img, 1.0, warped_output_bkg, -1.0, 0.0)
    final_result = cv2.addWeighted(base_result, 1.0, warped_output, 1.0, 0.0)
    
    #Radius of curvature
    y = np.array(res_yvals,np.float32)
    x = np.array(leftx, np.float32)
   
    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    curverad = ((1 + (2 * fit_cr[0] * yvals[-1] *ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2 * fit_cr[0])
   
    #offset of the car on the road
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center - warped.shape[1]/2)*xm_per_pix
    
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'
    
    cv2.putText(final_result, 'Radius of Curvature = '+str(round(curverad,3))+'(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(final_result, 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
     
    return final_result

# Make a list of test images
images = glob.glob('./test_images/*.jpg')
t_type = 1
# Step through the list and preprocess image
for idx, fname in enumerate(images):
    #read the image
    img = cv2.imread(fname)
    result = process_image(img,t_type)
    write_name = './test_images/tracked'+str(idx)+'.jpg'
    cv2.imwrite(write_name, result)
#==============================================================================

output_video = 'project_output.mp4'    
input_video = 'project_video.mp4'    
clip1 = VideoFileClip(input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)

challenge_output = 'challenge_output.mp4'    
challenge_input = 'challenge_video.mp4'    
clip2 = VideoFileClip(challenge_input)
video_clip = clip2.fl_image(process_image)
video_clip.write_videofile(challenge_output, audio=False)

harder_challenge_output = 'harder_challenge_output.mp4'    
harder_challenge_input = 'harder_challenge_video.mp4'    
clip3 = VideoFileClip(harder_challenge_input)
video_clip = clip3.fl_image(process_image)
video_clip.write_videofile(harder_challenge_output, audio=False)
