# import cv as cv
# import numpy as np
#
# cap = cv.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")
# ret, first_frame = cap.read()
# prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# # Parameters for Shi-Tomasi corner detection
# feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
# # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
# # mask = np.zeros_like(first_frame)
#
# width  = cap.get(cv.CAP_PROP_FRAME_WIDTH)   # float
# height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float
# print("w = %f" % width)
# print("h = %f" % height)
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Initialize the mask with all black pixels
#     mask = [[0]*int(width)]*int(height)
#     mask = np.asarray(mask)
#     mask = mask.astype(np.uint8)
#
#     # print( np.shape(mask) )
#     # Get the coordinates and dimensions of the detect_box
#     x = 74
#     y = 137
#     w = 64
#     h = 57
#     # Set the selected rectangle within the mask to white
#     mask[y:y+h, x:x+w] = 255
#
#     # prev = cv.goodFeaturesToTrack(prev_gray, mask = mask, **feature_params)
#     # output = cv.add(frame, mask)
#     mask = np.zeros_like(first_frame)
#     cv.imshow('Mask', mask)
#     # Updates previous frame
#     prev_gray = gray.copy()
#     # cv.imshow("sparse optical flow", output)

import cv2 as cv
import numpy as np

# Read an input image as a gray image
# first_frame = cv.imread('car.jpg')
cap =  cv.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")
ret, first_frame = cap.read()
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


# ret, first_frame = cap.read()
# create a mask
# x = 74
# y = 137
# w = 64
# h = 57
x = 100
y = 200
w = 200
h = 250
mask = np.zeros(first_frame.shape[:2], np.uint8)
mask[y:y+h, x:x+h] = 255
mask = mask.astype(np.uint8)
# compute the bitwise AND using the mask
# masked_first_frame = cv.bitwise_and(first_frame,first_frame,mask = mask)
prev = cv.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)
print("featues to track", prev)
print("Done")
# display the mask, and the output image
cv.imshow('Mask',mask)
# cv.waitKey(0)
# cv.imshow('Masked Image',masked_first_frame)
# height = mask.shape[0]
# width = mask.shape[1]
# channels = mask.shape[2]
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask_image = np.zeros_like(first_frame)
print("mask shape", mask.shape)
counter = 0
while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates sparse optical flow by Lucas-Kanade method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    if counter == 0:
        prev = cv.goodFeaturesToTrack(prev_gray, mask = mask, **feature_params)
    # prev = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    # Selects good feature points for previous position
    good_old = prev[status == 1].astype(int)
    # Selects good feature points for next position
    good_new = next[status == 1].astype(int)
    # print("Frame shape", frame.shape)
    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        # Draws line between new and old position with green color and 2 thickness
        # mask_image = cv.line(mask_image, (a, b), (c, d), [0,255,0], 1)
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        frame = cv.circle(frame, (a, b), 3, [0,0,255], 1)
    # Overlays the optical flow tracks on the original frame
    # output = cv.add(frame, mask_image)
    cv.imshow("sparse optical flow", frame)
    # Updates previous frame
    prev_gray = gray.copy()
    # Updates previous good feature points
    prev = good_new.reshape(-1, 1, 2)
    # Opens a new window and displays the output frame
    # cv.imshow("sparse optical flow", output)
    if cv.waitKey(100) & 0xFF == ord('q'):
        break

# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()