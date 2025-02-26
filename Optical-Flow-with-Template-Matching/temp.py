# import cv2
# # # import cv2 as cv
# import numpy as np

# # Parameters for Shi-Tomasi corner detection
# feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
# # Parameters for Lucas-Kanade optical flow
# lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# # The video feed is read in as a VideoCapture object
# cap = cv.VideoCapture("shibuya.mp4")
# # Variable for color to draw optical flow track
# color = (0, 255, 0)
# # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
# ret, first_frame = cap.read()
# # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
# prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
# # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
# prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
# # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
# mask = np.zeros_like(first_frame)
#
# while(cap.isOpened()):
#     # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
#     ret, frame = cap.read()
#     # Converts each frame to grayscale - we previously only converted the first frame to grayscale
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Calculates sparse optical flow by Lucas-Kanade method
#     # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
#     prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
#     next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
#     # Selects good feature points for previous position
#     good_old = prev[status == 1].astype(int)
#     # Selects good feature points for next position
#     good_new = next[status == 1].astype(int)
#     # Draws the optical flow tracks
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         # Returns a contiguous flattened array as (x, y) coordinates for new point
#         a, b = new.ravel()
#         # Returns a contiguous flattened array as (x, y) coordinates for old point
#         c, d = old.ravel()
#         # Draws line between new and old position with green color and 2 thickness
#         mask = cv.line(mask, (a, b), (c, d), color, 2)
#         # Draws filled circle (thickness of -1) at new position with green color and radius of 3
#         frame = cv.circle(frame, (a, b), 3, color, -1)
#     # Overlays the optical flow tracks on the original frame
#     output = cv.add(frame, mask)
#     # Updates previous frame
#     prev_gray = gray.copy()
#     # Updates previous good feature points
#     prev = good_new.reshape(-1, 1, 2)
#     # Opens a new window and displays the output frame
#     cv.imshow("sparse optical flow", output)
#     # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
#     if cv.waitKey(10) & 0xFF == ord('q'):
#         break
# # The following frees up resources and closes all windows
# cap.release()
# cv.destroyAllWindows()

#https://pysource.com/2018/05/14/optical-flow-with-lucas-kanade-method-opencv-3-4-with-python-3-tutorial-31/

import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")

# Create old frame
_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Lucas kanade params
lk_params = dict(winSize = (15, 15),
maxLevel = 4,
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)

point_selected = False
point = ()
old_points = np.array([[]])

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()
        old_points = new_points

        x, y = new_points.ravel()
        xint = int(x)
        yint = int(y)
        cv2.circle(frame, (xint, yint), 5, (0, 255, 0), -1)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
