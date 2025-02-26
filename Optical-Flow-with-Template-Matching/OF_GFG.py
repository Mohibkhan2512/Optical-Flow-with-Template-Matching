import numpy as np
import cv2

cap = cv2.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")
# # params for corner detection
# feature_params = dict(maxCorners=300,
#                       qualityLevel=0.3,
#                       minDistance=7,
#                       blockSize=7)
#
# # Parameters for lucas kanade optical flow
# lk_params = dict(winSize=(15, 15),
#                  maxLevel=4,
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
#                            10, 0.03))

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame,
                        cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None,
                             **feature_params)
# Create some random colors
color = (0, 0, 255)
# x = 100
# y = 200
# w = 200
# h = 250
x = 74
y = 137
w = 64
h = 57
mask = np.zeros(old_frame.shape[:2], np.uint8)
mask[y:y+h, x:x+h] = 255
mask = mask.astype(np.uint8)
# compute the bitwise AND using the mask
# masked_first_frame = cv.bitwise_and(first_frame,first_frame,mask = mask)
prev = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
print("featues to track", prev)
print("Done")
# display the mask, and the output image
cv2.imshow('Mask',mask)

# Create a mask image for drawing purposes
mask_image = np.zeros_like(old_frame)

while (1):

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,
                              cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                           frame_gray,
                                           prev, None,
                                           **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = prev[st == 1]
    # print("good new: ", good_new)
    # print("good old: ", good_old)
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new,
                                       good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        print("a and b : ", a, " ", b)
        print("c and d : ", c, " ", d)
        mask_image = cv2.line(mask_image, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 1)

        frame = cv2.circle(frame, (int(a), int(b)), 2, (0, 255, 0), -1)

    img = cv2.add(frame, mask_image)

    cv2.imshow('frame', img)

    k = cv2.waitKey(25)
    if k == 27:
        break

    # Updating Previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
