import numpy as np
import cv2 as cv
import math

# import matplotlib.pyplot as plt


# img1 = cv.imread('../images/main_image_swift_car.png',cv.IMREAD_GRAYSCALE) # queryImage
# img2 = cv.imread('../images/template_image_swift_car_rect_rotated.PNG',cv.IMREAD_GRAYSCALE) # trainImage

# colorimg1 = cv.imread('../images/res_3.png') # queryImage
# colorimg2 = cv.imread('../images/template_image_main_bigger.PNG') # trainImage

colorimg1 = cv.imread('../images/res_first_frame_gray_sedan_car.PNG') # queryImage
colorimg2 = cv.imread('../images/template_image_gray_sedan_rotated_car.PNG') # trainImage

# img1 = cv.imread('../images/res_3.png',cv.IMREAD_GRAYSCALE) # queryImage
# img2 = cv.imread('../images/template_image_main_bigger.PNG',cv.IMREAD_GRAYSCALE) # trainImage

img1 = cv.imread('../images/res_first_frame_gray_sedan_car.png',cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('../images/template_image_gray_sedan_rotated_car.PNG',cv.IMREAD_GRAYSCALE) # trainImage

imgW, imgH = img1.shape[::-1]
width,height = img2.shape[::-1]

print("Image width -> ", imgW, " and Image height -> ", imgH)
# print("Template width -> ", w, " and height -> ", h)

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# print(len(kp1))
# print(len(kp2))
# print(len(des1))
# print(len(des2))

# print(kp1[0].)
# print(des1[0])
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# print("matches -> ", matches[0].trainIdx)
pointsOfInterest = []
goodFeatures = []
objectMatchingPoints = []
cnt = 0
# threashold = 0.7
# matches = flann.match(des1,des2, None)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
# thArr = np.linspace(0,1,20, endpoint=True)
thArr = []
th = 0.7
for i in range (0, 101, 5):
   thArr.append(i/100)
   # i += 5
print(thArr)
# for th in thArr:
print("Current threshold -> {}".format(th))
for i,(m,n) in enumerate(matches):
     if m.distance < th*n.distance:
        print("m.distance -> ", m.distance)
        print("n.distance -> ", n.distance)
        cnt = cnt + 1
        matchesMask[i]=[1,0]
        kp1point = kp1[m.queryIdx]
        kp2point = kp2[m.trainIdx]
        # print("main desc -> ", m.queryIdx)
        # print("train desc -> ", m.trainIdx)
        goodFeatures.append(m)
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[n.trainIdx].pt
        # print(i, pt1, pt2)
        pointsOfInterest.append((int(pt1[0]), int(pt1[1])))
        # print("div -> ", (th*n.distance) / m.distance)
        # th += 0.05
        # if ((th*n.distance) / m.distance) > 10 or ((th*n.distance) / m.distance) < 20:
        # ## Draw pairs in red, to make sure the result is ok
        #     print((int(pt1[0]), int(pt1[1])))
        #     print((int(pt2[0]), int(pt2[1])))
        # print(int(int(pt1[0]) / 10))
        # if (int(int(pt1[0]) / 10) == 8):
            # cv.circle(colorimg1, (int(pt1[0]), int(pt1[1])), 5, (0, 255, 0), -1)
            # cv.circle(colorimg2, (int(pt2[0]), int(pt2[1])), 5, (0, 255, 0), -1)

print("total good features -> ", len(goodFeatures), " for threashold -> ", th)
# print(goodFeatures)

print("pointsOfInterest -> ", pointsOfInterest)
thWidth = width / 10
thHeight = height / 10
thWidth = thWidth * 10
# print(pointsOfInterest[1][1])
minVal = min(pointsOfInterest)
maxVal = max(pointsOfInterest)
print(minVal, " ", maxVal)

# for i in range(len(pointsOfInterest)):
#     for j in range(len(pointsOfInterest[i])):
#         if (pointsOfInterest[i][0] < (thWidth + minVal[0])): # or pointsOfInterest[i][1] < thHeight + minVal[1]
#             # print(pointsOfInterest[i][j])
#             objectMatchingPoints.append(pointsOfInterest[i])

for i in range(len(pointsOfInterest)):
    if i+1 < len(pointsOfInterest):
        res = math.hypot((pointsOfInterest[i][0] - pointsOfInterest[i + 1][0]), (pointsOfInterest[i][1] - pointsOfInterest[i + 1][1]))
        if res < 10:
            print("POI within res -> ", pointsOfInterest[i])
            objectMatchingPoints.append(pointsOfInterest[i])
        # print("res -> ",res)

print(objectMatchingPoints)
matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)

matches_mask = mask.ravel().tolist()

for i in range(len(objectMatchingPoints)):
    for j in range(len(objectMatchingPoints[i])):
        cv.circle(colorimg1, (objectMatchingPoints[i]), 5, (0, 255, 0), -1)

draw_params = dict(matchColor = (0,255,0),
singlePointColor = (255,0,0),
matchesMask = matchesMask,
flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

cv.imshow("final img", img3)
cv.imshow("main img", colorimg1)
cv.imshow("template img", colorimg2)
# cv.waitKey(5000)
cv.waitKey(0)
# plt.imshow(img3,),plt.show()