import numpy as np
import cv2

# query_img = cv2.imread('../images/res_template_matching_white_car.png')
# train_img = cv2.imread('../images/template_image_main_bigger.PNG')

query_img = cv2.imread('../images/main_image_swift_car.png')
train_img = cv2.imread('../images/template_image_swift_car_rect.PNG')

query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("main img", query_img_bw)
# cv2.imshow("template img", train_img_bw)
#
# cv2.waitKey(0)

orb = cv2.ORB_create(nfeatures=2000)

queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

# print(queryDescriptors[0])
# print(trainDescriptors[0])

imgqueryKeypoints = cv2.drawKeypoints(query_img, queryKeypoints, None)
imgtrainKeypoints = cv2.drawKeypoints(train_img, trainKeypoints, None)

cv2.imshow("main img", imgqueryKeypoints)
cv2.imshow("template img", imgtrainKeypoints)

cv2.waitKey(100)

matcher = cv2.BFMatcher()
# matches = matcher.match(queryDescriptors, trainDescriptors)
matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

good = []

for m,n in matches:
    if m.distance < 0.35*n.distance:
        good.append([m])
        print("m ->", m.distance)
print("no of good matches", len(good))
final_img = cv2.drawMatchesKnn(query_img, queryKeypoints,
                            train_img, trainKeypoints, good, None, flags=2)
# final_img = cv2.drawMatches(query_img, queryKeypoints,
#                             train_img, trainKeypoints, matches[:20], None)
#
# final_img = cv2.resize(final_img, (1000, 650))

# Show the final image
cv2.imshow("Matches", final_img)
cv2.waitKey(0)

# # from skimage.metrics import structural_similarity
# import cv2
#
#
# # Works well with images of different dimensions
# def orb_sim(img1, img2):
#     # SIFT is no longer available in cv2 so using ORB
#     orb = cv2.ORB_create()
#
#     # detect keypoints and descriptors
#     kp_a, desc_a = orb.detectAndCompute(img1, None)
#     kp_b, desc_b = orb.detectAndCompute(img2, None)
#
#     # define the bruteforce matcher object
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
#     # perform matches.
#     matches = bf.match(desc_a, desc_b)
#     # Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
#     similar_regions = [i for i in matches if i.distance < 50]
#     if len(matches) == 0:
#         return 0
#     return len(similar_regions) / len(matches)
#
#
# # Needs images to be same dimensions
# # def structural_sim(img1, img2):
# #     sim, diff = structural_similarity(img1, img2, full=True)
# #     return sim
#
#
# # img00 = cv2.imread('images/monkey_distorted.jpg', 0)
# # img01 = cv2.imread('images/monkey_rotated.jpg', 0)
#
# query_img = cv2.imread('../images/main_image_swift_car.png', 0)
# train_img = cv2.imread('../images/swift_template.PNG', 0)
#
# # img1 = cv2.imread('images/BSE.jpg', 0)  # 714 x 901 pixels
# # img2 = cv2.imread('images/BSE_noisy.jpg', 0)  # 714 x 901 pixels
# # img3 = cv2.imread('images/BSE_smoothed.jpg', 0)  # 203 x 256 pixels
# # img4 = cv2.imread('images/different_img.jpg', 0)  # 203 x 256 pixels
#
# orb_similarity = orb_sim(train_img, train_img)  # 1.0 means identical. Lower = not similar
#
# print("Similarity using ORB is: ", orb_similarity)
# # Resize for SSIM
# # from skimage.transform import resize
# #
# # img5 = resize(img3, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)
# #
# # ssim = structural_sim(img1, img5)  # 1.0 means identical. Lower = not similar
# # print("Similarity using SSIM is: ", ssim)