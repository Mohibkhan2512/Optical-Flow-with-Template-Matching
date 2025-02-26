import cv2 as cv
import numpy as np


def get_corrected_img(img1, img2):
    img1 = cv.imread(img1)
    img2 = cv.imread(img2)

    # Initiate SIFT detector
    orb = cv.ORB_create(nfeatures=1000)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)

    des1 = np.float32(des1)
    des2 = np.float32(des2)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for th in range(0, 1):
        print("Current threshold -> {}".format(th))
        for i, (m, n) in enumerate(matches):
            if m.distance < th * n.distance:
                matchesMask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)

        img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv.imshow("", img3)
        th += 0.05
        print("Current threshold -> {}".format(th))
        cv.waitKey(1000)
    # MIN_MATCHES = 5
    #
    # orb = cv.ORB_create(nfeatures=500)
    # kp1, des1 = orb.detectAndCompute(img1, None)
    # kp2, des2 = orb.detectAndCompute(img2, None)
    #
    # print("des1 -> ", des1)
    # print("des2 -> ", des2)
    # # index_params = dict(algorithm=6,
    # #                     table_number=6,
    # #                     key_size=12,
    # #                     multi_probe_level=2)
    # # search_params = {}
    # # flann = cv.FlannBasedMatcher(index_params, search_params)
    # # matches = flann.knnMatch(des1, des2, k=2)
    #
    # # # As per Lowe's ratio test to filter good matches
    # # good_matches = []
    # # for (m, n) in matches:
    # #     if m.distance < 0.55 * n.distance:
    # #         good_matches.append(m)
    #
    # # FLANN parameters
    # index_params = dict(algorithm=6,
    #                     table_number=6,
    #                     key_size=12,
    #                     multi_probe_level=2)
    # search_params = {}
    # flann = cv.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    # # matches = flann.knnMatch(des1, des2, k=2)
    # print("matches -> ", matches)
    # pointsToTrack = []
    # goodFeatures = []
    # cnt = 0
    # threashold = 0.55
    # # matches = flann.match(des1,des2, None)
    # # Need to draw only good matches, so create a mask
    # matchesMask = [[0, 0] for i in range(len(matches))]
    # # ratio test as per Lowe's paper
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < threashold * n.distance:
    #         print("m.distance -> ", m.distance)
    #         print("n.distance -> ", threashold * n.distance)
    #         cnt = cnt + 1
    #         matchesMask[i] = [1, 0]
    #         kp1point = kp1[m.queryIdx]
    #         kp2point = kp2[m.trainIdx]
    #         # print("main desc -> ", m.queryIdx)
    #         # print("train desc -> ", m.trainIdx)
    #         goodFeatures.append(m)
    #         pt1 = kp1[m.queryIdx].pt
    #         pt2 = kp2[n.trainIdx].pt
    #         print(i, pt1, pt2)
    #         pointsToTrack.append(pt1)
    #         # if i % 5 == 0:
    #         ## Draw pairs in red, to make sure the result is ok
    #         cv.circle(colorimg1, (int(pt1[0]), int(pt1[1])), 5, (0, 0, 255), -1)
    #         cv.circle(colorimg2, (int(pt2[0]), int(pt2[1])), 5, (0, 0, 255), -1)
    #         if len(good_matches) > MIN_MATCHES:
    #             src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #             dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #             m, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
    #             corrected_img = cv.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))

if __name__ == "__main__":
    # get_corrected_img("../images/res_first_frame.PNG", "../images/template_image_4.PNG")
    get_corrected_img("../images/res_3.PNG", "../images/template_image_main_bigger.PNG")