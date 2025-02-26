import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt


def keyPointMatching(image, template_img_path):
    # img1 = cv.imread('../images/main_image_swift_car.png',cv.IMREAD_GRAYSCALE) # queryImage
    # img2 = cv.imread('../images/template_image_swift_car_rect_rotated.PNG',cv.IMREAD_GRAYSCALE) # trainImage

    colorimg1 = image.copy()  # queryImage
    colorimg2 = template_img_path.copy()  # trainImage

    # img1 = cv.imread('../images/main_image_2.png', cv.IMREAD_GRAYSCALE)  # queryImage
    # img2 = cv.imread('../images/template_image_main_bigger.PNG', cv.IMREAD_GRAYSCALE)  # trainImage

    img1 = image.copy()  # queryImage
    img2 = template_img_path.copy()  # trainImage
    graySearchFor = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    w, h = graySearchFor.shape[::-1]

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # print(len(kp1))
    # print(len(kp2))
    # print(len(des1))
    # print(len(des2))

    # print(kp1[0].)
    # print(des1[0])
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    print("total matches -> ",len(matches))
    # print("matches -> ", matches[0].trainIdx)
    pointsToTrack = []
    goodFeatures = []
    cnt = 0
    threashold = 0.45
    thresholdGFs = 9
    # matches = flann.match(des1,des2, None)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < threashold * n.distance:
            print("m.distance -> ", m.distance)
            print("n.distance -> ", threashold * n.distance)
            cnt = cnt + 1
            matchesMask[i] = [1, 0]
            kp1point = kp1[m.queryIdx]
            kp2point = kp2[m.trainIdx]
            # print("main desc -> ", m.queryIdx)
            # print("train desc -> ", m.trainIdx)
            goodFeatures.append(m)
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[n.trainIdx].pt
            print(i, pt1, pt2)
            pointsToTrack.append(pt1)
            # if i % 5 == 0:
            ## Draw pairs in red, to make sure the result is ok
            cv.circle(colorimg1, (int(pt1[0]), int(pt1[1])), 5, (0, 0, 255), -1)
            cv.circle(colorimg2, (int(pt2[0]), int(pt2[1])), 5, (0, 0, 255), -1)
            if len(goodFeatures) >= thresholdGFs:
                flag = 1
                # # cv.waitKey(10000)
                # # cv.rectangle(int(pt1[0]), int(pt1[1]), w, h, (0,255,0), 1)
                # cv.circle(colorimg1, (int(pt1[0]), int(pt1[1])), 5, (0, 255, 0), -1)
                # # cv.waitKey(10000)
                # src_pts = np.float32([kp1[m.queryIdx].pt for m in goodFeatures]).reshape(-1, 1, 2)
                # dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodFeatures]).reshape(-1, 1, 2)
                #
                # M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                # matchesMask = mask.ravel().tolist()
                #
                # h, w = graySearchFor.shape[::-1]
                # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # dst = cv.perspectiveTransform(pts, M)
                #
                # img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
                # cv.imshow("img2 final", img2)
                # cv.waitKey(1000)
                cv.circle(colorimg1, (int(pt1[0]), int(pt1[1])), 5, (0, 0, 255), -1)
                cv.circle(colorimg2, (int(pt2[0]), int(pt2[1])), 5, (0, 0, 255), -1)
                print("Found max match!! total good features -> ", len(goodFeatures))
                return pointsToTrack, flag
        else:
            flag = 0
    print("total good features -> ", len(goodFeatures))
    # pointsOfInterest.append(m)
    # print("pointsOfInterest -> ", pointsOfInterest)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    cv.imshow("final img", img3)
    cv.imshow("main img with features", colorimg1)
    cv.imshow("template img with features", colorimg2)
    cv.waitKey(1000)
    cv.destroyAllWindows()
    # plt.imshow(img3,),plt.show()
    return pointsToTrack, flag

if __name__=="__main__":
    cap = cv.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")
    # cap = cv.VideoCapture(r"..\Video_Data\swiftCarRoad.mp4")
    # cap = cv.VideoCapture(r"..\Video_Data\roads_1952 (720p).mp4")
    # template_img = "../images/template_image_black_sedan_rotated_car.PNG"
    template_img_path = "../images/template_image_4.PNG"
    template_img = cv.imread(template_img_path)
    graySearchFor = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
    w, h = graySearchFor.shape[::-1]
    pointsToTrack = []
    # template_img = "../images/template_image_swift_car_rect.PNG"
    if (cap.isOpened() == False):
        print("Error opening the video file")
    res = None
    object_detected = False
    # stop_code=False
    frame_counter = 0
    flag = 0
    StopCode = False
    # starting_time = time.time()
    _, frame = cap.read()
    # old_image = frame.copy()
    # old_image = cv.imwrite('../images/res_first_frame_gray_sports_car.png', frame)
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if frame_counter % 10 == 0 and flag == 0:
                pointsToTrack, flag = keyPointMatching(frame, template_img)
                print("points To Tack -> ", pointsToTrack)
                # print(pointsToTrack[4][0], " ", pointsToTrack[4][1])
                # cv.line(frame, int(pointsToTrack[0][0]), int(pointsToTrack[0][1]), (255, 0, 0), -1)
                if flag == 1:
                    # cv.rectangle(frame, (int(pointsToTrack[3][0]), int(pointsToTrack[3][1])),
                    #              (int(pointsToTrack[3][0]) + h, int(pointsToTrack[3][1]) + w), (255, 0, 0), -1)
                    cv.rectangle(frame, (118, 251),
                                 (118 + h, 251 + w), (255, 0, 0), -1)
                    cv.imshow("final frame", frame)
                    cv.waitKey(0)
                # Closes all the frames
                cv.destroyAllWindows()
            # stop_code = False
            if cv.waitKey(0) & 0xFF == ord('q'):
                break
            frame_counter += 1
        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    # cv.destroyAllWindows()

