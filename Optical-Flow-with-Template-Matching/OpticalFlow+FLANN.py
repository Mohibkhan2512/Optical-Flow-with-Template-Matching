# import the necessary packages
import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode
import pyzbar
import time
import AiPhile

def keyPointMatching(image, template_img_path):
    # img1 = cv.imread('../images/main_image_swift_car.png',cv.IMREAD_GRAYSCALE) # queryImage
    # img2 = cv.imread('../images/template_image_swift_car_rect_rotated.PNG',cv.IMREAD_GRAYSCALE) # trainImage

    colorimg1 = image.copy()  # queryImage
    colorimg2 = cv.imread(template_img_path)  # trainImage

    # img1 = cv.imread('../images/main_image_2.png', cv.IMREAD_GRAYSCALE)  # queryImage
    # img2 = cv.imread('../images/template_image_main_bigger.PNG', cv.IMREAD_GRAYSCALE)  # trainImage

    img1 = image.copy()  # queryImage
    img2 = cv.imread(template_img_path)  # trainImage

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
    # print("matches -> ", matches[0].trainIdx)
    pointsToTrack = []
    goodFeatures = []
    cnt = 0
    threashold = 0.45
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
            cv.circle(colorimg1, (int(pt1[0]), int(pt1[1])), 5, (0, 0, 255), 1)
            cv.circle(colorimg2, (int(pt2[0]), int(pt2[1])), 5, (0, 0, 255), 1)
            if len(goodFeatures) >= 7:
                print("Found max match!!")
                flag = 1
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
    cv.waitKey(500)
    cv.destroyAllWindows()
    # plt.imshow(img3,),plt.show()
    return pointsToTrack, flag

if __name__=="__main__":
    cap = cv.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")
    # cap = cv.VideoCapture(r"..\Video_Data\roads_1952 (720p).mp4")
    _, frame = cap.read()
    old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # template_img = "../images/template_image_black_sedan_rotated_car.PNG"
    # template_img = "../images/template_image_4.PNG"
    template_img = "../images/template_image_gray_sedan_car.PNG"
    lk_params = dict(winSize=(10, 10),
                     maxLevel=4,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))
    mask_image = np.zeros_like(frame)

    # # cap = cv.VideoCapture(1)
    # for better results use this
    # lk_params = dict(winSize=(15, 15),
    #                  maxLevel=10,
    #                  criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))

    if (cap.isOpened() == False):
        print("Error opening the video file")

    # If we're able to caprutre frames
    cap = cv.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")
    res = None
    point_selected = False
    points = [()]
    old_points = np.array([[]])
    object_detected = False
    # stop_code=False
    frame_counter = 0
    # _, frame = cap.read()
    starting_time = time.time()
    flag = 0
    # old_gray = frame.copy()
    stop_code = False

    while (cap.isOpened()):
        frame_counter += 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            img = frame.copy()
            # Display the resulting frame
            cv.imshow('Frame', frame)
            cv.imshow('old frame ', old_gray)
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # using key-point matching to search if template is in image
            if flag == 0 and stop_code == False: # if frame_counter % 10 == 0 and flag == 0 and stop_code == False
                old_points, flag = keyPointMatching(frame, template_img)
                print("flag value -> ", flag)
                old_points = np.array(old_points, dtype=np.float32)
                # Closes all the frames
                cv.destroyAllWindows()
            if flag == 1 and stop_code == False:
                print("old_points -> ", old_points)
                stop_code = True
                object_detected = True
            if object_detected and stop_code == True:
                print("old_points in OF -> ", old_points)
                print('Tracking object using Optical Flow...')
                new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None,
                                                                    **lk_params)
                old_points = new_points
                new_points = new_points.astype(int)
                # Selects good feature points for previous position
                good_old = old_points.astype(int)
                # Selects good feature points for next position
                good_new = new_points.astype(int)
                # print("Frame shape", frame.shape)
                # Draws the optical flow tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    # Returns a contiguous flattened array as (x, y) coordinates for new point
                    a, b = new.ravel()
                    # Returns a contiguous flattened array as (x, y) coordinates for old point
                    c, d = old.ravel()
                    # Draws line between new and old position with green color and 2 thickness
                    mask_image = cv.line(mask_image, (a, b), (c, d), [0,255,0], 1)
                    # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                    frame = cv.circle(frame, (a, b), 3, [0, 0, 255], 1)
                n = (len(new_points))

                fps = frame_counter / (time.time() - starting_time)
                # AiPhile.textBGoutline(frame, f'FPS: {round(fps, 1)}', (30, 40), scaling=0.6)
                cv.imshow("sparse optical flow object tracking", frame)
                cv.imshow("sparse optical flow trajectory", mask_image)

            old_gray = gray_frame.copy()
            # cv.imshow('current img', img)
            # Press Q on keyboard to exit
            if cv.waitKey(55) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()
