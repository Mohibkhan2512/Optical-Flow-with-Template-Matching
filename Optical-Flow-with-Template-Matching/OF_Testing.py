# import the necessary packages
import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode
import pyzbar
import time
import AiPhile

# Parameters for Shi-Tomasi corner detection
# feature_params = dict(maxCorners = 500, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
feature_params = dict(maxCorners = 1000, qualityLevel = 0.1, minDistance = 1, blockSize = 5)

def exctractFeaturesFromTemplate(gray_frame, frame, x, y, w, h):
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[y:y + h, x:x + h] = 255
    mask = mask.astype(np.uint8)
    # compute the bitwise AND using the mask
    # masked_first_frame = cv.bitwise_and(first_frame,first_frame,mask = mask)
    prev = cv.goodFeaturesToTrack(gray_frame, mask=mask, **feature_params)
    print("featues to track", prev)
    print("Done")
    # display the mask, and the output image
    cv.imshow('Mask', mask)

    return prev

def detectObjectInFrame(image, template_img):
    # Template matching for finding ref(find OF for this image) file in frame
    img_rgb = cv.imread(image)
    # assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(template_img, cv.IMREAD_GRAYSCALE)
    # assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    template_found = None
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    if res is None:
        print("Template not found in image...")
    else:
        template_found = True
        print("Template found in image...")
    # threshold = 0.8
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    # cv.imwrite('../images/res_template_matching_white_car.png', img_rgb)
    # cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    key = cv.waitKey(1000)
    # cv.imshow('Image with template', img_rgb)
    return pt[0], pt[1], pt[0] + w, pt[1] + h, res, template_found, w, h, img_rgb


def findTemplateFromvideo(image, template_img):
    template_img = cv.imread(template_img, cv.IMREAD_UNCHANGED)

    img_rgb = image.copy()

    # global grayFrame
    grayFrame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    graySearchFor = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
    w, h = graySearchFor.shape[::-1]

    cv.imshow('current gray', grayFrame)
    cv.waitKey(100)
    res = cv.matchTemplate(grayFrame, graySearchFor, cv.TM_CCOEFF_NORMED)

    threshold = 0.8
    loc = np.where(res >= threshold)
    flag = np.size(loc)
    print("Size of output points: ", flag)
    if flag == 0:
        print("Template not found in image...")
        return flag
    else:
        print("Template found in image...")
    print("loc ->", loc)
    for pt in zip(*loc[::-1]):
        # cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
        x1 = pt[0]
        y1 = pt[1]
        x2 = x1 + w
        y2 = y1 + h
        list_of_points = [[x1,y1], [x1,y2], [x2,y2], [x2,y1]]
        list_to_np_array = np.array([list_of_points], dtype=np.int32)
        print("list_of_points: ", list_to_np_array)
        overlay = img_rgb.copy()  # coping the image
        cv.fillPoly(overlay, pts=[list_to_np_array], color=(0, 255, 255))
        new_img = cv.addWeighted(overlay, 0.6, img_rgb, 0.4, 0)
        # print(points_list)
        img_rgb = new_img
        cv.polylines(img_rgb, [list_to_np_array], True, (0, 255, 255), 2, cv.LINE_AA)
        # cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
    cv.imshow("matched image", img_rgb)
    # cv.waitKey(500)
    return pt[0], pt[1], pt[0] + w, pt[1] + h, flag, w, h

if __name__=="__main__":
    cap = cv.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")
    # cap = cv.VideoCapture(r"..\Video_Data\roads_1952 (720p).mp4")
    _, frame = cap.read()
    old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    old_image = cv.imwrite('../images/res_first_frame.png', frame)
    lk_params = dict(winSize=(10, 10),
                     maxLevel=4,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))
    mask_image = np.zeros_like(frame)

    # # cap = cv.VideoCapture(1)
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
            # using template matching to search if template is in image
            if frame_counter % 10 == 0 and flag == 0 and stop_code == False:
                print("Frame Number is: ", frame_counter)
                # flag = findTemplateFromvideo(frame, "../images/template_image_gray_overlap_car.PNG")
                flag = findTemplateFromvideo(frame, "../images/template_image_gray_sports_car.PNG")
            elif flag != 0 and stop_code == False:
                print("Print found template in image, Frame Number is: ", frame_counter)
                # pt1, pt2, pt3, pt4, res, w, h = findTemplateFromvideo(frame,
                #                                                       "../images/template_image_gray_overlap_car.PNG")

                pt1, pt2, pt3, pt4, res, w, h = findTemplateFromvideo(frame,
                                                                      "../images/template_image_gray_sports_car.PNG")
                x1 = pt1
                y1 = pt2
                x2 = x1 + w
                y2 = y1 + h
                old_points = exctractFeaturesFromTemplate(gray_frame, frame, x1, y1, w, h)
                print("old_points -> ", old_points)
                stop_code = True
                object_detected = True
                # Closes all the frames
                cv.destroyAllWindows()
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
                if np.size(good_new) > 10:
                    x, y, w, h = cv.boundingRect(good_new)
                    frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
                # print("good new -> ", good_new)
                # cv.rectangle(frame, good_new)
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
                AiPhile.textBGoutline(frame, f'FPS: {round(fps, 1)}', (30, 40), scaling=0.6)

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
