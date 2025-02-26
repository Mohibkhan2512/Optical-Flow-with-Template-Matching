# import the necessary packages
import cv2 as cv 
import numpy as np
from pyzbar.pyzbar import decode
import pyzbar
import time
import AiPhile

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
    cv.imwrite('../images/res_template_matching.png', img_rgb)
    # cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    key = cv.waitKey(1000)
    # cv.imshow('Image with template', img_rgb)
    return pt[0], pt[1], pt[0] + w, pt[1] + h, res, template_found, w, h, img_rgb

if __name__=="__main__":
    cap = cv.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")
    # cap = cv.VideoCapture(r"..\Video_Data\roads_1952 (720p).mp4")
    _, frame = cap.read()
    old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    old_image = cv.imwrite('../images/res_first_frame.png', frame)
    lk_params = dict(winSize=(10, 10),
                     maxLevel=4,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))
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
            # hull_points = detectObjectInFrame(frame)
            if res is None:
                pt1, pt2, pt3, pt4, res, template_found, w, h, frame_template = detectObjectInFrame("../images/res_first_frame.png", "../images/template_image_main.PNG")
                x1 = pt1
                y1 = pt2
                x2 = x1 + w
                y2 = y1 + h

                print("(x1, y1) -> ", x1, y1, " and (x1, y2) -> ", x1, y2, " and (x2, y1) -> ", x2, y1,
                      " and (x2, y2) -> ",
                      x2, y2)
                if template_found:
                    # cv.imshow('Image with template', img_rgb)
                    object_detected = True
                    stop_code = True
                    list_of_points = [[x1,y1], [x1,y2], [x2,y2], [x2,y1]]
                    # hull_points = [int(i) for i in list_of_points]
                    print("Modified list is : " ,list_of_points)
                    hull_points = [[x1,y1], [x1,y2], [x2,y2], [x2,y1]]
                    old_points = np.array([[x1,y1], [x1,y2], [x2,y2], [x2,y1]], dtype=np.float32)
                    print("old_points -> ", old_points)
                    frame = AiPhile.fillPolyTrans(frame_template, hull_points, AiPhile.MAGENTA, 0.4)
                    AiPhile.textBGoutline(frame, f'Detection of template in frame', (30, 80), scaling=0.5,
                                          text_color=(AiPhile.MAGENTA))
                    cv.imshow('img with template', frame)

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
                    # mask_image = cv.line(mask_image, (a, b), (c, d), [0,255,0], 1)
                    # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                    frame = cv.circle(frame, (a, b), 3, [0, 0, 255], 1)
                n = (len(new_points))
                cv.imshow("sparse optical flow", frame)
                # frame = AiPhile.fillPolyTrans(frame, new_points, AiPhile.GREEN, 0.4)
                # AiPhile.textBGoutline(frame, f'Detection: Optical Flow', (30, 80), scaling=0.5,
                #                       text_color=AiPhile.GREEN)
                # cv.circle(frame, (new_points[0]), 3, AiPhile.GREEN, 2)

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
