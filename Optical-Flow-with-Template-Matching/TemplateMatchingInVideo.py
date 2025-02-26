import cv2 as cv
import numpy as np



def detectObjectInFrame(image, template_img):
    # Template matching for finding ref(find OF for this image) file in frame
    # img_rgb = cv.imread(image)
    img_rgb = image.copy()
    cv.imshow("frame in func", img_rgb)
    cv.waitKey()
    # assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(template_img, cv.IMREAD_GRAYSCALE)
    # assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    # if res is None:
    #     print("Template not found in image...")
    # else:
    #     print("Template found in image...")
    # threshold = 0.8
    threshold = 0.8
    loc = np.where(res >= threshold)
    flag = np.any(loc)
    if flag is False:
        print("Template not found in image...")
        return
    else:
        print("Template found in image...")
    print("loc ->", loc)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
    cv.imwrite('../images/res_template_matching_white_car.png', img_rgb)
    # cv.imshow('../images/res_5.png')
    cv.imshow("matched image", img_rgb)
    return pt[0], pt[1], pt[0] + w, pt[1] + h, flag, w, h

def findTemplateFromvideo(image, template_img):
    template_img = cv.imread(template_img, cv.IMREAD_UNCHANGED)

    img_rgb = image.copy()

    # global grayFrame
    grayFrame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    graySearchFor = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
    w, h = graySearchFor.shape[::-1]

    cv.imshow('current gray', grayFrame)
    cv.imshow('template gray', grayFrame)
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
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
    cv.imshow("matched image", img_rgb)
    cv.waitKey(0)
    return pt[0], pt[1], pt[0] + w, pt[1] + h, flag, w, h

if __name__=="__main__":
    cap = cv.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")
    # cap = cv.VideoCapture(r"..\Video_Data\swiftCarRoad.mp4")
    # cap = cv.VideoCapture(r"..\Video_Data\roads_1952 (720p).mp4")
    template_img = "../images/template_image_gray_suv_car.PNG"
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
                print("Frame Number is: ", frame_counter)
                flag = findTemplateFromvideo(frame, template_img)
            elif frame_counter % 10 == 0 and flag != 0 and StopCode == False:
                print("Print found template in image, Frame Number is: ", frame_counter)
                pt1, pt2, pt3, pt4, res, w, h = findTemplateFromvideo(frame, template_img)
                StopCode = True
                # Closes all the frames
                cv.destroyAllWindows()
            # stop_code = False
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
            frame_counter += 1
        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()










# => Older working code

# def detectObjectInFrame(image, template_img):
#     # Template matching for finding ref(find OF for this image) file in frame
#     # img_rgb = cv.imread(image)
#     img_rgb = image.copy()
#     # assert img_rgb is not None, "file could not be read, check with os.path.exists()"
#     img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
#     template = cv.imread(template_img, cv.IMREAD_GRAYSCALE)
#     # assert template is not None, "file could not be read, check with os.path.exists()"
#     w, h = template.shape[::-1]
#     res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
#     if res is None:
#         print("Template not found in image...")
#     else:
#         print("Template found in image...")
#     # threshold = 0.8
#     threshold = 0.8
#     loc = np.where(res >= threshold)
#     print("loc ->", loc)
#     for pt in zip(*loc[::-1]):
#         cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
#     cv.imwrite('../images/res_template_matching_white_car.png', img_rgb)
#     # cv.imshow('../images/res_5.png')
#     return pt[0], pt[1], pt[0] + w, pt[1] + h, res, w, h
#
# if __name__=="__main__":
#     cap = cv.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")
#     # cap = cv.VideoCapture(r"..\Video_Data\roads_1952 (720p).mp4")
#
#     if (cap.isOpened() == False):
#         print("Error opening the video file")
#     res = None
#     object_detected = False
#     # stop_code=False
#     frame_counter = 0
#     # starting_time = time.time()
#     _, frame = cap.read()
#     # old_image = frame.copy()
#     # old_image = cv.imwrite('../images/res_first_frame_gray_sports_car.png', frame)
#     while (cap.isOpened()):
#         frame_counter += 1
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if ret == True:
#             img = frame.copy()
#             # Display the resulting frame
#             cv.imshow('Frame', frame)
#             # cv.imshow('old frame ', old_image)
#             # using template matching to search if template is in image
#             # hull_points = detectObjectInFrame(frame)
#             if res is None:
#                 pt1, pt2, pt3, pt4, res, w, h = detectObjectInFrame(frame, "../images/template_image_gray_sports_car.PNG")
#                 x1 = pt1
#                 y1 = pt2
#                 x2 = x1 + w
#                 y2 = y1 + h
#
#                 # print("pt1 -> ", pt1, " and pt2 -> ", pt2, " and pt3 -> ", pt3,
#                 #       " and pt4 -> ", pt4)
#                 print("(x1, y1) -> ", x1, y1, " and (x1, y2) -> ", x1, y2, " and (x2, y1) -> ", x2, y1,
#                       " and (x2, y2) -> ",
#                       x2, y2)
#             # print(old_points.size)
#             stop_code = False
#             # cv.imshow('current img', img)
#             # Press Q on keyboard to exit
#             if cv.waitKey(25) & 0xFF == ord('q'):
#                 break
#
#         # Break the loop
#         else:
#             break
#
#     # When everything done, release
#     # the video capture object
#     cap.release()
#
#     # Closes all the frames
#     cv.destroyAllWindows()

