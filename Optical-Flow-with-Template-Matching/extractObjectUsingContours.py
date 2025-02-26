import cv2 as cv
import numpy as np

if __name__=="__main__":
    cap = cv.VideoCapture(r"..\Video_Data\istockphoto-507652134-640_adpp_is.mp4")
    # cap = cv.VideoCapture(r"..\Video_Data\roads_1952 (720p).mp4")

    if (cap.isOpened() == False):
        print("Error opening the video file")
    res = None
    object_detected = False
    # stop_code=False
    frame_counter = 0
    flag = 0
    StopCode = False
    # starting_time = time.time()
    _, image = cap.read()
    # old_image = frame.copy()
    # old_image = cv.imwrite('../images/res_first_frame_gray_sports_car.png', frame)
    while (cap.isOpened()):

        # Grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Find Canny edges
        edged = cv.Canny(gray, 30, 200)
        cv.waitKey(0)

        # Finding Contours
        # Use a copy of the image e.g. edged.copy()
        # since findContours alters the image
        contours, hierarchy = cv.findContours(edged,
            cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        cv.imshow('Canny Edges After Contouring', edged)
        cv.waitKey(0)

        print("Number of Contours found = " + str(len(contours)))

        # Draw all contours
        # -1 signifies drawing all contours
        cv.drawContours(image, contours, -1, (0, 255, 0), 3)

        cv.imshow('Contours', image)
        cv.waitKey(0)
        cv.destroyAllWindows()


# import cv2 as cv
# import numpy as np
#
# def findTemplateFromvideo(image, template_img):
#     # template_img = cv.imread(template_img, cv.IMREAD_UNCHANGED)
#
#     img_rgb = image.copy()
#     template_img = template_img.copy()
#
#     cv.imshow("template image", template_img)
#     cv.waitKey(0)
#     # global grayFrame
#     grayFrame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     graySearchFor = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
#     w, h = graySearchFor.shape[::-1]
#
#     cv.imshow('current gray', grayFrame)
#     cv.imshow('template gray', grayFrame)
#     cv.waitKey(100)
#     res = cv.matchTemplate(grayFrame, graySearchFor, cv.TM_CCOEFF_NORMED)
#
#     threshold = 0.8
#     loc = np.where(res >= threshold)
#     flag = np.size(loc)
#     print("Size of output points: ", flag)
#     if flag == 0:
#         print("Template not found in image...")
#         return flag
#     else:
#         print("Template found in image...")
#     print("loc ->", loc)
#     for pt in zip(*loc[::-1]):
#         cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
#     cv.imshow("matched image", img_rgb)
#     cv.waitKey(0)
#     return pt[0], pt[1], pt[0] + w, pt[1] + h, flag, w, h
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
#     flag = 0
#     StopCode = False
#     # starting_time = time.time()
#     _, frame = cap.read()
#     # old_image = frame.copy()
#     # old_image = cv.imwrite('../images/res_first_frame_gray_sports_car.png', frame)
#     while (cap.isOpened()):
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         # Let's load a simple image with 3 black squares
#         image = cv.imread("../images/template_image_gray_sports_car.PNG")
#         # cv.waitKey(0)
#
#         # Grayscale
#         gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#
#         # Simple binary threshold
#         _, gray = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
#
#         # Find Canny edges
#         edged = cv.Canny(gray, 240, 250)
#         # cv.waitKey(100)
#
#         contours, hierarchy = cv.findContours(edged,
#                                                cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#
#         cv.imshow('Canny Edges After Contouring', edged)
#         # cv.waitKey(100)
#
#         print("Number of Contours found = " + str(len(contours)))
#
#         # Draw all contours
#         # -1 signifies drawing all contours
#         cv.drawContours(image, contours, -1, (0, 255, 255), 1)
#         image = cv.bitwise_not(image, mask = None)
#         cv.imshow('Contours of template image', image)
#         # cv.waitKey(0)
#         if ret == True:
#             if frame_counter % 10 == 0 and flag == 0 and StopCode == False:
#                 print("Frame Number is: ", frame_counter)
#                 flag = findTemplateFromvideo(frame, image)
#             elif flag != 0 and StopCode == False:
#                 print("Print found template in image, Frame Number is: ", frame_counter)
#                 pt1, pt2, pt3, pt4, res, w, h = findTemplateFromvideo(frame, image)
#                 StopCode = True
#                 # Closes all the frames
#                 cv.destroyAllWindows()
#             # stop_code = False
#             if cv.waitKey(25) & 0xFF == ord('q'):
#                 break
#             frame_counter += 1
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
#
#
# # cv.waitKey(0)
# # cv.destroyAllWindows()
