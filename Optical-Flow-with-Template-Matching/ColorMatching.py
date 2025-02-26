import cv2 as cv
import numpy as np

def detectSameColorObjectFrame(image):
    # Template matching for finding ref(find OF for this image) file in frame
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(0)

    while 1:
        ret, frame = cap.read()
        # ret will return a true value if the frame exists otherwise False
        into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # changing the color format from BGr to HSV
        # This will be used to create the mask
        L_limit = np.array([98, 50, 50])  # setting the blue lower limit
        U_limit = np.array([139, 255, 255])  # setting the blue upper limit

        b_mask = cv2.inRange(into_hsv, L_limit, U_limit)
        # creating the mask using inRange() function
        # this will produce an image where the color of the objects
        # falling in the range will turn white and rest will be black
        blue = cv2.bitwise_and(frame, frame, mask=b_mask)
        # this will give the color to mask.
        cv2.imshow('Original', frame)  # to display the original frame
        cv2.imshow('Blue Detector', blue)  # to display the blue object output

        if cv2.waitKey(1) == 27:
            break
    # this function will be triggered when the ESC key is pressed
    # and the while loop will terminate and so will the program
    cap.release()

    cv2.destroyAllWindows()


if __name__=="__main__":
    pt1, pt2, pt3, pt4 = detectSameColorObjectFrame("../images/main_image_2.PNG")
    print("pt1/(x1, y1) -> ", pt1, " and pt2/(x1, y2) -> ", pt2, " and pt3/(x2, y1) -> ", pt3, " and pt4/(x2, y2) -> ", pt4)
