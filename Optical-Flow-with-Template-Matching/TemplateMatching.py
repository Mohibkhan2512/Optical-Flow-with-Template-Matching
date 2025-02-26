import cv2 as cv
import numpy as np

def detectObjectInFrame(image, template_img):
    # Template matching for finding ref(find OF for this image) file in frame
    img_rgb = cv.imread(image)
    # assert img_rgb is not None, "file could not be read, check with os.path.exists()"
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    template = cv.imread(template_img, cv.IMREAD_GRAYSCALE)
    # assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    print("width: ", w, " and height: ", h)
    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    # res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    # res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    # res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    print(res.shape)  # output (1032, 1081)
    if res is None:
        print("Template not found in image...")
    else:
        print("Template found in image...")
    # threshold = 0.8
    threshold = 0.6
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        print("pt is -> ", pt)
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
    cv.imshow("Template image ", img_rgb)
    cv.waitKey()
    # cv.imwrite('../images/res_2.png', img_rgb)
    return pt[0], pt[1], pt[0] + w, pt[1] + h, w, h

# def multiObjectTracking(image, template_img):
#     colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#     results = []
#     # Template matching for finding ref(find OF for this image) file in frame
#     img_rgb = cv.imread(image)
#     # assert img_rgb is not None, "file could not be read, check with os.path.exists()"
#     img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
#     template = cv.imread(template_img, cv.IMREAD_GRAYSCALE)
#     # assert template is not None, "file could not be read, check with os.path.exists()"
#     w, h = template.shape[::-1]
#     print("width: ", w, " and height: ", h)
#     res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
#     if res is None:
#         print("Template not found in image...")
#     else:
#         print("Template found in image...")
#     res[res < 0.9] = 0
#     results.append(res)
#     print(res.shape)  # output (1032, 1081)
#
#     maximum_values_array = np.maximum(*results)
#     maximum_value_contains_array = np.array(results).argmax(axis=0)
#     print(maximum_values_array.shape)
#     print(maximum_value_contains_array.shape)
#
#     for i in range(len(maximum_values_array)):
#         for j in range(len(maximum_values_array[i])):
#             if maximum_values_array[i][j] > 0:
#                 print(maximum_values_array[i][j])
#                 colour = colours[maximum_value_contains_array[i][j]]
#                 top_lect = (j, i)
#                 bottom_right = (j + width, i + height)
#                 cv2.rectangle(img, top_lect, bottom_right, colour, 2)
#     cv2_imshow(img)
#     cv2.imwrite('output.png', img)

if __name__=="__main__":
    pt1, pt2, pt3, pt4, w, h = detectObjectInFrame("../images/main_image_swift_car.PNG", "../images/template_image_swift_car_rect.PNG")
    x1 = pt1
    y1 = pt2
    x2 = x1 + w
    y2 = y1 + h

    print("pt1/(x1, y1) -> ", pt1, " and pt2/(x1, y2) -> ", pt2, " and pt3/(x2, y1) -> ", pt3, " and pt4/(x2, y2) -> ", pt4)
    print("(x1, y1) -> ", x1, y1, " and (x1, y2) -> ", x1, y2, " and (x2, y1) -> ", x2, y1, " and (x2, y2) -> ",
          x2, y2)
    # multipoint("../images/res_first_frame.PNG", "../images/template_image_main.PNG")