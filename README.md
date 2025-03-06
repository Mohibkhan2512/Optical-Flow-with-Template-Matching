## Optical-Flow-with-Template-Matching

## For any queries reach out to me on linkedin
<a href="https://www.linkedin.com/in/mohibkhan25/">  https://www.linkedin.com/in/mohibkhan25/ </a>

---

## Basics of Optical Flow 🔍 

Optical flow is a technique that estimates the motion of objects in a series of images. => <a href="https://nanonets.com/blog/optical-flow/">  Click here to know more... </a>


## Basics of Template Matching 🔍

Template matching is a technique in image processing that compares a template image to an image to find matching parts. => <a href="https://medium.com/analytics-vidhya/image-processing-template-matching-aac0c1cbe2c0">  Click here to know more... </a>

## Problem statement
Object detection tasks using traditional machine learning approaches can be very compute-intensive and are not feasible for classifying objects with different features belonging to the same class. The task of a machine learning algorithm is not only to identify objects in an image but also to detect the coordinates where the objects are present. Additionally, machine learning algorithms need incredibly high prediction times to predict objects in real-time videos. A limited number of labeled data also poses a significant hurdle in training and testing these algorithms.

It becomes impractical to use machine learning approaches for detecting objects with different traits belonging to the same class. For example, identifying a certain model of car in a real-time video based on car brand, car color, and car type requires a large labeled dataset for each category.

Advancements in computational power in recent years have allowed the field of computer vision to grow exponentially. Computer vision has enabled computers to replicate, and in some cases, outperform the task of identifying and processing objects in an image. Feature matching is one such application of computer vision used for object detection in images. However, this technique is also susceptible to object localization problems (detecting objects and their coordinates) when used on real-time video data.

## Proposed solution

**Object Detection and Tracking Using Feature Matching and Optical Flow**
The goal of this solution is to provide an alternative approach for object tracking and detection using feature-matching and optical-flow-based methods. In a nutshell, we identify interesting features using feature matching techniques and use optical flow to iteratively track the flow of these features in subsequent frames.

**Feature Detection Techniques**
Feature detection techniques help extract feature points from images. These feature points are mostly the salient features of an image, which have lower odds of being mislabeled as any other feature. For detecting objects of interest in videos/frames, we use:

**Template Matching**
ORB + FLANN Based Detector
Feature Matching
Feature matching is the process of substantiating similarity between two frames of the same scene/object based on similarity measures such as:

**Euclidean distance**
Absolute difference measure
Square difference measure
Cross-correlation measure
Normalized cross-correlation measure
Template Matching is an approach where the image of interest (referred to as the template) is searched in the entire image. Based on the measure of similarity, a decision is made if the template is present in the image or not. However, template matching fails to detect objects in the main image if there is any change in the orientation of the template image compared to how it should appear in the main image.

**Key-Point and Descriptor-Based Feature Point Extraction**
Key-point and descriptor-based feature point extraction techniques such as:

**ORB (Oriented FAST and Rotated BRIEF)
SIFT (Scale-Invariant Feature Transform)
SURF (Speeded-Up Robust Features)**
These techniques are invariant to rotation and scale. According to Ebrahim Karami et al., ORB is the fastest compared to SIFT and SURF. ORB is also more scale and rotation invariant compared to SIFT and SURF. When it comes to noisy images, SIFT and ORB induce similar performances.

**Descriptor Matching**
Once we have extracted the key-points and descriptors using ORB, the next step is to match the descriptors of two images (i.e., the image of interest with the object to be detected and the image from the video or camera stream). There are several techniques used for matching the descriptors of images, such as:

**Brute-Force Matcher**
**FLANN (Fast Library for Approximate Nearest Neighbors) Based Matcher**
FLANN-based matcher outperforms brute-force-based matcher.

**Tracking Feature Points Using Optical Flow**
The next step is to track the feature points extracted using template matching and ORB using the Lucas-Kanade optical flow pyramidal implementation devised by Bouguet. The output of template matching will be the (x, y) coordinates of the current image where the template was detected. These few points don’t provide sufficient information for tracking the object using optical flow. To overcome this issue, we add an additional step after template matching: the corner detector method proposed by Shi-Thomasi to extract rich features for the detected object for better tracking using the optical flow technique.

For effective flow estimation, the intensity of the detected features should be constant in the current and next frame. Additionally, the point and its neighbors should have similar, if not the same, motion. The technique detects the flow or motion of these points and their neighbors by identifying the change in location in the current and next frame.

Our approach keeps track of each feature point associated with the detected object for x past frames, where x is the number of frames the object was present in a video stream.

![image](https://github.com/user-attachments/assets/6d908c6f-a1df-40c9-b839-3c4afba53aaf)


![image](https://github.com/user-attachments/assets/189da953-8c02-4d3e-93cb-a54af390f7bb)


## Results

![image](https://github.com/user-attachments/assets/7e61c5a2-096e-4ab0-a4f4-5127f7b8b774)

![image](https://github.com/user-attachments/assets/6a858343-9ba6-4c36-9ce7-af512e2e5d65)

![image](https://github.com/user-attachments/assets/fd8bf885-6bb7-4794-af10-9187efc50857)

![image](https://github.com/user-attachments/assets/b61e3611-522e-401e-b755-3159a0852f9a)

## Proposed Solution Benefits and Limitations

The proposed solution is computationally inexpensive, easy to use, and modifiable for different use cases. It gives good results in cases of data scarcity. This solution can help mitigate the dependency on machine learning approaches for tasks such as object detection and object tracking. By updating the threshold values of template matching and ORB algorithms, the same solution can also be used for classifying objects belonging to the same class (e.g., identifying different types of objects such as cars, bikes, trucks, etc.).

However, there are a few limitations:

Results are not as highly accurate as some deep learning models based on segmentation techniques.
Template matching requires accurate templates from the scene as it is sensitive to occlusion, background changes, changes in illumination, and non-rigid transformations (e.g., image from a different angle or distortion).
ORB + FLANN requires that both images (object image and main image) have interesting features (i.e., corners, edges, lines, etc.).
Optical flow works best with real-time videos that have objects moving in slow motion and no frequent changes in the illumination of the scene.
