import cv2
import numpy as np
from screeninfo import get_monitors



class stiching_const:
    IMAGE_11 = cv2.imread('C:\\Users\\user\\Pictures\\Camera Roll\\image11.jpg')
    IMAGE_12 = cv2.imread('C:\\Users\\user\\Pictures\\Camera Roll\\image12.jpg')

    IMAGE_21 = cv2.imread('C:\\Users\\user\\Pictures\\Camera Roll\\image21.jpg')
    IMAGE_22 = cv2.imread('C:\\Users\\user\\Pictures\\Camera Roll\\image22.jpg')

    IMAGE_31 = cv2.imread('C:\\Users\\user\\Pictures\\Camera Roll\\image31.jpg')
    IMAGE_32 = cv2.imread('C:\\Users\\user\\Pictures\\Camera Roll\\image32.jpg')

    IMAGE_41 = cv2.imread('C:\\Users\\user\\Pictures\\Camera Roll\\image41.jpg')
    IMAGE_42 = cv2.imread('C:\\Users\\user\\Pictures\\Camera Roll\\image42.jpg')

    X       = 0
    Y       = 1
    W       = 2
    H       = 3

class stiching_in_open_cv:

    def __init__(self, samples):
        # Load the images
        self.samples = sample
        self.set_sourse( self.samples)

    def step0(self):

        image1 = self.image1
        image2 = self.image2

        # Initialize the SIFT feature detector and extractor
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both images
        keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

        # Draw keypoints on the images
        image1_keypoints = cv2.drawKeypoints(image1, keypoints1, None)
        image2_keypoints = cv2.drawKeypoints(image2, keypoints2, None)

        # Display the images with keypoints
        cv2.imshow('Image 1 with Keypoints', image1_keypoints)
        cv2.imshow('Image 2 with Keypoints', image2_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def step1(self):

        image1 = self.image1
        image2 = self.image2

        # Initialize the feature detector and extractor (e.g., SIFT)
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both images
        keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

        # Initialize the feature matcher using brute-force matching
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match the descriptors using brute-force matching
        matches_bf = bf.match(descriptors1, descriptors2)

        # Sort the matches by distance (lower is better)
        matches_bf = sorted(matches_bf, key=lambda x: x.distance)

        # Draw the top N matches
        num_matches = 50
        image_matches_bf = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches_bf[:num_matches], None)

        # Initialize the feature matcher using FLANN matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Match the descriptors using FLANN matching
        matches_flann = flann.match(descriptors1, descriptors2)

        # Sort the matches by distance (lower is better)
        matches_flann = sorted(matches_flann, key=lambda x: x.distance)

        # Draw the top N matches
        image_matches_flann = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches_flann[:num_matches], None)

        # Display the images with matches
        win_nm = 'Brute-Force Matching'
        cv2.namedWindow(win_nm, cv2.WINDOW_NORMAL)

        coor = self.get_size(4)
        cv2.resizeWindow(win_nm, coor[stiching_const.W], coor[stiching_const.Y])
        cv2.moveWindow(win_nm, coor[stiching_const.X], coor[stiching_const.Y])
        cv2.imshow('Brute-Force Matching', image_matches_bf)

        # locate img
        win_nm = 'FLANN Matching'
        cv2.namedWindow(win_nm, cv2.WINDOW_NORMAL)

        coor = self.get_size(4)
        cv2.resizeWindow(win_nm, coor[stiching_const.W], coor[stiching_const.Y])
        cv2.moveWindow(win_nm, coor[stiching_const.X], coor[stiching_const.Y])
        cv2.imshow(win_nm, image_matches_flann)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def step2(self):

        image1 = self.image1
        image2 = self.image2

        # Convert the images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Initialize the feature detector and extractor (e.g., SIFT)
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both images
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        # Initialize the feature matcher using brute-force matching
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match the descriptors using brute-force matching
        matches = bf.match(descriptors1, descriptors2)

        # Extract the matched keypoints
        src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate the homography matrix using RANSAC
        homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        # Print the estimated homography matrix
        print("Estimated Homography Matrix:")
        print(homography)


    def step3(self):

        image1 = self.image1
        image2 = self.image2

        # Convert images to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Initialize the feature detector and extractor (e.g., SIFT)
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both images
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        # Initialize the feature matcher using brute-force matching
        bf = cv2.BFMatcher()

        # Match the descriptors using brute-force matching
        matches = bf.match(descriptors1, descriptors2)

        # Select the top N matches
        num_matches = 50
        matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

        # Extract matching keypoints
        src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

        # Estimate the homography matrix
        homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        # Warp the first image using the homography
        result = cv2.warpPerspective(self.image1, homography, (self.image2.shape[1], self.image2.shape[0]))

        win_nm = 'Result Image'
        cv2.namedWindow( win_nm, cv2.WINDOW_NORMAL)

        # locate img
        coor = self.get_size( 4 )
        cv2.resizeWindow( win_nm, coor[stiching_const.W], coor[stiching_const.Y] )
        cv2.moveWindow( win_nm, coor[stiching_const.X], coor[stiching_const.Y]   )

        # show img
        cv2.imshow( win_nm , result )

        # Blending the warped image with the second image using alpha blending
        alpha = 0.5  # blending factor
        blended_image = cv2.addWeighted(result, alpha, self.image2, 1 - alpha, 0)

        # Display the blended image
        # set nm
        win_nm = 'Blended Image'
        cv2.namedWindow( win_nm, cv2.WINDOW_NORMAL)

        # locate img
        coor = self.get_size( 3 )
        cv2.resizeWindow( win_nm, coor[stiching_const.W], coor[stiching_const.Y] )
        cv2.moveWindow( win_nm, coor[stiching_const.X], coor[stiching_const.Y]   )

        # show img
        cv2.imshow( win_nm , blended_image )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_source( self , win_title_prefix = 'img '):

        image_source  =  [ self.image1, self.image2 ]
        cnt           =  1

        for item in image_source :

            # set nm
            win_nm = win_title_prefix + str( cnt )
            cv2.namedWindow( win_nm, cv2.WINDOW_NORMAL)

            # locate img
            coor = self.get_size( cnt )
            cv2.resizeWindow( win_nm, coor[stiching_const.W], coor[stiching_const.H] )
            cv2.moveWindow( win_nm, coor[stiching_const.X], coor[stiching_const.Y]   )

            # show img
            cv2.imshow( win_nm , image_source[cnt-1] )
            cnt = cnt + 1

    def get_size(self, numb ):
        monitor = get_monitors()[0]
        x = 0
        y = 0
        w = int(monitor.width / 2)
        h = int(monitor.height / 2)

        # Зміна розміру вікна
        if (numb == 2):
            x = w + 10
        elif (numb == 3):
            y = h + 10
        elif (numb == 4):
            x = w + 10
            y = h + 10
        elif (numb==5):
            y = h + 10
            w = int(monitor.width )
        return (x,y,w,h)


    def set_sourse(self, samples):
        self.image1 = cv2.imread(
            "C:\\Users\\user\\Pictures\\Camera Roll\\image" + str(samples) + "1.jpg")  # stiching_const.IMAGE_1
        self.image2 = cv2.imread(
            "C:\\Users\\user\\Pictures\\Camera Roll\\image" + str(samples) + "2.jpg")  # stiching_const.IMAGE_2


#iterations = [1,2,3,4]
#for iteration in iterations:

sample = int ( input("Enter samples for stiching [0-exit,1..4]  =>" ) )

while sample != 0 :

    obj = stiching_in_open_cv(sample)

    print(                      "Step of stitching:")
    print(                      "1. Feature Detection (SIFT,SURF,ORB ")
    print(                      "2. Feature matching")
    print(                      "3. Blending")


    step = input(               "Enter process what you want watch [1..3] =>")

    obj.show_source()

    if step == "1":
        obj.step1()
    if step == "2":
        obj.step2()
    if step == "3":
        obj.step3()

    print("")
    print("")
    print("")
    print("")
    print("")
    sample = int(input("Enter samples for stitching [0-exit,1..4]  =>"))

