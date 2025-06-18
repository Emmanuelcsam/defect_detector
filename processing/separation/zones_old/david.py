import numpy as np
import cv2 as cv
from tkinter import filedialog
 
 
def get_image():
    image_object = filedialog.askopenfile()
    if image_object:
        image_path = image_object.name
        image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
        return image
    else:
        exit()
 
def preprocess_image():
    """
    This function takes in the raw greyscale image file,
    runs a contrast enhancement (CLAHE),
    and runs a canny filter to make the cirlces easier to see.
    """
   
    # apply contrast limited adaptive histogram equalization to masked image:
    img = get_image()
    assert img is not None, "file could not be read, check with os.path.exists()"
   
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
    clahe_image = clahe.apply(img)
    # cv.imshow("CLAHE'd & masked image", clahe_image)
   
    canny_image = cv.Canny(clahe_image, 100, 200, apertureSize= 3)
 
    # apply a gaussian blur to improve yeild of circle recognition
    img_blurred = cv.GaussianBlur(canny_image,(5, 5), 0)
    # cv.imshow("Image with Gaussian Blur", img_blurred)
 
 
    img_preprocessed: np.ndarray = img_blurred
 
    processed_dict = {
        "blurred_canny": img_preprocessed,
        "original_img": img          
    }
    return processed_dict
 
 
def circle_extract(image: np.ndarray, x0: int, y0: int, radius: int) -> np.ndarray:
    """Takes in a grayscale image"""
    arr = image
    rows, cols = arr.shape
 
    for i in range(rows):
        for j in range(cols):
            # test against the equation for a circle. Make radius slightly smaller to eliminate transition area.
            if np.sqrt(np.square(i - y0) + np.square(j - x0)) > (radius):
                arr[i][j] = 0
            else:
                pass
    output_image = arr
    return output_image
 
def core_mask(image: np.ndarray, x0: int, y0: int, radius: int) -> np.ndarray:
    """Takes in a grayscale image"""
    arr = image
    rows, cols = arr.shape
 
    for i in range(rows):
        for j in range(cols):
            # test against the equation for a circle. Make radius slightly smaller to eliminate transition area.
            if np.sqrt(np.square(i - y0) + np.square(j - x0)) <= (radius):
                arr[i][j] = 0
            else:
                pass
    output_image = arr
    return output_image
 
 
 
def run_analysis():
   
    preproc = preprocess_image()
    img = preproc["original_img"]
   
    proc_img = preproc["blurred_canny"]
 
 
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
   
    # Detects Cladding
    cladding_circle = cv.HoughCircles(proc_img, cv.HOUGH_GRADIENT,dp=1, minDist=10,
                             param1=50, param2=40, minRadius=100, maxRadius=500)
   
    cladding_x0 = int(cladding_circle[0][0][0])
    cladding_y0 = int(cladding_circle[0][0][1])
    cladding_radius = int(cladding_circle[0][0][2])
    print(cladding_x0, cladding_y0, cladding_radius)
 
    cladding = circle_extract(img, cladding_x0, cladding_y0, cladding_radius)
    cv.imshow("cropped cladding", cladding)
 
    cladding_copy = cladding.copy() # needs to create separate matrix for core extraction.
   
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
    clahe_cladding = clahe.apply(cladding)
    # cv.imshow("CLAHE'd cladding", clahe_cladding)
 
    canny_clad = cv.Canny(clahe_cladding, 100, 200)
   
    # cv.imshow("canny cladding", canny_clad)
    canny_clad_blurred = cv.GaussianBlur(canny_clad,(7, 7), 0)
 
    canny_clad_blurred_copy = canny_clad_blurred.copy()
    blurred_canny_color_copy = cv.cvtColor(canny_clad_blurred_copy, cv.COLOR_GRAY2BGR)
 
    # Detects Core
    core_circle = cv.HoughCircles(canny_clad_blurred, cv.HOUGH_GRADIENT,dp=1, minDist=10,
                          param1=50, param2=40, minRadius=20, maxRadius=80)
   
    if core_circle is None:
        print("could not sense core")
        cv.imshow("canny clad that cant see core", canny_clad_blurred)
 
        #alpha = 2.5
        #beta = 1
        #adjusted_cladding = cv.convertScaleAbs(cladding, alpha=alpha, beta=beta)
        #cv.imshow("brighter cladding", adjusted_cladding)
   
 
        clahe_adjusted_cladding = clahe.apply(cladding)
        cv.imshow("brighter CLAHE cladding", clahe_adjusted_cladding)
 
        new_cladding_canny = cv.Canny(clahe_adjusted_cladding, 100, 200)
        cv.imshow("new canny cladding", new_cladding_canny)
 
        new_canny_clad_blurred = cv.GaussianBlur(new_cladding_canny,(7, 7), 0)
        cv.imshow("new canny cladding blurred", new_canny_clad_blurred)
        cv.waitKey(0)
 
 
        core_circle = cv.HoughCircles(new_canny_clad_blurred, cv.HOUGH_GRADIENT,dp=1, minDist=10,
                    param1=50, param2=40, minRadius=20, maxRadius=60)
 
    cv.waitKey(0)
    core_x0 = int(core_circle[0][0][0])
    core_y0 = int(core_circle[0][0][1])
    core_radius = int(core_circle[0][0][2])
    print(core_x0, core_y0, core_radius)
 
    core = circle_extract(cladding, core_x0, core_y0, core_radius)
    cv.imshow("cropped core", core)
 
    # cladding_only = cv.bitwise_xor(cladding, core)
 
    cladding_only = core_mask(cladding_copy, core_x0, core_y0, core_radius)
    cv.imshow("cladding_only", cladding_only)
    #-------------------------- draw the circles on the originial image ----------------------------------#
    # draw the outer core circle
    cv.circle(cimg, (core_x0, core_y0), core_radius, (0, 255, 0), 2)
    # draw the center of the core circle
    cv.circle(cimg, (core_x0, core_y0), 2, (0, 0, 255), 3)
 
    # draw the outer cladding circle
    cv.circle(cimg, (cladding_x0, cladding_y0), cladding_radius, (255, 255, 0), 2)
    # draw the center of the cladding circle
    cv.circle(cimg, (cladding_x0, cladding_y0), 2, (0, 0, 255), 3)
 
    #-------------------------- draw the circles on the cropped cladding image ---------------------------#
    # draw the outer core circle (canny)
    cv.circle(blurred_canny_color_copy, (core_x0, core_y0), core_radius, (0, 255, 0), 2)
    # draw the center of the core circle (canny)
    cv.circle(blurred_canny_color_copy, (core_x0, core_y0), 2, (0, 0, 255), 3)
 
    # draw the outer cladding circle (canny)
    cv.circle(blurred_canny_color_copy, (cladding_x0, cladding_y0), cladding_radius, (255, 255, 0), 2)
    # draw the center of the cladding circle (canny)
    cv.circle(blurred_canny_color_copy, (cladding_x0, cladding_y0), 2, (0, 0, 255), 3)
 
 
 
    cv.imshow("output", cimg)
    cv.imshow("output (on canny)", blurred_canny_color_copy)
    cv.waitKey(0)
    cv.destroyAllWindows()
 
 
def main():
    pass
 
if __name__ == "__main__":
 
    run_analysis()
    exit()
 
 
 