import os
import cv2
from jiwer import cer
import pytesseract
import numpy as np
import imutils

import re

# Define the path to tesseract.exe present in the Tesseract-OCR folder
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Configuration for tesseract
'''
Following are the arguments used in the tesseract config-

    -l: Specifies the language, over here "eng" refers to the english language used in the ocr
    -oem: This stands for OCR Engine Mode and for our implementation "Neural nets LSTM engine only" is used and hence its corresponding code of 1 is specified.
    -psm: PSM stands for Page Segmentation Mode and since license plate is a single uniform block of text, mode 6 is used. 

'''
config = ('-l eng --oem 1 --psm 6')

# Defining pattern for regular expression
# Pattern contails Uppercased characters and numbers ranging from 0-9

pattern = r'[A-Z0-9]+'


# Implement this function that will return a string representing what in seen in the given image
def get_plate_text(img) -> str:
    """
    Used to determine the characters on a license plate
    Args:
        img: The input image of license plate.
    Returns:
        Returns the OCR result of the input
    """
    
    # The input image is converted into a grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filtering is performed to reduce the noise along side preserving the edges.
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Calculating the lower and upper thresholds of canny edge detection by using the median value of the grayscale image.
    med_val = np.median(gray) 
    lower = int(max(0 ,0.7*med_val))
    upper = int(min(255,1.3*med_val))
    
    # Computing the edges
    edged = cv2.Canny(gray, lower, upper)
    
    # Finding the contours in the edged image
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    
    # Find the contours with 4 edges
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    
    # If there are contours in an image, the contour with the maximum information is cropped and is passed through a pytesseract for extracting the text.
    if screenCnt is not None:
        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        new_image = cv2.bitwise_and(img,img,mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]

        temp = pytesseract.image_to_string(Cropped, config=config)
        
        # The minimum number of character is set to by 4, if it is less than 4 the grayscale image after bilateral filtering is passed through the pytesseract rather than the cropped image.
        if len(temp) >= 4:
            text = temp
        else:
            text = pytesseract.image_to_string(gray, config=config)
    
    # If the image has no contours, the grayscale image after bilateral filtering is directly passed through the pytesseract.
    else:
        
        text = pytesseract.image_to_string(gray, config=config)
    
    
    # The resulting text output is checked for the defined regular expression pattern.
    matches = re.findall(pattern, text)
    merged_string = ''.join(matches)
    
    return merged_string


if __name__ == "__main__":
    # The image directory
    img_dir = "plate_images/"
    number_of_plates = 0
    error = 0.0
    # Loops through the given directory reading images in
    for filename in os.listdir(img_dir):
        # Loads the image feel free to do this using whatever library you like opencv as an example
        img = cv2.imread(os.path.join(img_dir, filename))
        # Gets the actual from the file name
        actual = filename.split(".")[0]
        guess = get_plate_text(img)
        # This is one way to get the error of the guess compared to actual feel free to evaluate your algorithm differently
        error += cer(actual, guess)
        number_of_plates += 1

    accuracy = (1 - error/number_of_plates) * 100
    print("Overall accuracy: ", accuracy)

