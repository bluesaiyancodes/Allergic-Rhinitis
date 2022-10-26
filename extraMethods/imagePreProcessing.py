import cv2
import imutils
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt



def cleanImage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _ ,thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 300, param1 = 250, param2 = 10, minRadius = 100, maxRadius = 1000)
    
    x, y, r = list(map(int, circles[0][0]))

    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (x,y), r, (255,255,255), -1)
    # Apply mask
    res = np.zeros_like(img)
    res[(mask > 0)] = img[(mask > 0)]
    return res

def cleanImage2(orig_im):

    im = orig_im.copy()

    h, w = im.shape[0], im.shape[1]

    # Seed points for floodFill (use two points at each corner for improving robustness)
    seedPoints = ((0, 0), (10, 10), (w-1, 0), (w-1, 10), (0, h-1), (10, h-1), (w-1, h-1), (w-10, h-10))
    
    # Fill background with black color
    for seed in seedPoints:
        cv2.floodFill(im, None, seedPoint=seed, newVal=(0, 0, 0), loDiff=(0, 0, 0), upDiff=(18, 18, 18))
    
    # Use "close" morphological operation
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)));
    
    #Convert to Grayscale, and then to binary image.
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    ret, thresh_gray = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    #Find contours
    contours, _ = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea) # Get the largest contour

    # Smooth contour
    # https://agniva.me/scipy/2016/10/25/contour-smoothing.html
    x,y = c.T
    x = x.tolist()[0]
    y = y.tolist()[0]
    tck, u = splprep([x,y], u=None, s=1.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 20)
    x_new, y_new = splev(u_new, tck, der=0)
    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
    smoothened = np.asarray(res_array, dtype=np.int32)
    
    # To view Image with detected contour
    test_im = orig_im.copy()
    cv2.drawContours(test_im, [smoothened], 0, (0, 255, 0), 1)

    # Build a mask
    mask = np.zeros_like(thresh_gray)
    cv2.drawContours(mask, [smoothened], -1, 255, -1)

    # Apply mask
    res = np.zeros_like(orig_im)
    res[(mask > 0)] = orig_im[(mask > 0)]
    
    return res


def cropImage(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
    #cv2.imwrite("otsu.png", thresholded)
    bbox = cv2.boundingRect(thresholded)
    x, y, w, h = bbox
    #print(bbox)
    croppedImg = img[y:y+h, x:x+w]
    return croppedImg

def addContours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    edged = cv2.Canny(blur, 10, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)   
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    #(cnts, _) = imutils.contours.sort_contours(cnts)
    cv2.drawContours(image, cnts, -1, (0,255,0), 1)
    return image

def resizeImage(img, size=(28,28)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


def savePlot(self, image, text = ""):
    plt.figure()
    plt.imshow(image)
    figName = "[custom]plot-" + text +"-"+datetime.now().strftime('%H-%M-%S')
    plt.savefig(figName)