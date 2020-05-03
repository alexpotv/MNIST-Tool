import cv2
from scipy import ndimage
import math
import numpy as np

# Returns the coordinates of the center of mass in order to shift the image
def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

# Shifts the image in parameter according to the coordinates given
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted 

# Prepares the file for the model. Loads the specified image file in grayscale, and resizes to 28x28.
def prepare(file):
    IMG_SIZE = 28
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    # Removing the outer rows and column which are only black
    while np.sum(img_array[0]) == 0:
        img_array = img_array[1:]
    
    while np.sum(img_array[:,0]) == 0:
        img_array - np.delete(img_array, 0, 1)
    
    while np.sum(img_array[-1]) == 0:
        img_array = img_array[:-1]
    
    while np.sum(img_array[:, -1]) == 0:
        img_array = np.delete(img_array, -1, 1)
    
    rows, cols = img_array.shape

    # Resize to a 20x20 format
    if rows > cols:
      factor = 20.0/rows
      rows = 20
      cols = int(round(cols*factor))
      img_array = cv2.resize(img_array, (cols,rows))
    else:
      factor = 20.0/cols
      cols = 20
      rows = int(round(rows*factor))
      img_array = cv2.resize(img_array, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    img_array = np.lib.pad(img_array,(rowsPadding,colsPadding),'constant')

    # Shift the 20x20 image according to the center of mass
    shiftx,shifty = getBestShift(img_array)
    shifted = shift(img_array,shiftx,shifty)
    img_array = shifted

    img_array.reshape(IMG_SIZE, IMG_SIZE)

    return [img_array / 255.0]
