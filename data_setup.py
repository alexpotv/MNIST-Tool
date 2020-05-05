## @package data_setup
#  @author Alexandre Potvin-Demers
#  @date 04-05-2020
#  @brief Contains functions which help format external files in order to match the criteria of the
#  different datasets.
#  @details Contains functions which help format external files in order to match the criteria of
#  the different datasets. It transforms user-made files or external files in a format that can be
#  interpreted successfully by the trained models of the program, following the constraints of the
#  datasets used to train the models.

import cv2
from scipy import ndimage
import math
import numpy as np

# Prepares the file for the model. Loads the specified image file in grayscale, and resizes to 28x28.
## @function prepare
#  @brief Prepares a file for its interpretation by a MNIST model.
#  @details Prepares a file for its interpretation by a MNIST model. The function loads the
#  specified file as a grayscale image, resizes it to fit the 28x28 criteria, removes the black rows
#  and columns around the actual pixels representing the drawn number, shifts the image according to
#  the center of mass and fits it inside a 20x20, centered in the original 28x28 image.
#  @param file The local path to the number image file to prepare
#  @returns Returns the modified image as a numpy array, ready for interpretation
def prepare(file):

    ## @function getBestShift
    #  @brief Calculates the shift of an image according to its center of mass.
    #  @details Calculates the shift of an image according to its center of mass, and gets the
    #  coordinates in which to shift the image to center it according to said center of mass.
    #  @param img The image to analyse and determinate shift coordinates
    #  @returns Returns a tuple of the x and y coordinates to shift the image to.
    def getBestShift(img):
        cy,cx = ndimage.measurements.center_of_mass(img)

        rows,cols = img.shape
        shiftx = np.round(cols/2.0-cx).astype(int)
        shifty = np.round(rows/2.0-cy).astype(int)

        return shiftx,shifty


    ## @function shift
    #  @brief Shifts the image provided according to the coordinates provided.
    #  @details Shifts the image provided according to the coordinates provided.
    #  @param img The image to shift
    #  @param sx The x coordinate to shift the image to
    #  @param sy The y coordinate to shift the image to
    #  @returns Returns the shifted image.
    def shift(img,sx,sy):
        rows,cols = img.shape
        M = np.float32([[1,0,sx],[0,1,sy]])
        shifted = cv2.warpAffine(img,M,(cols,rows))
        return shifted 


    IMG_SIZE = 28
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    while np.sum(img_array[0]) == 0:
        img_array = img_array[1:]
    
    while np.sum(img_array[:,0]) == 0:
        img_array - np.delete(img_array, 0, 1)
    
    while np.sum(img_array[-1]) == 0:
        img_array = img_array[:-1]
    
    while np.sum(img_array[:, -1]) == 0:
        img_array = np.delete(img_array, -1, 1)
    
    rows, cols = img_array.shape

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

    shiftx,shifty = getBestShift(img_array)
    shifted = shift(img_array,shiftx,shifty)
    img_array = shifted

    img_array.reshape(IMG_SIZE, IMG_SIZE)

    return [img_array / 255.0]
