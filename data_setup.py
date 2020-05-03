import cv2

# Prepares the file for the model. Loads the specified image file in grayscale, and resizes to 28x28.
# Returns an array
def prepare(file):
    IMG_SIZE = 28
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array.reshape(IMG_SIZE, IMG_SIZE)

    return [img_array / 255.0]
