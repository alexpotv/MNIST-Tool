## @file main.py
#  @author Alexandre Potvin-Demers
#  @date 05-05-2020
#  @brief Contains the main execution program for MNIST Tool.
#  @details Contains the main execution program for MNIST Tool. This is the CLI of the program,
#  allowing the user to create new models, test them on external data and save them.

import tensorflow as tf
import models
import data_setup

print("Welcome to the MNIST Tool!")
print("Would you like to create a new model (N), or use an existing model (E)? ")
newOrLoad = input()

if (newOrLoad == 'N'):
    print("Would you like to use the MNIST dataset (M) or the EMNIST dataset (E)?")
    datasetChoice = input()
    print("How many epochs would you like to run?")
    epochNumber = input()
    print("What is the batch size of your model?")
    batchSize = input()
    print("Enter a name for the model: ")
    modelName = input()

    if (datasetChoice == 'M'):
        createdModel = models.newModelMNIST(int(epochNumber))

        print("Created MNIST model with the following metrics:")
        print("Loss: ", createdModel[0])
        print("Accuracy: ", createdModel[1])
        createdModel[2].save("models/" + modelName + ".h5")
        currentModel = createdModel[2]
        print("Model saved as " + modelName + ".h5")
        print("This is the model's architecture: ")
        currentModel.summary()
    
    elif (datasetChoice == 'E'):
        createdModel = models.newModelEMNIST(int(epochNumber), int(batchSize))

        print("Created EMNIST model with the following metrics:")
        print("Loss: ", createdModel[0])
        print("Accuracy: ", createdModel[1])
        createdModel[2].save("models/" + modelName + ".h5")
        currentModel = createdModel[2]
        print("Model saved as " + modelName + ".h5")
        print("This is the model's architecture: ")
        currentModel.summary()

    else:
        print("Invalid.")

elif (newOrLoad == 'E'):
    print("Enter the name of the model to load: ")
    modelName = input()
    currentModel = tf.keras.models.load_model("models/" + modelName + ".h5")
    print("Model loaded successfully.")
    print("This is the model's architecture: ")
    currentModel.summary()

else:
    print("Invalid.")

print("Enter the name of the image file to evaluate: ")
imageName = input()

preparedImage = data_setup.prepareEMNIST("images/" + imageName)
predictionForImage = currentModel.predict([preparedImage[:1]])[0]
print("The best guess for this image is: ", predictionForImage.argmax())