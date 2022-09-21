# DogBreedClassifier

A dog breed classifier written in python using tensorflow. The model takes the input as a tensor with a shape of (224, 224, 3) and returns a numerical value coresponding to a certain breed. It uses the MobileNetV2 architecture from keras as a base on top of some convolutional layers. Results come from the output layer, which has 120 neurons that show the probability distribution of an image belonging to each of the 120 breeds the model recognizes. It has an average accuracy of %.
