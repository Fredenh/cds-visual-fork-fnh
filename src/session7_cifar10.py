
# generic tools
import numpy as np
import cv2
# tools from sklearn
from sklearn.preprocessing import LabelBinarizer 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

# matplotlib
import matplotlib.pyplot as plt

def load_data():
    print("loading data")
    # loading data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # assigning labels
    my_labels = ["airplane", 
          "automobile", 
          "bird", 
          "cat", 
          "deer", 
          "dog", 
          "frog", 
          "horse", 
          "ship", 
          "truck"]
    
    return X_train, X_test, y_train, y_test

def greyscale(data_train, data_test):
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data_test])
    X_test_scaled = (X_test_grey)/255.0 
    X_train_scaled = (X_train_grey)/255.0
    return X_train_scaled, X_test_scaled

def reshaping(X_train_scaled, X_test_scaled):
    nsamples, nx, ny = X_train_scaled.shape 
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))
    return X_train_dataset, X_test_dataset

def label_conversion(label_train, label_test):
    print("doing label conversion")
    # convert labels to one-hot encoding
    lb = LabelBinarizer() # one-hot encoding = turning a string into a vector of numbers. All the possible labels as vectors of numbers
    y_train = lb.fit_transform(label_train)
    y_test = lb.fit_transform(label_test)
    return y_train, y_test, lb                 


def model_architecture():
    print("defining the architecture of the model")
    # define architecture 784x256x128x10
    model = Sequential() # a sequential model or "feed forward neural network"
    model.add(Dense(256, # adding a dense layer because its fully connected
                    input_shape=(1024,), # the input shape is dependant on the size of the input data
                    activation="relu")) # the type of activation function used. Then we avoid vanishing 
    model.add(Dense(128, # adding the second hidden layer which takes 128 nodes with a relu activation function
                    activation="relu"))
    model.add(Dense(10, # last layer / output layer
                    activation="softmax")) #function predicting 1 or 0. A generalized version of a logistic regression
    model_summary = model.summary()
    print(model_summary)
    return model

def train_model(nn_model):
    print("training the model")
    # train model using SGD
    sgd = SGD(0.01) # stocastic grading descent = the learning rate of 0.01. The higher the value, the quicker its gonna learn. If it tries to learn too quickly it might overshoot and not find the most optimal way of solving a problem
    nn_model.compile(loss="categorical_crossentropy", # .compile (makes it a computational graph structure)
                optimizer=sgd, # we use the sgd previously defined
                metrics=["accuracy"]) # metric were trying to improve is accuracy. it can also be recall, precision or f1 score
    return nn_model

def model_history(nn_model, data_train, label_train):
    print("creating model history")
    # train model
    history = nn_model.fit(data_train, label_train,  #creating history which will be beneficial later
                        epochs=10, # number of epochs
                        batch_size=32)
    return history

def clas_matrix(nn_model, data_test, label_test, lb):
    print("[INFO] evaluating network...")
    predictions = nn_model.predict(data_test, batch_size=32)

    print(classification_report(label_test.argmax(axis=1), # for everything in y-test give us the biggest value of that array
                                predictions.argmax(axis=1), 
                                target_names=[str(x) for x in lb.classes_]))

def main():
    X_train, X_test, y_train, y_test = load_data()
    X_train_scaled, X_test_scaled = greyscale(X_train, X_test)
    X_train_dataset, X_test_dataset = reshaping(X_train_scaled, X_test_scaled)
    y_train, y_test, lb = label_conversion(y_train, y_test)
    model = model_architecture()
    nn_model = train_model(model)
    history = model_history(nn_model, X_train_dataset, y_train)
    clas_matrix(nn_model, X_test_dataset, y_test, lb)
    
if __name__ == "__main__":
    main()
